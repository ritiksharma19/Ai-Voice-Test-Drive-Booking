"""
booking/service.py
Test drive and sales meeting bookings.

  • Slots: BUSINESS_HOURS on BUSINESS_DAYS in BOOKING_SLOT_MINUTES steps, per
    showroom, with TEST_DRIVE_CAPACITY / MEETING_CAPACITY bookings per slot.
  • Validation happens here, never in the LLM: Indian mobile number (Indic
    digits accepted), known car model and showroom (fuzzy-matched), a future
    slot inside opening hours and within BOOKING_MAX_DAYS_AHEAD.
  • Storage: SQLite (BOOKING_DB_PATH); the capacity check and the insert run
    under one lock, so two callers cannot take the last seat in a slot.
  • Repeating the same booking (same phone, kind, slot) returns the existing
    booking instead of creating a duplicate.
  • BOOKING_WEBHOOK_URL (optional) receives every new booking as JSON, e.g.
    a CRM, Zapier / Make, or a Google Apps Script that appends to a Sheet.

Every result is a JSON-safe dict written for the LLM to read aloud: on errors
it says which field is wrong and offers alternative slots.
"""
from __future__ import annotations

import asyncio
import difflib
import re
import secrets
import sqlite3
import threading
from contextlib import closing
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from config.logging_config import get_logger
from config.settings import Settings
from core.privacy import indian_mobile

logger = get_logger("booking")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_ID_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"   # no 0/O, 1/I/L: easy to read aloud
_KINDS = {"test_drive": "TD", "meeting": "MT"}
_KIND_ALIASES = {"test drive": "test_drive", "testdrive": "test_drive", "test-drive": "test_drive",
                 "td": "test_drive", "sales_meeting": "meeting", "consultation": "meeting",
                 "sales meeting": "meeting", "appointment": "meeting"}
_TIME_RE = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)?\s*$", re.IGNORECASE)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bookings (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    customer_name TEXT NOT NULL,
    phone TEXT NOT NULL,
    car_model TEXT,
    showroom TEXT NOT NULL,
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    notes TEXT,
    language TEXT,
    session_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bookings_slot ON bookings (showroom, date, time, kind, status);
"""


class BookingError(Exception):
    def __init__(self, error: str, message: str, **extra) -> None:
        super().__init__(message)
        self.result = {"status": "error", "error": error, "message": message, **extra}


class BookingService:
    def __init__(self, settings: Settings,
                 clock: Callable[[], datetime] | None = None) -> None:
        self.s = settings
        self.tz = ZoneInfo(settings.business_timezone)
        self._clock = clock or (lambda: datetime.now(self.tz))
        opening, closing_ = settings.business_hours.split("-")
        self.open_time = _parse_hhmm(opening)
        self.close_time = _parse_hhmm(closing_)
        self.slot = timedelta(minutes=settings.booking_slot_minutes)
        self.days = {d[:3] for d in settings.business_days} or set(_WEEKDAYS)
        path = Path(settings.booking_db_path)
        self.db_path = path if path.is_absolute() else _PROJECT_ROOT / path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._background: set[asyncio.Task] = set()
        with closing(self._connect()) as db:
            db.executescript(_SCHEMA)
        logger.info("Bookings | db=%s hours=%s slot=%dmin showrooms=%s webhook=%s",
                    self.db_path, settings.business_hours, settings.booking_slot_minutes,
                    ", ".join(settings.showrooms), "on" if settings.booking_webhook_url else "off")

    # ── public ────────────────────────────────────────────────────────────────

    def now(self) -> datetime:
        return self._clock().astimezone(self.tz)

    async def execute(self, action: dict, session_id: str = "", language: str = "en") -> dict:
        """Run one LLM action; never raises."""
        name = str(action.get("name", "")).strip()
        try:
            if name == "check_availability":
                return await asyncio.to_thread(self.check_availability, action)
            if name == "book_appointment":
                result = await asyncio.to_thread(self.book, action, session_id, language)
                if result.get("status") == "confirmed" and not result.get("already_booked"):
                    self._post_webhook(result["booking_id"])
                return result
            return {"status": "error", "error": "unknown_action",
                    "message": "Use check_availability or book_appointment."}
        except BookingError as exc:
            return exc.result
        except Exception as exc:
            logger.exception("Booking action %s failed: %r", name, exc)
            return {"status": "error", "error": "system_error",
                    "message": "The booking system is unavailable right now. Apologise and "
                               "offer that the showroom will call the customer back."}

    def check_availability(self, action: dict) -> dict:
        kind = self._kind(action.get("kind", "test_drive"))
        day = self._date(action.get("date"))
        self._check_open_day(day)
        showrooms = ([self._showroom(action["showroom"])] if action.get("showroom")
                     else self.s.showrooms)
        availability = {room: self.free_times(kind, room, day) for room in showrooms}
        return {"status": "ok", "kind": kind, "date": day.isoformat(),
                "weekday": day.strftime("%A"), "available_times": availability,
                "note": "Times are slot start times (24-hour). Empty list = fully booked."}

    def book(self, action: dict, session_id: str = "", language: str = "en") -> dict:
        kind = self._kind(action.get("kind"))
        fields: dict[str, str] = {}
        name = str(action.get("customer_name", "")).strip()
        if not 2 <= len(name) <= 60:
            fields["customer_name"] = "ask for the customer's name"
        phone = self._phone(action.get("phone", ""))
        if not phone:
            fields["phone"] = "must be a 10-digit Indian mobile number starting with 6, 7, 8 or 9"
        car_model = self._car_model(action.get("car_model", ""), required=kind == "test_drive",
                                    fields=fields)
        showroom = ""
        try:
            showroom = self._showroom(action.get("showroom", ""))
        except BookingError as exc:
            fields["showroom"] = exc.result["message"]
        slot_start = None
        try:
            slot_start = self._slot(action.get("date"), action.get("time"))
        except BookingError as exc:
            fields["date_time"] = exc.result["message"]
        if fields:
            result = {"status": "error", "error": "missing_or_invalid", "fields": fields,
                      "message": "Ask the customer only for the fields listed, then try again."}
            if "date_time" in fields and showroom:
                result["alternatives"] = self._alternatives(kind, showroom, action.get("date"))
            return result

        day, hhmm = slot_start.date().isoformat(), slot_start.strftime("%H:%M")
        notes = str(action.get("notes", "")).strip()[:300]
        with self._lock, closing(self._connect()) as db, db:
            existing = db.execute(
                "SELECT * FROM bookings WHERE phone=? AND kind=? AND date=? AND time=? "
                "AND status='confirmed'", (phone, kind, day, hhmm)).fetchone()
            if existing:
                return {**self._public(dict(existing)), "already_booked": True}
            taken = db.execute(
                "SELECT COUNT(*) FROM bookings WHERE showroom=? AND date=? AND time=? "
                "AND kind=? AND status='confirmed'", (showroom, day, hhmm, kind)).fetchone()[0]
            if taken >= self._capacity(kind):
                raise BookingError(
                    "slot_full", f"{hhmm} on {day} at {showroom} is fully booked.",
                    alternatives=self._alternatives(kind, showroom, day, db=db))
            row = {
                "id": f"{_KINDS[kind]}-{''.join(secrets.choice(_ID_ALPHABET) for _ in range(5))}",
                "kind": kind, "status": "confirmed", "customer_name": name, "phone": phone,
                "car_model": car_model, "showroom": showroom, "date": day, "time": hhmm,
                "notes": notes, "language": language, "session_id": session_id,
                "created_at": self.now().isoformat(timespec="seconds"),
            }
            db.execute(f"INSERT INTO bookings ({', '.join(row)}) VALUES "
                       f"({', '.join('?' * len(row))})", tuple(row.values()))
        logger.info("📅 Booked %s %s %s %s %s", row["id"], kind, showroom, day, hhmm)
        return self._public(row)

    def free_times(self, kind: str, showroom: str, day: date,
                   db: sqlite3.Connection | None = None) -> list[str]:
        earliest = self.now() + timedelta(minutes=self.s.booking_min_lead_minutes)
        own = db is None
        db = db or self._connect()
        try:
            counts = dict(db.execute(
                "SELECT time, COUNT(*) FROM bookings WHERE showroom=? AND date=? AND kind=? "
                "AND status='confirmed' GROUP BY time", (showroom, day.isoformat(), kind)).fetchall())
        finally:
            if own:
                db.close()
        cap = self._capacity(kind)
        return [t.strftime("%H:%M") for t in self._grid(day)
                if t >= earliest and counts.get(t.strftime("%H:%M"), 0) < cap]

    def list_bookings(self, day: str | None = None) -> list[dict]:
        with closing(self._connect()) as db:
            if day:
                rows = db.execute("SELECT * FROM bookings WHERE date=? ORDER BY time", (day,))
            else:
                rows = db.execute("SELECT * FROM bookings ORDER BY date, time")
            return [dict(r) for r in rows.fetchall()]

    # ── validation helpers ────────────────────────────────────────────────────

    def _kind(self, value) -> str:
        kind = str(value or "").strip().lower()
        kind = _KIND_ALIASES.get(kind, kind)
        if kind not in _KINDS:
            raise BookingError("invalid_kind", "kind must be 'test_drive' or 'meeting'.")
        return kind

    @staticmethod
    def _phone(value) -> str:
        return indian_mobile(value)

    def _car_model(self, value, required: bool, fields: dict) -> str:
        text = str(value or "").strip()
        if not text:
            if required:
                fields["car_model"] = f"ask which car: {', '.join(self.s.car_models)}"
            return ""
        if not self.s.car_models:
            return text
        match = _fuzzy(text, self.s.car_models)
        if not match:
            fields["car_model"] = (f"'{text}' is not a model we sell; we offer "
                                   f"{', '.join(self.s.car_models)}")
        return match

    def _showroom(self, value) -> str:
        text = str(value or "").strip()
        if not text:
            if len(self.s.showrooms) == 1:
                return self.s.showrooms[0]
            raise BookingError("missing_showroom",
                               f"ask which showroom: {', '.join(self.s.showrooms)}")
        match = _fuzzy(text, self.s.showrooms)
        if not match:
            raise BookingError("invalid_showroom",
                               f"unknown showroom; choose one of {', '.join(self.s.showrooms)}")
        return match

    def _date(self, value) -> date:
        text = str(value or "").strip().lower()
        today = self.now().date()
        if text in ("today", "aaj"):
            return today
        if text in ("tomorrow", "kal"):
            return today + timedelta(days=1)
        try:
            day = date.fromisoformat(text)
        except ValueError:
            raise BookingError("invalid_date", "date must be YYYY-MM-DD") from None
        if day < today:
            raise BookingError("past_date", f"{day.isoformat()} is in the past; today is "
                                            f"{today.isoformat()}")
        if day > today + timedelta(days=self.s.booking_max_days_ahead):
            raise BookingError("too_far", f"bookings open only {self.s.booking_max_days_ahead} "
                                          f"days ahead (until "
                                          f"{(today + timedelta(days=self.s.booking_max_days_ahead)).isoformat()})")
        return day

    def _check_open_day(self, day: date) -> None:
        if _WEEKDAYS[day.weekday()] not in self.days:
            raise BookingError("closed_day", f"the showroom is closed on {day.strftime('%A')}s")

    def _slot(self, day_value, time_value) -> datetime:
        day = self._date(day_value)
        self._check_open_day(day)
        m = _TIME_RE.match(str(time_value or ""))
        if not m:
            raise BookingError("invalid_time", "time must be HH:MM (24-hour)")
        hour, minute, ampm = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower()
        if ampm.startswith("p") and hour < 12:
            hour += 12
        elif ampm.startswith("a") and hour == 12:
            hour = 0
        if hour > 23 or minute > 59:
            raise BookingError("invalid_time", "time must be HH:MM (24-hour)")
        start = datetime.combine(day, time(hour, minute), self.tz)
        if start not in self._grid(day):
            last = (datetime.combine(day, self.close_time) - self.slot).strftime("%H:%M")
            raise BookingError("outside_hours", f"slots start every "
                                                f"{self.s.booking_slot_minutes} minutes from "
                                                f"{self.open_time.strftime('%H:%M')} to {last}")
        if start < self.now() + timedelta(minutes=self.s.booking_min_lead_minutes):
            raise BookingError("too_soon", f"the earliest bookable time is "
                                           f"{self.s.booking_min_lead_minutes} minutes from now")
        return start

    def _grid(self, day: date) -> list[datetime]:
        t = datetime.combine(day, self.open_time, self.tz)
        end = datetime.combine(day, self.close_time, self.tz)
        out = []
        while t + self.slot <= end:
            out.append(t)
            t += self.slot
        return out

    def _alternatives(self, kind: str, showroom: str, day_value,
                      db: sqlite3.Connection | None = None) -> list[str]:
        """Up to 3 free slots on the requested day, else on the next open days."""
        try:
            start = self._date(day_value)
        except BookingError:
            start = self.now().date()
        out: list[str] = []
        for offset in range(8):
            day = start + timedelta(days=offset)
            if _WEEKDAYS[day.weekday()] not in self.days:
                continue
            out += [f"{day.isoformat()} {t}" for t in self.free_times(kind, showroom, day, db)]
            if len(out) >= 3:
                break
        return out[:3]

    def _capacity(self, kind: str) -> int:
        return self.s.test_drive_capacity if kind == "test_drive" else self.s.meeting_capacity

    # ── storage / webhook ────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def _public(self, row: dict) -> dict:
        day = date.fromisoformat(row["date"])
        return {"status": "confirmed", "booking_id": row["id"], "kind": row["kind"],
                "customer_name": row["customer_name"], "phone_last4": row["phone"][-4:],
                "car_model": row["car_model"] or "", "showroom": row["showroom"],
                "date": row["date"], "weekday": day.strftime("%A"), "time": row["time"]}

    def _post_webhook(self, booking_id: str) -> None:
        if not self.s.booking_webhook_url:
            return

        async def send() -> None:
            from core.http import get_http_client
            rows = await asyncio.to_thread(
                lambda: [r for r in self.list_bookings() if r["id"] == booking_id])
            if not rows:
                return
            try:
                resp = await get_http_client().post(self.s.booking_webhook_url,
                                                    json={"event": "booking.created", **rows[0]},
                                                    timeout=10)
                resp.raise_for_status()
            except Exception as exc:
                logger.error("Booking webhook failed for %s: %r", booking_id, exc)

        task = asyncio.create_task(send())
        self._background.add(task)
        task.add_done_callback(self._background.discard)


def _parse_hhmm(text: str) -> time:
    hour, minute = text.strip().split(":")
    return time(int(hour), int(minute))


def _fuzzy(text: str, options: list[str]) -> str:
    """Canonical option for `text` ('ion' → 'Aurora Ion', 'viman nagar' → 'Viman
    Nagar Showroom'), or ''. Words every option shares ('Aurora', 'Showroom')
    never decide a match, so 'Aurora Nova' is rejected rather than guessed."""
    low = " ".join(re.findall(r"\w+", text.lower()))
    if not low:
        return ""
    words = {o: set(re.findall(r"\w+", o.lower())) for o in options}
    shared = set.intersection(*words.values()) if len(options) > 1 else set()
    distinctive = {o: w - shared for o, w in words.items()}
    said = set(low.split()) - shared
    for opt in options:
        if low == opt.lower() or (said and said <= distinctive[opt]):
            return opt
    hits = [o for o in options if distinctive[o] and distinctive[o] <= set(low.split())]
    if len(hits) == 1:
        return hits[0]
    stripped = {" ".join(sorted(d)): o for o, d in distinctive.items() if d}
    match = difflib.get_close_matches(" ".join(sorted(said)), list(stripped), n=1, cutoff=0.75)
    return stripped[match[0]] if match else ""
