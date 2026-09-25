"""
telephony/callback.py
Abuse limits for the public "Call me back" button (the AI agent phones the customer).

Anyone who can open the page can ask for a call to any number, so three
in-memory limits guard it (they reset on restart, which is fine for one server):
  • per number: at most one call per CALLBACK_NUMBER_COOLDOWN_MIN minutes
  • per client (IP): CALLBACK_PER_CLIENT_PER_HOUR
  • overall: CALLBACK_PER_HOUR, which caps what a flood can cost
"""
from __future__ import annotations

import time
from collections import deque

from config.settings import Settings

_HOUR = 3600.0


class CallbackLimiter:
    def __init__(self, s: Settings) -> None:
        self.s = s
        self._by_number: dict[str, float] = {}
        self._by_client: dict[str, deque[float]] = {}
        self._all: deque[float] = deque()

    @staticmethod
    def _trim(times: deque[float], now: float) -> None:
        while times and now - times[0] > _HOUR:
            times.popleft()

    def check(self, number: str, client: str) -> str | None:
        """None if the call may go ahead (and records it), else the reason it may not."""
        now = time.monotonic()
        last = self._by_number.get(number)
        if last is not None and now - last < self.s.callback_number_cooldown_min * 60:
            return "We are already calling this number. Please wait a few minutes before trying again."
        client_times = self._by_client.setdefault(client, deque())
        self._trim(client_times, now)
        if self.s.callback_per_client_per_hour and len(client_times) >= self.s.callback_per_client_per_hour:
            return "Too many call requests. Please try again later."
        self._trim(self._all, now)
        if self.s.callback_per_hour and len(self._all) >= self.s.callback_per_hour:
            return "Our lines are busy right now. Please try again later."

        self._by_number[number] = now
        client_times.append(now)
        self._all.append(now)
        if len(self._by_number) > 10_000:   # forget numbers whose cooldown has passed
            cutoff = now - self.s.callback_number_cooldown_min * 60
            self._by_number = {n: t for n, t in self._by_number.items() if t > cutoff}
        if len(self._by_client) > 10_000:   # forget clients with no call in the last hour
            for times in self._by_client.values():
                self._trim(times, now)
            self._by_client = {c: t for c, t in self._by_client.items() if t}
        return None

    def release(self, number: str, client: str) -> None:
        """Undo the last check() when the call could not be placed, so the customer can retry."""
        self._by_number.pop(number, None)
        if self._all:
            self._all.pop()
        client_times = self._by_client.get(client)
        if client_times:
            client_times.pop()
