"""Knowledge base uploads and the "Call me back" button (no network or keys)."""
from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import server
from core.privacy import indian_mobile
from llm.knowledge_base import (
    _MAX_CHUNK_CHARS, LocalKnowledgeBase, _split_long, delete_document, document_text, save_document)
from telephony.callback import CallbackLimiter
from tests.test_pipeline import FakeBackend, base_settings, make_orchestrator
from tests.test_server import FakeSTT, FakeTTS


def _minimal_pdf(text: str) -> bytes:
    """A one-page PDF showing `text` (enough for pypdf's text extraction)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


# ── knowledge base documents ─────────────────────────────────────────────────

def test_long_sections_are_split_at_paragraphs():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(12))
    pieces = _split_long("Brochure", body)
    assert len(pieces) > 1
    assert all(len(text) <= _MAX_CHUNK_CHARS for _, text in pieces)
    assert pieces[0][0] == "Brochure (part 1)"
    assert _split_long("Short", "tiny") == [("Short", "tiny")]


def test_document_text_formats():
    assert document_text("offers.md", "# Diwali offer\nFree insurance".encode()).startswith("# Diwali")
    assert "Monsoon service camp" in document_text("camp.pdf", _minimal_pdf("Monsoon service camp"))
    with pytest.raises(ValueError, match="Unsupported"):
        document_text("virus.exe", b"MZ")
    with pytest.raises(ValueError, match="UTF-8"):
        document_text("notes.txt", "कीमत".encode("utf-16"))
    with pytest.raises(ValueError, match="no readable text"):
        document_text("empty.txt", b"   ")


def test_save_and_delete_stay_inside_uploads():
    root = Path(tempfile.mkdtemp())
    (root / "cars.md").write_text("# Cars\nBuilt-in catalog", encoding="utf-8")
    stored = save_document(root, "../../Price List (Oct).pdf", _minimal_pdf("Pico price 6 lakh"))
    assert stored == "Price-List-Oct.txt"
    assert (root / "uploads" / stored).is_file()
    assert not delete_document(root, "../cars.md")          # built-in files can't be deleted
    assert (root / "cars.md").is_file()
    assert delete_document(root, stored) and not (root / "uploads" / stored).exists()


# ── upload API ───────────────────────────────────────────────────────────────

@pytest.fixture
def kb_app():
    kb_dir = Path(tempfile.mkdtemp())
    (kb_dir / "cars.md").write_text("# Aurora Pico\nThe Pico costs 6 lakh.\n", encoding="utf-8")
    settings = base_settings(kb_provider="local", kb_dir=str(kb_dir))
    server.engines.stt, server.engines.tts = FakeSTT(), FakeTTS()
    server.engines.llm = make_orchestrator(FakeBackend("fake"), settings=settings)
    return TestClient(server.app), kb_dir


def test_upload_is_searchable_then_deletable(kb_app):
    client, kb_dir = kb_app
    retrieval = server.engines.llm.retrieval
    assert retrieval.local_kb.search("monsoon service camp") == ""

    assert client.get("/config").json()["kb_uploads"] is True
    res = client.put("/kb/documents/camp.md",
                     content=b"# Monsoon service camp\nFree monsoon service camp checks for all cars in July.")
    assert res.status_code == 200 and res.json()["stored_as"] == "camp.md"
    assert "Free monsoon service camp" in retrieval.local_kb.search("monsoon service camp")

    docs = client.get("/kb/documents").json()["documents"]
    assert {(d["name"], d["uploaded"]) for d in docs} == {("cars.md", False), ("camp.md", True)}

    assert client.delete("/kb/documents/cars.md").status_code == 404   # built-in
    assert client.delete("/kb/documents/camp.md").status_code == 200
    assert retrieval.local_kb.search("monsoon service camp") == ""


def test_upload_needs_page_token_only_when_set_and_valid_file(kb_app, monkeypatch):
    client, _ = kb_app
    assert client.put("/kb/documents/x.exe", content=b"MZ").status_code == 422
    monkeypatch.setattr(server, "settings", replace(server.settings, kb_upload_max_mb=0))
    assert client.put("/kb/documents/big.md", content=b"x" * 10).status_code == 413
    monkeypatch.setattr(server, "settings", replace(server.settings, kb_upload_max_mb=10,
                                                    access_token="demo"))
    assert client.get("/kb/documents").status_code == 401
    assert client.put("/kb/documents/x.md?token=wrong", content=b"# x\ny").status_code == 401
    assert client.put("/kb/documents/x.md?token=demo", content=b"# x\ny").status_code == 200
    assert client.delete("/kb/documents/x.md?token=demo").status_code == 200


def test_uploads_off_when_kb_is_not_local(kb_app):
    client, _ = kb_app
    server.engines.llm = make_orchestrator(FakeBackend("fake"))     # KB_PROVIDER=off
    assert client.get("/config").json()["kb_uploads"] is False
    assert client.put("/kb/documents/x.md", content=b"# x\ny").status_code == 409


# ── call me back ─────────────────────────────────────────────────────────────

def test_indian_mobile_normalisation():
    assert indian_mobile("+91 98765-43210") == "+919876543210"
    assert indian_mobile("098765 43210") == "+919876543210"
    assert indian_mobile("९८७६५४३२१०") == "+919876543210"
    assert indian_mobile("12345 67890") == "" and indian_mobile("") == ""


def test_callback_limiter():
    s = replace(base_settings(), callback_per_hour=3, callback_per_client_per_hour=2,
                callback_number_cooldown_min=10)
    limiter = CallbackLimiter(s)
    assert limiter.check("+919876543210", "ip1") is None
    assert "already calling" in limiter.check("+919876543210", "ip2")     # same number
    assert limiter.check("+919876543211", "ip1") is None
    assert "Too many" in limiter.check("+919876543212", "ip1")            # same visitor
    assert limiter.check("+919876543213", "ip3") is None
    assert "busy" in limiter.check("+919876543214", "ip4")                # hourly cap
    limiter.release("+919876543213", "ip3")                               # call failed
    assert limiter.check("+919876543213", "ip3") is None


@pytest.fixture
def callback_app(monkeypatch):
    s = replace(server.settings, exotel_account_sid="sid", exotel_api_key="key",
                exotel_api_token="tok", exotel_caller_id="08012345678", exotel_app_id="123",
                callback_enabled=True, access_token="", callback_per_hour=20,
                callback_per_client_per_hour=5, callback_number_cooldown_min=10)
    monkeypatch.setattr(server, "settings", s)
    monkeypatch.setattr(server, "callbacks", CallbackLimiter(s))
    calls = []

    async def fake_place_call(settings, to):
        calls.append(to)
        return {"call_sid": "c1", "status": "in-progress"}
    monkeypatch.setattr(server.exotel, "place_call", fake_place_call)
    server.engines.stt, server.engines.tts = FakeSTT(), FakeTTS()
    server.engines.llm = make_orchestrator(FakeBackend("fake"))
    return TestClient(server.app), calls


def test_callback_places_call_once(callback_app):
    client, calls = callback_app
    assert client.get("/config").json()["callback_ready"] is True
    res = client.post("/callback", json={"phone": "98765 43210"})
    assert res.status_code == 200 and res.json() == {"status": "calling"}
    assert calls == ["+919876543210"]
    again = client.post("/callback", json={"phone": "+91 98765 43210"})
    assert again.status_code == 429 and calls == ["+919876543210"]
    assert client.post("/callback", json={"phone": "12345"}).status_code == 422


def test_callback_hidden_without_exotel_and_guarded_by_access_token(callback_app, monkeypatch):
    client, calls = callback_app
    monkeypatch.setattr(server, "settings", replace(server.settings, access_token="demo"))
    assert client.post("/callback", json={"phone": "9876543210"}).status_code == 401
    assert client.post("/callback?token=demo", json={"phone": "9876543210"}).status_code == 200
    monkeypatch.setattr(server, "settings", replace(server.settings, exotel_app_id=""))
    cfg = client.get("/config").json()
    assert cfg["callback_enabled"] is True and cfg["callback_ready"] is False   # button shown, says unavailable
    assert client.post("/callback?token=demo", json={"phone": "9876543211"}).status_code == 404
    assert calls == ["+919876543210"]


def test_failed_callback_can_be_retried(callback_app, monkeypatch):
    client, _ = callback_app

    async def broken(settings, to):
        raise RuntimeError("Exotel HTTP 500")
    monkeypatch.setattr(server.exotel, "place_call", broken)
    assert client.post("/callback", json={"phone": "9876543210"}).status_code == 502
    assert server.callbacks.check("+919876543210", "testclient") is None    # not stuck in cooldown


def test_local_kb_reads_uploaded_folder():
    root = Path(tempfile.mkdtemp())
    save_document(root, "offers.md", b"# Festive offer\nZero down payment on the Sprint.")
    assert "Zero down payment" in LocalKnowledgeBase(str(root)).search("festive offer sprint")
