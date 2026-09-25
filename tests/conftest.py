"""Keep server tests independent of the developer's .env (tokens and limits)."""
from __future__ import annotations

from dataclasses import replace

import pytest

import server


@pytest.fixture(autouse=True)
def default_server_security(monkeypatch):
    """Tests that need a token or a limit set it themselves."""
    monkeypatch.setattr(server, "settings", replace(
        server.settings, admin_token="", access_token="", exotel_ws_token="",
        max_sessions=50, max_turns_per_minute=20))
