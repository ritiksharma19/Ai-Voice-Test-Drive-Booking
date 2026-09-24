"""
core/privacy.py
Keeps customer contact details out of logs and third-party search engines.
"""
from __future__ import annotations

import re
import unicodedata

# 7+ digits, optionally separated by spaces / dashes (phone numbers as spoken or typed).
# \d matches Devanagari and other Indic digits too.
_PHONE_RE = re.compile(r"(?:\+?\d[\s\-]?){7,}\d")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def ascii_digits(text: str) -> str:
    """Only the digits of `text`, with Indic digits (९८७…) converted to 0-9."""
    return "".join(str(unicodedata.digit(c)) for c in text if c.isdigit())


def contains_contact_details(text: str) -> bool:
    return bool(_PHONE_RE.search(text) or _EMAIL_RE.search(text))


def mask_pii(text: str) -> str:
    """'call 98765 43210' → 'call ******3210'; emails → '<email>'."""
    def _mask(m: re.Match) -> str:
        digits = ascii_digits(m.group())
        return "*" * max(len(digits) - 4, 0) + digits[-4:]
    return _EMAIL_RE.sub("<email>", _PHONE_RE.sub(_mask, text))
