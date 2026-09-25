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


def indian_mobile(value) -> str:
    """'+91 98765-43210' / '098765 43210' / '९८७६५४३२१०' → '+919876543210'; '' if invalid."""
    digits = ascii_digits(str(value))
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    return f"+91{digits}" if len(digits) == 10 and digits[0] in "6789" else ""


def contains_contact_details(text: str) -> bool:
    return bool(_PHONE_RE.search(text) or _EMAIL_RE.search(text))


def mask_pii(text: str) -> str:
    """'call 98765 43210' → 'call ******3210'; emails → '<email>'."""
    def _mask(m: re.Match) -> str:
        digits = ascii_digits(m.group())
        return "*" * max(len(digits) - 4, 0) + digits[-4:]
    return _EMAIL_RE.sub("<email>", _PHONE_RE.sub(_mask, text))
