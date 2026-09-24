"""
booking/intent.py
Cheap, multilingual detection of "the customer wants to book something".

It only decides routing: while a booking is being collected, answers such as a
name or a date must not be sent to a web search engine. The LLM itself decides
when to call the booking actions.
"""
from __future__ import annotations

import re

_BOOKING_RE = re.compile(
    r"("
    # English
    r"\b(book|booking|reserve|schedule|reschedule|appointment|slot|"
    r"test\s*-?\s*drive|meeting|visit|come\s+to\s+the\s+showroom|callback|call\s+back)\b"
    r"|"
    # Romanized Hindi
    r"\b(book\s+kar|booking\s+kar|test\s*drive\s+(chahiye|karna|leni|lena)|milna|mulakat)\b"
    r"|"
    # Devanagari (Hindi / Marathi)
    r"बुक|बुकिंग|टेस्ट\s*ड्राइव|मीटिंग|अपॉइंटमेंट|अपोइंटमेंट|मुलाकात|शोरूम\s*आ"
    r"|"
    # Tamil, Telugu, Kannada, Bengali, Gujarati
    r"புக்|டெஸ்ட்\s*டிரைவ்|బుక్|టెస్ట్\s*డ్రైవ్|ಬುಕ್|ಟೆಸ್ಟ್\s*ಡ್ರೈವ್|বুক|টেস্ট\s*ড্রাইভ|બુક|ટેસ્ટ\s*ડ્રાઇવ"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def looks_like_booking(text: str) -> bool:
    return bool(_BOOKING_RE.search(text))
