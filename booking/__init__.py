"""
booking — test drive and sales meeting bookings made by the voice agent.

  actions.py  streaming parser for the LLM's <action>{...}</action> calls
  intent.py   booking-intent detection (keeps booking turns away from web search)
  service.py  slots, validation, SQLite storage and the optional CRM webhook
"""
from booking.actions import ActionFilter
from booking.intent import looks_like_booking
from booking.service import BookingService

__all__ = ["ActionFilter", "BookingService", "looks_like_booking"]
