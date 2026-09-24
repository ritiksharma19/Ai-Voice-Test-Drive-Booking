"""
llm/topic_guard.py
Keeps the agent on its job: cars, car ownership and this dealership.

Two layers:
  1. Here, before retrieval (deterministic, ~0 ms): a message is "on topic" when
     it uses automotive vocabulary in any supported language, names one of our
     models / showrooms / the brand, or is part of a booking. Only on-topic
     messages may use web search, so off-topic questions never reach Google
     and never cost a search.
  2. In the LLM (llm/prompts.py → GUARDRAILS): a message that is not clearly on
     topic *and* has no knowledge-base match is sent with a [TOPIC CHECK] note,
     and the model declines politely in the customer's language and steers back
     to cars. The model makes the final call, so a car question phrased without
     car words ("is it safe for kids?") is still answered, while "who won the
     match?" or "write me a poem" is declined.

Generic words that also appear in other domains ("price", "rate", "loan",
"insurance", "service") are deliberately NOT automotive vocabulary: "gold price
today" or "health insurance" must not count as car questions.
"""
from __future__ import annotations

from booking.intent import looks_like_booking
from config.settings import Settings
from llm.knowledge_base import tokenize

_AUTOMOTIVE_VOCABULARY = """
car cars vehicle vehicles automobile auto automotive suv suvs sedan hatchback mpv muv crossover
pickup coupe convertible ev evs electric hybrid phev cng lpg petrol diesel fuel mileage kmpl
engine motor gearbox transmission automatic manual amt cvt dct clutch brake brakes tyre tyres
tire tires wheel wheels alloy battery charging charger chargers fastcharge kwh bhp horsepower
torque airbag airbags adas ncap sunroof moonroof dashboard infotainment carplay
headlamp headlights suspension steering odometer speedometer
showroom dealership dealer exshowroom onroad rto fastag
driving licence license
servicing mechanic garage puncture towing roadside
maruti suzuki tata mahindra hyundai kia toyota honda skoda volkswagen renault nissan
jeep citroen bmw mercedes benz audi volvo tesla byd lexus porsche jaguar landrover isuzu
nexon creta seltos brezza thar xuv scorpio innova fortuner verna sonet
dzire baleno ertiga harrier hector
gaadi gadi gaddi gaadiyan
गाड़ी गाडी गाड़ियां कार कारें वाहन इंजन पेट्रोल डीजल सीएनजी माइलेज एवरेज टायर ब्रेक क्लच गियर
इलेक्ट्रिक चार्जिंग बैटरी शोरूम डीलर ड्राइविंग लाइसेंस आरटीओ रजिस्ट्रेशन सनरूफ एयरबैग
கார் வண்டி வாகனம் பெட்ரோல் டீசல் மைலேஜ் ఎలక్ట్రిక్ కారు బండి వాహనం పెట్రోల్ డీజిల్ మైలేజ్
ಕಾರು ಗಾಡಿ ವಾಹನ ಪೆಟ್ರೋಲ್ ಡೀಸೆಲ್ ಮೈಲೇಜ್ കാർ വണ്ടി വാഹനം പെട്രോൾ ഡീസൽ മൈലേജ്
গাড়ি গাড়ী পেট্রোল ডিজেল মাইলেজ ગાડી કાર વાહન પેટ્રોલ ડીઝલ માઇલેજ ਗੱਡੀ ਕਾਰ ਪੈਟਰੋਲ ਡੀਜ਼ਲ ਮਾਈਲੇਜ
"""

_GENERIC = {"price", "cost", "rate", "loan", "insurance", "service", "range", "colour", "seat",
            "variant", "offer", "discount", "delivery", "exchange", "warranty", "finance"}


class TopicGuard:
    def __init__(self, settings: Settings) -> None:
        own_names = " ".join([settings.business_name, *settings.car_models, *settings.showrooms])
        # Words every model name shares with the brand are fine as vocabulary
        # (a customer saying "Aurora" means the brand); generic words are not.
        self.vocabulary = (set(tokenize(_AUTOMOTIVE_VOCABULARY)) | set(tokenize(own_names))) - _GENERIC

    def is_on_topic(self, text: str) -> bool:
        return bool(self.vocabulary & set(tokenize(text))) or looks_like_booking(text)
