"""
llm/knowledge_base.py
Local knowledge base: Markdown / text files in KB_DIR, searched with BM25.

Each heading section becomes one chunk, prefixed with its heading path
("Aurora Ion > Aurora Ion price and variants") so a chunk is understandable on
its own. Search runs in well under a millisecond for a dealership-sized catalog,
so it never adds latency to a turn.

A result counts as an answer only when the matching chunks cover at least
KB_MIN_COVERAGE of the query's meaningful terms. "petrol price today" shares
only "price" with the car catalog, so it is treated as a miss and the question
falls through to web search instead of being answered with car prices.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from config.logging_config import get_logger

logger = get_logger("llm.knowledge_base")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_STOPWORDS = frozenset("""
a an and any are as at be by can could do does did for from get give has have how i
if in into is it its me my of on or our please show so tell than that the their them
there these this those to us was we what when where which who why will with would you
your about also just like know want need much many some more most very yeah yes no
hai hain kya ka ki ke ko mein me se aur bhi hi batao bataiye kaise kitna kitne
है हैं क्या का की के को में से और भी ही बताओ बताइए कैसे कितना कितनी कितने मुझे
""".split())

# Hinglish / Indic words mapped to the English terms the catalog uses.
_SYNONYMS = {
    "kimat": "price", "keemat": "price", "daam": "price", "dam": "price", "rate": "price",
    "cost": "price", "costs": "price", "prices": "price", "कीमत": "price", "दाम": "price",
    "average": "mileage", "maileage": "mileage", "माइलेज": "mileage", "एवरेज": "mileage",
    "gaadi": "car", "gadi": "car", "गाड़ी": "car", "गाडी": "car", "कार": "car",
    "ev": "electric", "बैटरी": "battery", "रेंज": "range",
    "colour": "colours", "color": "colours", "colors": "colours", "रंग": "colours",
    "seater": "seat", "seats": "seat", "सीट": "seat",
    "टेस्ट": "test", "ड्राइव": "drive", "शोरूम": "showroom",
}


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    out = []
    for raw in _TOKEN_RE.findall(text.lower()):
        tok = _SYNONYMS.get(raw, raw)
        if tok in _STOPWORDS or (len(tok) < 2 and not tok.isdigit()):
            continue
        out.append(_stem(tok))
    return out


@dataclass
class Chunk:
    title: str
    text: str
    tokens: list[str]


def _split_markdown(text: str, source: str) -> list[tuple[str, str]]:
    """[(heading path, body)] — one entry per heading section with a body."""
    text = _COMMENT_RE.sub("", text)
    path: list[tuple[int, str]] = []
    sections: list[tuple[str, str]] = []
    body: list[str] = []

    def close() -> None:
        content = "\n".join(body).strip()
        if content:
            title = " > ".join(h for _, h in path[1:] or path) or source
            sections.append((title, content))
        body.clear()

    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            close()
            level = len(m.group(1))
            path = [(lv, h) for lv, h in path if lv < level] + [(level, m.group(2).strip())]
        else:
            body.append(line)
    close()
    return sections


class LocalKnowledgeBase:
    k1, b = 1.5, 0.75

    def __init__(self, kb_dir: str, min_coverage: float = 0.5, top_k: int = 3) -> None:
        root = Path(kb_dir)
        self.root = root if root.is_absolute() else _PROJECT_ROOT / root
        self.min_coverage = min_coverage
        self.top_k = top_k
        self.chunks: list[Chunk] = []
        self._load()

    def _load(self) -> None:
        files = sorted(p for ext in ("*.md", "*.txt") for p in self.root.rglob(ext)) \
            if self.root.is_dir() else []
        for path in files:
            for title, body in _split_markdown(path.read_text(encoding="utf-8"), path.stem):
                self.chunks.append(Chunk(title, body, tokenize(f"{title}\n{body}")))
        n = len(self.chunks) or 1
        self._avgdl = sum(len(c.tokens) for c in self.chunks) / n
        df: Counter[str] = Counter()
        for c in self.chunks:
            df.update(set(c.tokens))
        self._idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}
        self._tf = [Counter(c.tokens) for c in self.chunks]
        logger.info("Local KB: %d chunks from %d files in %s", len(self.chunks), len(files), self.root)

    @property
    def available(self) -> bool:
        return bool(self.chunks)

    def _score(self, i: int, terms: list[str]) -> float:
        tf, dl = self._tf[i], len(self.chunks[i].tokens)
        score = 0.0
        for t in terms:
            f = tf.get(t, 0)
            if f:
                score += self._idf[t] * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
        return score

    def search(self, query: str) -> str:
        """Top matching chunks as context text, or '' when the KB has no answer."""
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms or not self.chunks:
            return ""
        ranked = sorted(((self._score(i, terms), i) for i in range(len(self.chunks))),
                        reverse=True)
        top = [i for score, i in ranked[:self.top_k] if score > 0]
        if not top:
            return ""
        covered = {t for i in top for t in terms if t in self._tf[i]}
        # Terms that appear in no chunk at all (e.g. "cricket") count against coverage.
        if len(covered) / len(terms) < self.min_coverage:
            logger.info("KB miss (coverage %d/%d) for %r", len(covered), len(terms), query[:60])
            return ""
        return "\n\n".join(f"Source: {self.chunks[i].title}\n{self.chunks[i].text}" for i in top)
