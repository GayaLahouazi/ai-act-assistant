"""Parse the Official Journal XHTML of Regulation (EU) 2024/1689 into retrievable chunks.

The Official Journal markup is regular: every article is a ``div#art_N``, every recital a
``div#rct_N``, every annex a ``div#anx_X``. We use that structure instead of cutting the text
every N characters, so that a chunk never straddles two articles and always knows where it
comes from.

Vocabulary
----------
source : a citable unit of the regulation ("Article 6", "Considérant 12", "Annexe III").
chunk  : a piece of a source, small enough to be embedded and shown to the LLM.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

# Target size of a chunk body. The embedding model reads at most 128 tokens (about 600 characters
# of French): anything beyond is silently ignored, so longer chunks hurt dense retrieval.
MAX_CHARS = 700
ROMAN = re.compile(r"^[IVXLC]+$")
DEFINITION = re.compile(r"^(\d+)\)\s*«\s*([^»]+?)\s*»")


@dataclass
class Chunk:
    id: str  # "art-6#1"
    source_id: str  # "art-6"
    label: str  # "Article 6"
    kind: str  # article | recital | annex | definition
    title: str  # title of the article / annex
    context: str  # "Chapitre III - Systèmes d'IA à haut risque > Section 1 - ..."
    text: str  # body of the chunk
    term: str = ""  # defined term, for kind == "definition"
    extra: dict = field(default_factory=dict)

    @property
    def header(self) -> str:
        parts = [self.label]
        if self.title:
            parts.append(self.title)
        head = " - ".join(parts)
        return f"{head} ({self.context})" if self.context else head

    def indexed_text(self, with_header: bool = True) -> str:
        """Text that is embedded and indexed. The header gives the chunk its context."""
        return f"{self.header}\n{self.text}" if with_header else self.text

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------------------------
# XHTML -> plain text
# --------------------------------------------------------------------------------------------


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ").replace("’", "'")
    return re.sub(r"[ \t\r\n]+", " ", text).strip()


def _render(node: Tag) -> list[str]:
    """Flatten a subtree into lines. Two-column tables ("a)" | text) become one line per row."""
    lines: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            continue
        if child.name == "table":
            for row in child.find_all("tr"):
                if row.find_parent("table") is not child:
                    continue  # rows of nested tables are rendered with their own cell
                cells = [" ".join(_render(td)) or _clean(td.get_text(" ")) for td in row.find_all("td", recursive=False)]
                line = _clean(" ".join(c for c in cells if c))
                if line:
                    lines.append(line)
        elif child.name == "p":
            line = _clean(child.get_text(" "))
            if line:
                lines.append(line)
        else:
            lines.extend(_render(child))
    return lines


def _pack(lines: list[str], max_chars: int = MAX_CHARS) -> list[str]:
    """Greedy packing of consecutive lines into chunks of at most ``max_chars`` characters."""
    chunks, current, size = [], [], 0
    for line in lines:
        # A single very long line is split on sentence boundaries.
        pieces = [line] if len(line) <= max_chars else re.split(r"(?<=[.;:])\s+", line)
        for piece in pieces:
            if current and size + len(piece) > max_chars:
                chunks.append("\n".join(current))
                current, size = [], 0
            current.append(piece)
            size += len(piece) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def _title_of(div: Tag) -> str:
    title = div.find(id=f"{div['id']}.tit_1")
    return _clean(title.get_text(" ")) if title else ""


def _context_of(div: Tag) -> str:
    """Chapter and section an article belongs to."""
    parts = []
    for parent in reversed(list(div.parents)):
        pid = parent.get("id", "") if isinstance(parent, Tag) else ""
        if pid.startswith("cpt_") and ".tit" not in pid:
            level = "Section" if ".sct_" in pid else "Chapitre"
            number = pid.split("_")[-1]
            parts.append(f"{level} {number} - {_title_of(parent)}")
    return " > ".join(parts)


# --------------------------------------------------------------------------------------------
# Parsers for the three kinds of sources
# --------------------------------------------------------------------------------------------


def _parse_recitals(soup: BeautifulSoup) -> list[Chunk]:
    chunks = []
    for div in soup.find_all("div", id=re.compile(r"^rct_\d+$")):
        n = div["id"].split("_")[1]
        text = re.sub(r"^\(\d+\)\s*", "", " ".join(_render(div)))
        for i, body in enumerate(_pack([text], 2200), 1):
            chunks.append(Chunk(f"rct-{n}#{i}", f"rct-{n}", f"Considérant {n}", "recital", "", "Préambule", body))
    return chunks


def _parse_articles(soup: BeautifulSoup, max_chars: int = MAX_CHARS) -> list[Chunk]:
    chunks = []
    for div in soup.find_all("div", id=re.compile(r"^art_\d+$")):
        n = div["id"].split("_")[1]
        title, context = _title_of(div), _context_of(div)
        lines = _render(div)
        # Drop the two heading lines ("Article 6", title) that _render picked up.
        lines = [ln for ln in lines if ln not in (f"Article {n}", title)]

        if n == "3":  # one chunk per definition: users ask "what is a deployer?"
            intro = lines[0]
            for line in lines[1:]:
                m = DEFINITION.match(line)
                if not m:
                    chunks[-1].text += "\n" + line  # sub-points a), b) of the previous definition
                    continue
                num, term = m.group(1), m.group(2)
                chunks.append(
                    Chunk(f"art-3#{num}", "art-3", "Article 3", "definition", title, context, f"{intro} {line}", term=term)
                )
            continue

        for i, body in enumerate(_pack(lines, max_chars), 1):
            chunks.append(Chunk(f"art-{n}#{i}", f"art-{n}", f"Article {n}", "article", title, context, body))
    return chunks


def _parse_annexes(soup: BeautifulSoup, max_chars: int = MAX_CHARS) -> list[Chunk]:
    chunks = []
    for div in soup.find_all("div", id=re.compile(r"^anx_[IVXLC]+$")):
        n = div["id"].split("_")[1]
        heads = [_clean(p.get_text(" ")) for p in div.find_all("p", class_="oj-doc-ti")]
        title = " - ".join(h for h in heads if not h.upper().startswith("ANNEXE"))
        lines = [ln for ln in _render(div) if ln not in heads]
        for i, body in enumerate(_pack(lines, max_chars), 1):
            chunks.append(Chunk(f"anx-{n}#{i}", f"anx-{n}", f"Annexe {n}", "annex", title, "Annexes", body))
    return chunks


def parse_regulation(html_path: Path, max_chars: int = MAX_CHARS) -> list[Chunk]:
    with warnings.catch_warnings():  # the file is XHTML; the lenient HTML parser is what we want
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")
    for note in soup.select("span.oj-note-tag"):  # footnote markers such as "(12)"
        note.decompose()
    return _parse_articles(soup, max_chars) + _parse_recitals(soup) + _parse_annexes(soup, max_chars)


# --------------------------------------------------------------------------------------------
# Baseline used in the evaluation: structure-blind fixed-size windows
# --------------------------------------------------------------------------------------------


def naive_chunks(chunks: list[Chunk], size: int = 1000, overlap: int = 150) -> list[Chunk]:
    """Re-cut every source with a sliding window of characters, as a naive pipeline would."""
    by_source: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_source.setdefault(c.source_id, []).append(c)
    out = []
    for source_id, group in by_source.items():
        first, text = group[0], " ".join(c.text for c in group)
        for i, start in enumerate(range(0, max(len(text) - overlap, 1), size - overlap), 1):
            kind = "article" if first.kind == "definition" else first.kind
            out.append(Chunk(f"{source_id}#n{i}", source_id, first.label, kind, first.title, first.context,
                             text[start : start + size]))
    return out


# --------------------------------------------------------------------------------------------
# Normalising what users and models write: "art. 6", "Article 6(2)", "annexe 3", "considérant 12"
# --------------------------------------------------------------------------------------------

_ROMANS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII"]
REFERENCE = re.compile(
    r"\b(art(?:icle)?s?\.?|consid[ée]rants?|annexes?)\s*(\d{1,3}|[IVX]{1,4}\b)", re.IGNORECASE
)


def to_source_id(kind: str, number: str) -> str | None:
    kind = kind.lower()
    if kind.startswith("art"):
        return f"art-{int(number)}" if number.isdigit() else None
    if kind.startswith("consid"):
        return f"rct-{int(number)}" if number.isdigit() else None
    if kind.startswith("annexe"):
        if number.isdigit():
            return f"anx-{_ROMANS[int(number) - 1]}" if 1 <= int(number) <= 13 else None
        return f"anx-{number.upper()}" if ROMAN.match(number.upper()) else None
    return None


def find_references(text: str) -> list[str]:
    """All source ids mentioned in a text, in order of first appearance, without duplicates."""
    seen: dict[str, None] = {}
    for kind, number in REFERENCE.findall(text):
        sid = to_source_id(kind, number)
        if sid:
            seen.setdefault(sid)
    return list(seen)
