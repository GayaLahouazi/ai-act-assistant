"""Deterministic checks around the model: every reference in an answer must be backed by the context.

The LLM is asked to cite its sources, but a prompt is not a guarantee. After each answer we extract
every reference it contains and classify it:

* ``verified``    - the source was returned by a tool during this turn: the model has read it;
* ``mentioned``   - the reference only appears inside a text the model has read (an article that
                    points to an annex, for instance): plausible, but not read first-hand;
* ``unsupported`` - the reference appears nowhere in the context: likely hallucinated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ingest import find_references

MAX_QUESTION_CHARS = 2000


@dataclass
class CitationReport:
    verified: list[str] = field(default_factory=list)
    mentioned: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        """An answer is grounded if it cites at least one source it has read and nothing invented."""
        return bool(self.verified) and not self.unsupported

    def to_dict(self) -> dict:
        return {
            "verified": self.verified,
            "mentioned": self.mentioned,
            "unsupported": self.unsupported,
            "grounded": self.grounded,
        }


def check_citations(answer: str, seen: set[str] | dict, mentioned: set[str]) -> CitationReport:
    report = CitationReport()
    for source_id in find_references(answer):
        if source_id in seen:
            report.verified.append(source_id)
        elif source_id in mentioned:
            report.mentioned.append(source_id)
        else:
            report.unsupported.append(source_id)
    return report


def validate_question(question: str) -> str:
    question = question.strip()
    if not question:
        raise ValueError("La question est vide.")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"La question dépasse {MAX_QUESTION_CHARS} caractères.")
    return question
