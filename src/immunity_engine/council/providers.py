"""Two ways to poll the council, behind one interface.

The deterministic reviewer is not a placeholder for the language-model one. It is
the rule engine, and it runs first. Every check it makes is a threshold this
programme has already been burned by, expressed as code that gives the same answer
every time and can be argued with line by line.

The language-model reviewer is the second pass. It reads the same evidence packet
and reasons about the things a threshold cannot express: whether a target makes
sense for this place, whether an operational plan is deliverable, whether the
argument for a recommendation actually follows. It can raise findings the rules
did not anticipate. It cannot silently overturn one, because both sets of findings
reach the adjudicator.

That order is deliberate. A recommendation that fails a deterministic check fails
it whether or not a model is available, and no run of this engine depends on a
network call to produce its safety findings.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from .experts import Confidence, Expert, Verdict

#: Environment variable naming the model the council calls. Set it to move the
#: council to a different model without touching code.
MODEL_ENV_VAR = "IMMUNITY_COUNCIL_MODEL"

#: Environment variable holding the API key. Absent means the language-model pass
#: is skipped and only the deterministic reviewer runs.
API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"


@dataclass
class Finding:
    """One thing an expert noticed.

    Attributes:
        expert: Key of the seat that raised it.
        verdict: What the seat concluded.
        confidence: How much weight the seat's verdict can bear.
        summary: One sentence stating the finding.
        evidence: The numbers the finding rests on, so a reader can check it.
        condition: What must be true or be done for the recommendation to stand.
        missing_datum: The one measurement that would most raise this seat's
            confidence. Naming it is how the council directs the next data spend.
        source: ``rules`` or ``model``.
    """

    expert: str
    verdict: Verdict
    confidence: Confidence
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    condition: str | None = None
    missing_datum: str | None = None
    source: str = "rules"

    def to_dict(self) -> dict[str, Any]:
        return {
            "expert": self.expert,
            "verdict": self.verdict.value,
            "confidence": self.confidence.value,
            "summary": self.summary,
            "evidence": self.evidence,
            "condition": self.condition,
            "missing_datum": self.missing_datum,
            "source": self.source,
        }


class Reviewer(Protocol):
    """Anything that can review an evidence packet on behalf of a seat."""

    def review(self, expert: Expert, packet: dict[str, Any]) -> list[Finding]:
        """Return the findings this seat raises about the packet."""
        ...


class ModelReviewer:
    """Polls a language model, one call per seat.

    The seats are polled independently and never see each other's answers. Shown
    another expert's verdict, a model tends to converge on it, and a council that
    converges has stopped doing the one job it was assembled for.

    Example:
        >>> reviewer = ModelReviewer()  # doctest: +SKIP
        >>> findings = reviewer.review(expert, packet)  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1400,
        timeout_seconds: float = 90.0,
    ) -> None:
        self.model = model or os.environ.get(MODEL_ENV_VAR, "claude-opus-5")
        self.api_key = api_key or os.environ.get(API_KEY_ENV_VAR)
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    @property
    def available(self) -> bool:
        """Whether this reviewer can run. False means the council runs on rules alone."""
        if not self.api_key:
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def review(self, expert: Expert, packet: dict[str, Any]) -> list[Finding]:
        """Ask one seat to review the packet.

        Args:
            expert: The seat.
            packet: Evidence packet from ``council.build_packet``.

        Returns:
            The findings raised. An empty list when the reviewer is unavailable or
            the response could not be parsed; the caller records that as a gap in
            the council record rather than as an endorsement.
        """
        if not self.available:
            return []

        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout_seconds)
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"{expert.brief()}\n\n"
                            "EVIDENCE PACKET\n"
                            f"{json.dumps(packet, indent=2, default=str)}\n\n"
                            "Return JSON only, matching the schema in your instructions."
                        ),
                    }
                ],
            )
        except Exception as error:  # network, auth, rate limit, model error
            return [
                Finding(
                    expert=expert.key,
                    verdict=Verdict.CHALLENGE,
                    confidence=Confidence.LOW,
                    summary=(
                        "This seat could not be polled, so its failure modes went unreviewed."
                    ),
                    evidence={"error": type(error).__name__},
                    missing_datum="A successful review from this seat.",
                    source="model",
                )
            ]

        text = "".join(block.text for block in response.content if block.type == "text")
        return _parse_findings(expert, text)


_SYSTEM_PROMPT = """You review a campaign-requirement recommendation for a polio outbreak response programme.

You hold one seat on a council. Other seats cover other failure modes; you will not see their views and must not speculate about them. Stay inside your mandate.

Rules of the seat:
- Judge the recommendation on the evidence packet. Do not invent figures. If a number you need is absent, say which one and treat its absence as a limit on your confidence, not as reassurance.
- A wide interval honestly stated is better than a narrow one. Do not challenge a recommendation for being uncertain; challenge it for claiming certainty it has not earned.
- "Run more rounds" is the default action this programme reaches for. Your job is not to approve or reject it on principle but to check whether this evidence supports it here.
- Name a condition only if it is checkable. "Improve coordination" is not a condition; "confirm the microplan denominator against the last enumeration before round 1" is.
- Reserve a block for something that makes the recommendation unsafe or wrong inside your mandate, not for something you would have done differently.

Return only a JSON object, no prose around it:
{
  "verdict": "endorse" | "endorse_with_conditions" | "challenge" | "block",
  "confidence": "low" | "medium" | "high",
  "findings": [
    {
      "summary": "one sentence stating the finding",
      "evidence": {"field_from_packet": value},
      "condition": "what must be true or done, or null",
      "missing_datum": "the one measurement that would most raise your confidence, or null"
    }
  ]
}"""


def _parse_findings(expert: Expert, text: str) -> list[Finding]:
    """Turn a model response into findings, failing loudly rather than silently."""
    payload = _extract_json(text)
    if payload is None:
        return [
            Finding(
                expert=expert.key,
                verdict=Verdict.CHALLENGE,
                confidence=Confidence.LOW,
                summary="This seat's response could not be parsed, so its review is missing.",
                evidence={"raw_prefix": text[:200]},
                source="model",
            )
        ]

    try:
        verdict = Verdict(payload.get("verdict", "challenge"))
        confidence = Confidence(payload.get("confidence", "low"))
    except ValueError:
        verdict, confidence = Verdict.CHALLENGE, Confidence.LOW

    if not expert.may_block and verdict is Verdict.BLOCK:
        verdict = Verdict.CHALLENGE

    entries = payload.get("findings") or []
    if not entries:
        return [
            Finding(
                expert=expert.key,
                verdict=verdict,
                confidence=confidence,
                summary="No specific finding raised.",
                source="model",
            )
        ]

    return [
        Finding(
            expert=expert.key,
            verdict=verdict,
            confidence=confidence,
            summary=str(entry.get("summary", "")).strip() or "No summary given.",
            evidence=entry.get("evidence") or {},
            condition=entry.get("condition"),
            missing_datum=entry.get("missing_datum"),
            source="model",
        )
        for entry in entries
    ]


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.lower().startswith("json") else text
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
