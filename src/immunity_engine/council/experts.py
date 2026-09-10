"""The council: six mandates that each see a different way for the answer to be wrong.

A single reviewer, human or model, checks the things they happen to know about.
The failures that reach a programme are the ones nobody's mandate covered. So the
mandates are fixed in advance, they do not overlap by accident, and each one owns
a named list of failure modes drawn from what this programme has actually got
wrong before.

Every expert answers in the same shape, so their answers can be compared rather
than merged into prose. Disagreement between them is the product. An adjudicator
that averaged them away would destroy the only thing the council adds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Verdict(str, Enum):
    """What an expert concluded about a recommendation."""

    ENDORSE = "endorse"
    """Sound as it stands."""

    ENDORSE_WITH_CONDITIONS = "endorse_with_conditions"
    """Sound if the stated conditions are met. The conditions are binding."""

    CHALLENGE = "challenge"
    """A material objection that changes the recommendation but does not void it."""

    BLOCK = "block"
    """The recommendation must not go forward as written.

    One block holds the recommendation. It is not outvoted, because the mandates
    do not overlap: an expert blocking inside their own mandate is the only one
    looking at that failure mode.
    """


class Confidence(str, Enum):
    """How much weight the expert's own verdict can bear."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class Expert:
    """One seat on the council.

    Attributes:
        key: Stable identifier.
        title: Role as it appears in the record.
        mandate: What this seat is responsible for, and nothing else.
        failure_modes: Named ways the recommendation can be wrong that this seat
            owns. Drawn from what the programme has previously got wrong.
        may_block: Whether this seat can hold a recommendation on its own.
    """

    key: str
    title: str
    mandate: str
    failure_modes: tuple[str, ...]
    may_block: bool = True

    def brief(self) -> str:
        """The seat's instructions, as given to a reviewer."""
        lines = [
            f"You hold the {self.title} seat.",
            f"Mandate: {self.mandate}",
            "You are responsible for these failure modes and no others. Do not comment "
            "outside your mandate; another seat owns it.",
        ]
        lines.extend(f"  - {mode}" for mode in self.failure_modes)
        return "\n".join(lines)


COUNCIL: tuple[Expert, ...] = (
    Expert(
        key="epidemiologist",
        title="Field Epidemiologist",
        mandate=(
            "Whether the immunity target is the right target for this place, and whether "
            "reaching it would actually interrupt transmission."
        ),
        failure_modes=(
            "A flat national herd-immunity threshold applied where the local reproduction "
            "number is far from the national one, making the target too high or too low.",
            "Population immunity treated as sufficient when the susceptible children are "
            "clustered rather than spread, so transmission continues inside the cluster.",
            "A round count that ignores how long the lineage has already circulated "
            "undetected, so the plan acts on a susceptible profile a year out of date.",
            "Immunity measured against a target age band that does not match the ages "
            "the virus is actually circulating in.",
        ),
    ),
    Expert(
        key="programme_manager",
        title="Immunisation Programme Manager",
        mandate=(
            "Whether the plan can be executed at the tempo and scale stated, with the "
            "teams, vaccine and money that exist."
        ),
        failure_modes=(
            "A round interval shorter than the operational cycle time for microplanning, "
            "team training and vaccine arrival.",
            "Community and team fatigue after repeated rounds, which the reach model "
            "cannot see because it has never observed that many rounds in a row.",
            "Rounds scheduled into the rainy season, harvest or a period of population "
            "movement, when reach is structurally lower.",
            "A plan that consumes the vaccine or the workforce another geography needs "
            "more, without that trade being named.",
        ),
    ),
    Expert(
        key="statistician",
        title="Adversarial Biostatistician",
        mandate=(
            "Whether the numbers support the confidence attached to them. Argue against "
            "the recommendation using its own evidence."
        ),
        failure_modes=(
            "Prediction intervals that are not calibrated, so a stated confidence level "
            "means less than it says.",
            "A recommendation naming specific units after scanning many, without "
            "correcting for the number of comparisons made.",
            "Inference drawn at state level where the epidemiologically relevant "
            "variation lives at LGA level, hiding the distribution beneath a mean.",
            "Circularity: a predictor that is definitionally linked to the outcome it "
            "is being used to predict.",
            "A non-significant result or a small effect reported as evidence of no "
            "effect, when the sample could not have detected one.",
        ),
    ),
    Expert(
        key="access_analyst",
        title="Access and Security Analyst",
        mandate=(
            "Whether the children the plan counts on reaching can in fact be reached, "
            "and what it would take to change that."
        ),
        failure_modes=(
            "An unreachable core estimated from settlements recorded as inaccessible, "
            "when the larger group is children in accessible settlements who are never "
            "enumerated.",
            "Mobile and nomadic populations counted in the denominator but absent from "
            "the microplan, so they are missed every round by construction.",
            "Insecurity treated as equivalent to inaccessibility, when a settlement can "
            "be dangerous and still reachable, or calm and still unmapped.",
            "Access assumed static across a plan that runs for a year or more.",
        ),
    ),
    Expert(
        key="systems_analyst",
        title="Systems Analyst",
        mandate=(
            "Which stock the recommendation moves, which feedback loop it acts on, and "
            "whether that is the highest-leverage available action."
        ),
        failure_modes=(
            "Strengthening a balancing loop that has a long delay in it, which increases "
            "oscillation rather than control.",
            "Adding response strength where reducing the detection or decision delay "
            "would do more, at lower cost.",
            "A parameter adjustment presented as a strategy.",
            "Resource following measurability rather than risk, so the places that "
            "report well keep receiving what the places that report badly need.",
            "A rule that will be met rather than achieved once it becomes a target.",
        ),
        may_block=False,
    ),
    Expert(
        key="data_auditor",
        title="Data Integrity Auditor",
        mandate=(
            "Whether the inputs mean what the engine assumed they mean, starting with "
            "the denominator."
        ),
        failure_modes=(
            "A coverage figure used without its denominator provenance, so two "
            "incompatible denominators are compared as if they were one.",
            "Administrative counts treated as reach when independent verification says "
            "otherwise, in a place where the two diverge widely.",
            "A settlement set that changed between rounds, making a trend an artefact "
            "of the geography rather than of coverage.",
            "Coverage above 100% read as success rather than as a defective denominator.",
            "Verified counts from a lot too small to support the precision claimed.",
        ),
    ),
)

COUNCIL_BY_KEY: dict[str, Expert] = {expert.key: expert for expert in COUNCIL}
