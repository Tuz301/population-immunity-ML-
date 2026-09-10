"""Data contract for the campaign-requirement engine.

Every input the engine uses is declared here. A field that the programme
cannot supply is optional and has a documented fallback, so a partial
dataset degrades the answer instead of stopping it. Nothing in this module
computes epidemiology; it only fixes the shape and the units.

Grain
-----
The panel is one row per (geographic unit, campaign round). The geographic
unit is a settlement, a ward or an LGA. The unit of *inference* is the LGA
(MAP v1.0 4.3); ward and settlement rows exist for targeting and are
aggregated upward with their own uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------


class Grain(str, Enum):
    """Geographic grain of a panel row."""

    SETTLEMENT = "settlement"
    WARD = "ward"
    LGA = "lga"


class DenominatorBasis(str, Enum):
    """Denominator provenance. Mirrors ``ref.denominator_basis`` in the NEOC portal.

    No coverage figure is accepted without this stamp (MAP v1.0 3.6).
    """

    CENSUS_PROJECTED = "census_projected"
    MICROPLAN = "microplan"
    HH_COUNT = "hh_count"
    WPV_TARGET = "wpv_target"


class AccessibilityStatus(str, Enum):
    """GTS accessibility label.

    ``PARTIALLY_ACCESSIBLE`` is deliberately not treated as unreachable.
    Excluding it shrinks the denominator and inflates coverage.
    """

    FULLY_ACCESSIBLE = "Fully Accessible"
    PARTIALLY_ACCESSIBLE = "Partially Accessible"
    INACCESSIBLE = "Inaccessible"


class Feasibility(str, Enum):
    """Verdict on whether campaigns alone can reach the immunity target."""

    FEASIBLE = "feasible"
    """The target is reached within the round budget."""

    FEASIBLE_BEYOND_BUDGET = "feasible_beyond_budget"
    """The target is reachable, but only after more rounds than the budget allows."""

    INFEASIBLE_BY_CAMPAIGNS = "infeasible_by_campaigns"
    """The immunity ceiling is below the target. More rounds cannot close the gap.

    This verdict is the point of the engine. It routes the unit to an access,
    strategy or denominator intervention instead of another round.
    """

    INSUFFICIENT_DATA = "insufficient_data"
    """Inputs are too thin to give an answer. The engine refuses rather than guesses."""


# --------------------------------------------------------------------------
# Column contract
# --------------------------------------------------------------------------

#: Columns the panel must carry. The engine stops if one is absent.
REQUIRED_COLUMNS: dict[str, str] = {
    "unit_id": "Stable identifier of the geographic unit (settlement_sk, ward code or LGA code).",
    "grain": "One of Grain.",
    "lga_code": "LGA of the unit. Present on every grain, because the LGA is the inference unit.",
    "state_code": "State of the unit.",
    "round_code": "Campaign round identifier, e.g. '2026-OBR2'.",
    "round_index": "Integer ordering of rounds in time. Ties are not allowed.",
    "round_start": "Round start date.",
    "campaign_type": "Vaccine and modality, e.g. 'nOPV2-SIA'.",
    "target_pop": "Denominator: children in the target age band.",
    "denominator_basis": "One of DenominatorBasis. The stamp on target_pop.",
    "admin_vaccinated": "Administrative numerator, e.g. eTally doses. Reported, not verified.",
}

#: Columns that sharpen the answer. Each has a fallback documented in
#: ``FALLBACKS`` and every use of a fallback is recorded in the provenance log.
OPTIONAL_COLUMNS: dict[str, str] = {
    "verified_vaccinated": "Independently verified numerator (IEV/FIONET/LQAS). The reach signal the engine trusts.",
    "verified_sample_n": "Sample size behind verified_vaccinated. Drives the width of the reach interval.",
    "accessibility_status": "One of AccessibilityStatus.",
    "security_compromised": "Reachable but insecure. Orthogonal to accessibility.",
    "is_inaccessible": "Denominator-exclusion flag. True only when accessibility_status is Inaccessible.",
    "refusal_count": "Recorded household refusals in the round.",
    "latitude": "Decimal degrees.",
    "longitude": "Decimal degrees.",
    "mlos_version": "Master List of Settlements version. Guards trend comparability.",
    "births_per_month": "Monthly births into the target cohort. Drives susceptible replenishment.",
    "ri_coverage": "Routine immunisation coverage of the birth cohort, 0-1.",
    "npafp_rate": "Non-polio AFP rate per 100k children under 15.",
    "stool_adequacy": "Share of AFP cases with adequate stool specimens, 0-1.",
    "es_positive_90d": "Environmental surveillance positives in the last 90 days in the LGA.",
    "r0_local": "Local basic reproduction number, if estimated. Sets a unit-specific immunity target.",
    "teams_deployed": "Vaccination teams deployed in the round. Proxy for delivery capacity.",
    "nomadic_share": "Share of the target population that is nomadic or seasonally mobile, 0-1.",
}

#: What the engine does when an optional column is absent.
FALLBACKS: dict[str, str] = {
    "verified_vaccinated": (
        "Reach falls back to administrative coverage shrunk by the programme-wide "
        "verified-to-administrative ratio. The reach interval is widened accordingly."
    ),
    "verified_sample_n": "Assumes the LQAS default lot size in EngineConfig.",
    "accessibility_status": "Assumed Fully Accessible. Recorded as an optimistic assumption.",
    "security_compromised": "Assumed False.",
    "is_inaccessible": "Derived from accessibility_status when that is present, otherwise False.",
    "refusal_count": "Refusal contribution to the unreachable core falls back to the national prior.",
    "births_per_month": "Derived from target_pop and the crude birth rate in EngineConfig.",
    "ri_coverage": "Falls back to the state-level prior in EngineConfig.",
    "r0_local": "Falls back to the programme R0 in EngineConfig, which raises the immunity target.",
    "teams_deployed": "Team-density feature is dropped from the reach model.",
    "nomadic_share": "Assumed 0. Recorded as an optimistic assumption.",
}

ALL_COLUMNS = {**REQUIRED_COLUMNS, **OPTIONAL_COLUMNS}


class ContractError(ValueError):
    """The panel does not satisfy the contract."""


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of validating a panel against the contract."""

    n_rows: int
    n_units: int
    n_rounds: int
    missing_optional: list[str]
    assumptions: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_panel(panel: pd.DataFrame) -> ValidationReport:
    """Check a panel against the contract and report what had to be assumed.

    Raises ``ContractError`` when a required column is absent or a value
    breaks a hard invariant. Optional gaps are reported, not raised.
    """
    missing_required = [c for c in REQUIRED_COLUMNS if c not in panel.columns]
    if missing_required:
        raise ContractError(
            "Panel is missing required columns: " + ", ".join(sorted(missing_required))
        )
    if panel.empty:
        raise ContractError("Panel is empty.")

    warnings: list[str] = []

    # Hard invariants. These break the arithmetic downstream, so they raise.
    if (panel["target_pop"] < 0).any():
        raise ContractError("target_pop contains negative values.")
    if (panel["admin_vaccinated"] < 0).any():
        raise ContractError("admin_vaccinated contains negative values.")

    dup = panel.duplicated(subset=["unit_id", "round_code"]).sum()
    if dup:
        raise ContractError(f"Panel has {dup} duplicate (unit_id, round_code) rows.")

    bad_basis = set(panel["denominator_basis"].dropna().unique()) - {
        b.value for b in DenominatorBasis
    }
    if bad_basis:
        raise ContractError(f"Unknown denominator_basis values: {sorted(bad_basis)}")

    # Soft signals. These are the known failure modes of the source systems,
    # so they warn loudly and carry into the report.
    zero_denom = int((panel["target_pop"] == 0).sum())
    if zero_denom:
        warnings.append(
            f"{zero_denom} rows have target_pop = 0. Coverage is undefined for them "
            "and they are excluded from reach fitting."
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        admin_rate = panel["admin_vaccinated"] / panel["target_pop"].replace(0, np.nan)
    implausible = int((admin_rate > 1.10).sum())
    if implausible:
        warnings.append(
            f"{implausible} rows report administrative coverage above 110%. Treated as a "
            "denominator defect, not as success (MAP v1.0 3.6)."
        )

    n_basis = panel["denominator_basis"].nunique(dropna=True)
    if n_basis > 1:
        warnings.append(
            f"Panel mixes {n_basis} denominator bases. Coverage is only comparable within a basis; "
            "the engine stamps every output with the basis it used."
        )

    if "mlos_version" in panel.columns and panel["mlos_version"].nunique(dropna=True) > 1:
        warnings.append(
            "Panel spans more than one Master List of Settlements version. Cross-round trends "
            "may reflect a changed settlement set rather than changed coverage."
        )

    missing_optional = [c for c in OPTIONAL_COLUMNS if c not in panel.columns]
    assumptions = [f"{c}: {FALLBACKS[c]}" for c in missing_optional if c in FALLBACKS]

    if "verified_vaccinated" in missing_optional:
        warnings.append(
            "No independently verified numerator. Reach is inferred from administrative counts, "
            "which the programme measures as 18 pp optimistic nationally. Intervals are widened, "
            "but the point estimate keeps that bias."
        )

    return ValidationReport(
        n_rows=int(len(panel)),
        n_units=int(panel["unit_id"].nunique()),
        n_rounds=int(panel["round_code"].nunique()),
        missing_optional=missing_optional,
        assumptions=assumptions,
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Engine configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineConfig:
    """Every epidemiological constant the engine uses, in one place.

    The defaults are the Nigeria cVDPV2 programme values from MAP v1.0. Change
    them here rather than in the code, and the change appears in every report.
    """

    # --- Immunity target -------------------------------------------------
    r0: float = 6.0
    """Programme basic reproduction number for cVDPV2 (MAP v1.0 3.1)."""

    schedule_failure: float = 0.126
    """Failure of the delivered vaccine schedule, the epsilon in Vc(adj)."""

    doses_in_schedule: int = 3
    """Doses the schedule failure refers to. Used to reconcile it with per-dose take."""

    use_local_r0: bool = False
    """When True and r0_local is present, each unit gets its own immunity target.

    The default is False because MAP v1.0 3.1 makes Vc(adj) the single external
    referent. Local R0 is reported alongside as a secondary view.
    """

    # --- Vaccine take ----------------------------------------------------
    per_dose_take_mean: float = 0.500
    """Probability that one delivered dose immunises a susceptible child.

    Derived, not assumed: 1 - (1 - take)^doses = 1 - schedule_failure.
    ``reconcile_take`` checks this and reports the residual.
    """

    per_dose_take_sd: float = 0.060
    """Uncertainty on per-dose take. Propagated into the round count."""

    # --- Population ------------------------------------------------------
    crude_birth_rate: float = 0.0378
    """Annual births per head of total population, Nigeria. Used only as a fallback."""

    target_age_months: int = 60
    """Width of the target age band. Sets the rate at which children age out."""

    default_ri_coverage: float = 0.55
    """Fallback routine immunisation coverage of the birth cohort."""

    # --- Reach -----------------------------------------------------------
    admin_to_verified_ratio: float = 0.82
    """Programme-wide shrinkage from administrative to verified coverage.

    National eTally 97% against FIONET 79% (MAP v1.0 5.1) gives 0.81. Rounded
    to 0.82 and used only when no verified numerator exists.
    """

    default_lqas_lot_size: int = 60
    """Assumed LQAS sample size when verified_sample_n is absent."""

    # --- Reach drift -----------------------------------------------------
    estimate_reach_drift: bool = True
    """Measure how reach changes across successive rounds and project it forward.

    A model that assumes reach is stationary will predict too few rounds wherever
    teams and communities tire, which is most places. The drift is estimated from
    the panel rather than assumed, and it is reported.
    """

    reach_drift_floor: float = 0.60
    """Lower bound on the cumulative drift factor. Fatigue plateaus rather than
    compounding to zero, and letting it compound would manufacture infeasibility."""

    max_reach_wobble_cv: float = 0.40
    """Cap on the measured round-to-round movement in a settlement's own reach.

    Drift is where reach is heading; this is how far it strays on the way. The
    simulator needs it because stickiness decides which children a round misses,
    never how many, so without it every round performs exactly as well as the last
    and a draw either clears the target early or never clears it at all. The cap
    guards against a panel whose denominators move more than its campaigns do.
    """

    max_reach_drift_per_round: float = 0.05
    """Cap on the magnitude of the estimated drift, in either direction. A larger
    measured slope is almost always a changing settlement set or a changed
    verification method rather than a real trend in reach."""

    # --- Unreachable core ------------------------------------------------
    refusal_hard_core: float = 0.015
    """Share of the target population that refuses every round, national prior."""

    inaccessible_reach: float = 0.0
    """Reach achieved in a settlement flagged Inaccessible."""

    # --- Inference -------------------------------------------------------
    n_draws: int = 4000
    """Monte Carlo draws per unit. Sets the resolution of the round-count interval."""

    max_rounds: int = 12
    """Round budget. Beyond this the verdict is feasible_beyond_budget."""

    assurance: float = 0.90
    """Probability the plan must reach the target with. Drives the planning round count."""

    random_seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 < self.r0:
            raise ValueError("r0 must be positive.")
        if self.r0 <= 1.0:
            raise ValueError(
                "r0 must exceed 1. Below 1 the disease does not persist and no "
                "herd-immunity threshold is defined."
            )
        if not 0.0 <= self.schedule_failure < 1.0:
            raise ValueError("schedule_failure must lie in [0, 1).")
        if not 0.0 < self.per_dose_take_mean <= 1.0:
            raise ValueError("per_dose_take_mean must lie in (0, 1].")
        if not 0.0 < self.assurance < 1.0:
            raise ValueError("assurance must lie in (0, 1).")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1.")

    # -- derived quantities ------------------------------------------------

    @property
    def vc_adjusted(self) -> float:
        """Critical vaccination coverage adjusted for vaccine failure.

        Vc(adj) = (1 - 1/R0) / (1 - epsilon), MAP v1.0 3.1.
        """
        return (1.0 - 1.0 / self.r0) / (1.0 - self.schedule_failure)

    def vc_for_r0(self, r0_local: float) -> float:
        """Immunity target for a unit with its own reproduction number."""
        if r0_local <= 1.0:
            return 0.0
        return (1.0 - 1.0 / r0_local) / (1.0 - self.schedule_failure)

    def reconcile_take(self) -> dict[str, float]:
        """Check per-dose take against the schedule failure it must imply.

        The programme states epsilon = 0.126, that is a schedule efficacy of
        87.4%. A per-dose take of tau over d doses gives 1 - (1 - tau)^d. The
        two must agree. This method reports the residual instead of hiding it.
        """
        implied_schedule_efficacy = 1.0 - (1.0 - self.per_dose_take_mean) ** self.doses_in_schedule
        stated_schedule_efficacy = 1.0 - self.schedule_failure
        implied_take = 1.0 - stated_schedule_efficacy ** (1.0 / self.doses_in_schedule)
        # The inverse of 1 - (1-tau)^d = 1 - eps is tau = 1 - eps^(1/d).
        implied_take = 1.0 - self.schedule_failure ** (1.0 / self.doses_in_schedule)
        return {
            "per_dose_take_used": self.per_dose_take_mean,
            "per_dose_take_implied_by_epsilon": implied_take,
            "schedule_efficacy_implied_by_take": implied_schedule_efficacy,
            "schedule_efficacy_stated": stated_schedule_efficacy,
            "residual_pp": 100.0 * (implied_schedule_efficacy - stated_schedule_efficacy),
        }


@dataclass
class ProvenanceLog:
    """Record of every assumption the engine made, carried into the report.

    An assumption that is not written down becomes a fact by the time it
    reaches a decision maker. This class exists to stop that.
    """

    entries: list[dict[str, str]] = field(default_factory=list)

    def record(self, field_name: str, assumption: str, impact: str) -> None:
        self.entries.append(
            {"field": field_name, "assumption": assumption, "impact": impact}
        )

    def __len__(self) -> int:
        return len(self.entries)

    def to_frame(self) -> pd.DataFrame:
        if not self.entries:
            return pd.DataFrame(columns=["field", "assumption", "impact"])
        return pd.DataFrame(self.entries)
