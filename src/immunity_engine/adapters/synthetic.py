"""A synthetic programme with known truth, for validating the engine.

Real panels cannot answer "was the round count right", because the counterfactual
was never run. A synthetic programme can: it simulates children, campaigns and
the reporting systems that observe them, keeps the true immunity trajectory, and
then hands the engine only what a real programme would see.

The generator is deliberately *not* the estimator with noise on top. It differs
from the estimator's assumptions in the ways real programmes differ:

* Reach drifts downward across rounds as teams and communities tire, while the
  estimator assumes it is stationary.
* A share of settlements over-report administratively, some of them grossly.
* Denominators are inflated by a settlement-specific factor the engine never sees.
* Accessibility flips between rounds as insecurity moves.
* Verified counts carry finite-sample noise from a small LQAS lot.

Recovery under that misspecification is the claim the validation makes. It is a
claim about the estimator, not about Nigeria. Field accuracy needs the field panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..contracts import AccessibilityStatus, DenominatorBasis


@dataclass
class SyntheticProgramme:
    """Parameters of the simulated programme.

    Defaults are shaped to the Nigeria cVDPV2 setting: a national administrative
    coverage near the high nineties against verified coverage near eighty, a
    minority of settlements insecure, and a birth rate that refills the target
    cohort in about five years.

    Attributes:
        n_states: Number of states.
        lgas_per_state: LGAs in each state.
        wards_per_lga: Wards in each LGA.
        settlements_per_ward: Settlements in each ward.
        n_rounds: Campaign rounds simulated and reported.
        forward_rounds: Extra rounds simulated after the reported window to
            establish the true round count. Not shown to the engine.
        interval_months: Months between rounds in the reported window.
        forward_interval_months: Months between rounds after the cut point. Outbreak
            response runs faster than routine campaigns, so the planning tempo is
            not the historical tempo.
        target_immunity: Threshold the true round count is measured against.
        per_dose_take: True per-dose take.
        ri_protection_mean: Mean share of newborns protected by routine immunisation.
        fatigue_per_round: Proportional fall in reach per successive round.
        falsifying_share: Share of settlements that over-report administratively.
        lqas_lot_size: Sample size behind each verified count.
        seed: Random seed.
    """

    n_states: int = 4
    lgas_per_state: int = 6
    wards_per_lga: int = 5
    settlements_per_ward: int = 8
    n_rounds: int = 8
    forward_rounds: int = 24
    interval_months: float = 6.0
    forward_interval_months: float = 3.0
    target_immunity: float = 0.9535
    per_dose_take: float = 0.50
    ri_protection_mean: float = 0.52
    fatigue_per_round: float = 0.015
    falsifying_share: float = 0.12
    lqas_lot_size: int = 60
    seed: int = 7

    age_band_months: int = 60
    n_cells: int = 48
    rho_true: float = 0.72
    kappa_true: float = 8.0

    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    @property
    def n_settlements(self) -> int:
        return (
            self.n_states
            * self.lgas_per_state
            * self.wards_per_lga
            * self.settlements_per_ward
        )


def _settlement_frame(spec: SyntheticProgramme) -> pd.DataFrame:
    """Build the geography and the latent attributes of every settlement."""
    rng = spec._rng
    rows = []
    for s in range(spec.n_states):
        # States differ systematically: some are structurally harder than others.
        state_quality = rng.normal(0.0, 0.55)
        for g in range(spec.lgas_per_state):
            lga_quality = state_quality + rng.normal(0.0, 0.45)
            centre_lat = 8.0 + 3.5 * rng.random()
            centre_lon = 4.0 + 8.0 * rng.random()
            for w in range(spec.wards_per_lga):
                for t in range(spec.settlements_per_ward):
                    rows.append(
                        {
                            "unit_id": f"S{s}-{g}-{w}-{t}",
                            "ward_code": f"W{s}-{g}-{w}",
                            "lga_code": f"LGA{s}-{g}",
                            "state_code": f"ST{s}",
                            "latent_quality": lga_quality + rng.normal(0.0, 0.7),
                            "latitude": centre_lat + rng.normal(0.0, 0.18),
                            "longitude": centre_lon + rng.normal(0.0, 0.18),
                        }
                    )
    df = pd.DataFrame(rows)
    n = len(df)

    # Reach quality maps a latent index onto a plausible verified-coverage range.
    df["mu_base"] = np.clip(0.79 + 0.115 * df["latent_quality"], 0.20, 0.98)

    df["nomadic_share"] = np.clip(rng.beta(1.3, 16.0, n), 0.0, 0.6)
    df["security_base"] = (rng.random(n) < np.clip(0.16 - 0.05 * df["latent_quality"], 0.01, 0.5))
    df["inaccessible_base"] = (rng.random(n) < np.clip(0.07 - 0.03 * df["latent_quality"], 0.0, 0.35))

    df["true_pop"] = np.clip(rng.lognormal(6.4, 0.75, n), 120, 40000).round()
    # Denominators are inflated, and worse where microplans are least supervised.
    df["denominator_inflation"] = np.clip(
        rng.lognormal(np.log(1.10) - 0.035 * df["latent_quality"], 0.16), 0.85, 2.2
    )
    df["ri_protection"] = np.clip(
        rng.normal(spec.ri_protection_mean + 0.06 * df["latent_quality"], 0.09), 0.05, 0.92
    )
    df["births_per_month"] = df["true_pop"] / spec.age_band_months * rng.normal(1.0, 0.05, n)
    df["falsifies"] = rng.random(n) < spec.falsifying_share
    # Administrative counts run ahead of verified ones everywhere, and far ahead
    # where reporting is being managed rather than measured.
    df["report_inflation"] = np.where(
        df["falsifies"], rng.uniform(1.60, 2.20, n), rng.uniform(1.18, 1.40, n)
    )
    df["initial_immunity"] = np.clip(
        rng.normal(0.38 + 0.09 * df["latent_quality"], 0.12, n), 0.02, 0.88
    )
    return df


def _reach_grid(mu: np.ndarray, kappa: float, pi_zero: np.ndarray, n_cells: int):
    """Discretise reach exactly as the engine does, so only the dynamics differ."""
    from ..vectorised import build_reach_grid

    return build_reach_grid(mu, np.full_like(mu, kappa), pi_zero, n_cells=n_cells)


def generate_synthetic_panel(
    spec: SyntheticProgramme | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a programme and return what it reports plus what was true.

    Args:
        spec: Programme parameters. Defaults are used when omitted.

    Returns:
        A pair ``(panel, truth)``.

        ``panel`` is a settlement-round panel satisfying the data contract. It
        holds only what a reporting system would produce.

        ``truth`` is one row per settlement with the true immunity at the end of
        the reported window, the true reach, the true unreachable core, and
        ``true_rounds_needed``: the number of further rounds at which true
        immunity first clears the target, or ``inf`` when it never does within
        ``forward_rounds``.
    """
    spec = spec or SyntheticProgramme()
    rng = spec._rng
    settlements = _settlement_frame(spec)
    n = len(settlements)

    from ..heterogeneity import unreachable_core

    # ---- latent state ----------------------------------------------------
    inaccessible = settlements["inaccessible_base"].to_numpy()
    pi_zero = np.array(
        [
            unreachable_core(
                inaccessible_share=1.0 if inaccessible[i] else 0.0,
                refusal_share=float(np.clip(rng.normal(0.018, 0.012), 0.0, 0.12)),
                nomadic_share=float(settlements["nomadic_share"].iloc[i]),
            )
            for i in range(n)
        ]
    )

    true_pop = settlements["true_pop"].to_numpy(dtype=float)
    births = settlements["births_per_month"].to_numpy(dtype=float)
    ri = settlements["ri_protection"].to_numpy(dtype=float)

    nodes, weights = _reach_grid(
        np.clip(settlements["mu_base"].to_numpy() / np.maximum(1 - pi_zero, 1e-6), 1e-3, 0.999),
        spec.kappa_true,
        pi_zero,
        spec.n_cells,
    )
    susceptible = (true_pop * (1.0 - settlements["initial_immunity"].to_numpy()))[:, None] * weights
    total = true_pop.copy()

    decay = float(np.exp(-spec.interval_months / spec.age_band_months))
    inflow = spec.age_band_months * (1.0 - decay)

    records: list[dict] = []
    base_date = pd.Timestamp("2023-03-01")

    def step(round_number: int, record: bool, interval: float) -> None:
        """Advance one round. ``record`` controls whether it is reported."""
        nonlocal susceptible, total

        step_decay = float(np.exp(-interval / spec.age_band_months))
        step_inflow = spec.age_band_months * (1.0 - step_decay)
        total = total * step_decay + births * step_inflow
        susceptible = (
            susceptible * step_decay + (births * (1.0 - ri) * step_inflow)[:, None] * weights
        )

        # Reach this round: the settlement's base, worn down by fatigue, nudged
        # by season, and shocked by whatever the round itself brought.
        fatigue = (1.0 - spec.fatigue_per_round) ** (round_number - 1)
        season = 1.0 + 0.05 * np.sin(2 * np.pi * ((round_number * interval) % 12) / 12.0)
        shock = rng.normal(1.0, 0.07, n)
        round_mu = np.clip(settlements["mu_base"].to_numpy() * fatigue * season * shock, 0.02, 0.99)

        # Accessibility moves between rounds; insecurity is not a fixed property.
        flipped = rng.random(n) < 0.06
        round_inaccessible = np.where(flipped, ~inaccessible, inaccessible)
        effective_mu = np.where(round_inaccessible, round_mu * 0.15, round_mu)

        reachable_mu = np.clip(effective_mu / np.maximum(1 - pi_zero, 1e-6), 1e-3, 0.999)
        round_nodes, round_weights = _reach_grid(reachable_mu, spec.kappa_true, pi_zero, spec.n_cells)

        # The population-mean per-round reach. This is the quantity the engine
        # tries to recover, so it is defined here once and everything observable
        # is derived from it.
        true_reached = np.clip((round_weights * round_nodes).sum(axis=1), 0.0, 1.0)

        susceptible = susceptible * (1.0 - spec.per_dose_take * round_nodes)
        remaining = susceptible.sum(axis=1)
        susceptible = (
            spec.rho_true * susceptible
            + (1.0 - spec.rho_true) * remaining[:, None] * round_weights
        )

        if not record:
            return

        # ---- what the reporting systems see ------------------------------
        reported_target = np.maximum(
            (true_pop * settlements["denominator_inflation"].to_numpy()).round(), 1.0
        )
        # Administrative counts are doses claimed against the microplan target.
        admin = np.minimum(
            (true_reached * total * settlements["report_inflation"].to_numpy()).round(),
            (reported_target * 1.15).round(),
        )
        # Verified counts come from a small lot, so they are unbiased but noisy.
        verified_hits = rng.binomial(spec.lqas_lot_size, true_reached)
        verified = (verified_hits / spec.lqas_lot_size * reported_target).round()

        status = np.where(
            round_inaccessible,
            AccessibilityStatus.INACCESSIBLE.value,
            np.where(
                settlements["security_base"].to_numpy(),
                AccessibilityStatus.PARTIALLY_ACCESSIBLE.value,
                AccessibilityStatus.FULLY_ACCESSIBLE.value,
            ),
        )

        records.append(
            {
                "unit_id": settlements["unit_id"].to_numpy(),
                "grain": "settlement",
                "ward_code": settlements["ward_code"].to_numpy(),
                "lga_code": settlements["lga_code"].to_numpy(),
                "state_code": settlements["state_code"].to_numpy(),
                "round_code": f"2023-R{round_number:02d}",
                "round_index": round_number,
                "round_start": base_date + pd.DateOffset(months=int(spec.interval_months * (round_number - 1))),
                "campaign_type": "nOPV2-SIA" if round_number % 3 else "bOPV-SIA",
                "target_pop": reported_target,
                "denominator_basis": DenominatorBasis.MICROPLAN.value,
                "admin_vaccinated": admin,
                "verified_vaccinated": verified,
                "verified_sample_n": spec.lqas_lot_size,
                "accessibility_status": status,
                "security_compromised": settlements["security_base"].to_numpy(),
                "is_inaccessible": round_inaccessible,
                "refusal_count": rng.binomial(
                    reported_target.astype(int), np.clip(rng.normal(0.02, 0.01, n), 0.0, 0.15)
                ),
                "latitude": settlements["latitude"].to_numpy(),
                "longitude": settlements["longitude"].to_numpy(),
                "mlos_version": 3,
                "births_per_month": births,
                "ri_coverage": ri,
                "nomadic_share": settlements["nomadic_share"].to_numpy(),
                "teams_deployed": np.maximum((reported_target / 500).round(), 1),
                "npafp_rate": np.nan,
                "stool_adequacy": np.nan,
                "es_positive_90d": 0,
            }
        )

    for r in range(1, spec.n_rounds + 1):
        step(r, record=True, interval=spec.interval_months)

    immunity_at_cut = np.clip(1.0 - susceptible.sum(axis=1) / total, 0.0, 1.0)
    susceptible_cut, total_cut = susceptible.copy(), total.copy()

    # ---- ground truth: keep running and see when truth clears the target --
    true_rounds = np.full(n, np.inf)
    for r in range(spec.n_rounds + 1, spec.n_rounds + spec.forward_rounds + 1):
        step(r, record=False, interval=spec.forward_interval_months)
        immunity = np.clip(1.0 - susceptible.sum(axis=1) / total, 0.0, 1.0)
        newly = (immunity >= spec.target_immunity) & ~np.isfinite(true_rounds)
        true_rounds[newly] = r - spec.n_rounds

    true_ceiling = np.clip(1.0 - susceptible.sum(axis=1) / total, 0.0, 1.0)

    panel = pd.concat([pd.DataFrame(chunk) for chunk in records], ignore_index=True)
    panel = panel.sort_values(["unit_id", "round_index"]).reset_index(drop=True)

    truth = pd.DataFrame(
        {
            "unit_id": settlements["unit_id"],
            "lga_code": settlements["lga_code"],
            "state_code": settlements["state_code"],
            "true_immunity_at_cut": immunity_at_cut,
            "true_pop": true_pop,
            "true_mu_base": settlements["mu_base"],
            "true_pi_zero": pi_zero,
            "true_denominator_inflation": settlements["denominator_inflation"],
            "true_ri_protection": ri,
            "true_ceiling": true_ceiling,
            "true_rounds_needed": true_rounds,
        }
    )
    # Restore the cut-point state so callers can reuse the generator deterministically.
    susceptible, total = susceptible_cut, total_cut
    return panel, truth
