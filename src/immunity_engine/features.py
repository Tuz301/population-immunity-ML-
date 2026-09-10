"""Feature construction for the reach model.

Every feature here is built from rounds strictly before the round it describes.
That rule is enforced by construction rather than checked afterwards, because a
single leaked feature makes the backtest look excellent and the field
performance look nothing like it.

The features answer one question: given what this place looked like before the
round, what share of its children will a team actually put a dose in?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import AccessibilityStatus, EngineConfig, ProvenanceLog

#: Accessibility as an ordinal, so the reach model can carry a monotone constraint.
ACCESSIBILITY_ORDINAL: dict[str, int] = {
    AccessibilityStatus.FULLY_ACCESSIBLE.value: 0,
    AccessibilityStatus.PARTIALLY_ACCESSIBLE.value: 1,
    AccessibilityStatus.INACCESSIBLE.value: 2,
}

#: Features whose effect on reach the programme knows the sign of. LightGBM is
#: constrained to respect them, which costs a little fit and buys a model that
#: cannot recommend a campaign because insecurity went up.
MONOTONE_CONSTRAINTS: dict[str, int] = {
    "accessibility_ordinal": -1,
    "security_compromised": -1,
    "nomadic_share": -1,
    "refusal_rate_prev": -1,
    "chronic_miss_count": -1,
    "teams_per_1000": +1,
    "reach_prev": +1,
    "reach_mean_prev3": +1,
}

FEATURE_COLUMNS: list[str] = [
    "log_target_pop",
    "accessibility_ordinal",
    "security_compromised",
    "nomadic_share",
    "refusal_rate_prev",
    "chronic_miss_count",
    "rounds_targeted_before",
    "rounds_since_last_targeted",
    "teams_per_1000",
    "reach_prev",
    "reach_mean_prev3",
    "reach_trend_prev",
    "admin_verified_gap_prev",
    "month_of_round",
    "npafp_rate",
    "stool_adequacy",
    "es_positive_90d",
    "distance_to_lga_centroid_km",
    "mlos_changed",
]

CATEGORICAL_COLUMNS: list[str] = ["campaign_type"]


def _observed_reach(panel: pd.DataFrame, config: EngineConfig, log: ProvenanceLog) -> pd.Series:
    """Best available estimate of the share of children a round actually reached.

    Verified counts are used where they exist. Where they do not, the
    administrative count is shrunk by the programme-wide verified-to-administrative
    ratio, and that substitution is recorded: it keeps the direction of the
    administrative bias even though it removes most of its size.
    """
    denom = panel["target_pop"].replace(0, np.nan)
    admin_rate = (panel["admin_vaccinated"] / denom).clip(upper=1.5)

    if "verified_vaccinated" in panel.columns:
        verified_rate = (panel["verified_vaccinated"] / denom).clip(upper=1.0)
        n_missing = int(verified_rate.isna().sum() - admin_rate.isna().sum())
        reach = verified_rate.fillna(admin_rate * config.admin_to_verified_ratio)
        if n_missing > 0:
            log.record(
                "verified_vaccinated",
                f"{n_missing} rounds had no verified numerator and were filled with "
                f"administrative coverage times {config.admin_to_verified_ratio}.",
                "Reach for those rounds keeps the administrative optimism; intervals widen.",
            )
    else:
        reach = admin_rate * config.admin_to_verified_ratio
        log.record(
            "verified_vaccinated",
            "No verified numerator anywhere in the panel. Reach is administrative "
            f"coverage times {config.admin_to_verified_ratio} throughout.",
            "Round counts are optimistic by an unknown amount. Treat them as a lower bound.",
        )

    return reach.clip(0.0, 1.0)


def build_features(
    panel: pd.DataFrame,
    config: EngineConfig,
    log: ProvenanceLog | None = None,
) -> tuple[pd.DataFrame, ProvenanceLog]:
    """Turn a validated panel into a leak-free feature matrix.

    Args:
        panel: Panel satisfying the contract in ``contracts``.
        config: Engine configuration.
        log: Provenance log to append to. A new one is created when omitted.

    Returns:
        A pair ``(features, log)``. ``features`` carries the identifier columns,
        every column in ``FEATURE_COLUMNS``, the categorical columns and the
        target column ``reach``.
    """
    log = log if log is not None else ProvenanceLog()
    df = panel.sort_values(["unit_id", "round_index"]).reset_index(drop=True).copy()

    df["reach"] = _observed_reach(df, config, log)

    # -- optional columns and their fallbacks ------------------------------
    if "accessibility_status" in df.columns:
        df["accessibility_ordinal"] = (
            df["accessibility_status"].map(ACCESSIBILITY_ORDINAL).fillna(0).astype(float)
        )
    else:
        df["accessibility_ordinal"] = 0.0
        log.record(
            "accessibility_status",
            "Absent. Every settlement assumed Fully Accessible.",
            "The unreachable core is understated, so round counts are optimistic.",
        )

    for column, fallback, impact in (
        ("security_compromised", 0.0, "Insecurity is invisible to the reach model."),
        ("nomadic_share", 0.0, "Mobile populations are treated as enumerated."),
        ("stool_adequacy", np.nan, "Surveillance quality is dropped as a predictor."),
        ("npafp_rate", np.nan, "Surveillance sensitivity is dropped as a predictor."),
        ("es_positive_90d", 0.0, "Recent environmental signal is treated as absent."),
    ):
        if column not in df.columns:
            df[column] = fallback
            log.record(column, "Absent from the panel.", impact)
    df["security_compromised"] = df["security_compromised"].astype(float)

    df["log_target_pop"] = np.log1p(df["target_pop"].clip(lower=0))

    if "refusal_count" in df.columns:
        refusal_rate = df["refusal_count"] / df["target_pop"].replace(0, np.nan)
    else:
        refusal_rate = pd.Series(np.nan, index=df.index)
        log.record(
            "refusal_count",
            "Absent. Refusals fall back to the national hard-core prior.",
            "Unit-level variation in refusal is invisible.",
        )

    if "teams_deployed" in df.columns:
        df["teams_per_1000"] = 1000.0 * df["teams_deployed"] / df["target_pop"].replace(0, np.nan)
    else:
        df["teams_per_1000"] = np.nan
        log.record("teams_deployed", "Absent.", "Delivery capacity is dropped as a predictor.")

    admin_rate = df["admin_vaccinated"] / df["target_pop"].replace(0, np.nan)
    df["admin_verified_gap"] = (admin_rate - df["reach"]).clip(-1.0, 1.5)

    df["month_of_round"] = pd.to_datetime(df["round_start"], errors="coerce").dt.month.astype(float)

    # -- lagged features, shifted within unit so no round sees itself ------
    grouped = df.groupby("unit_id", sort=False)
    df["reach_prev"] = grouped["reach"].shift(1)
    df["reach_mean_prev3"] = (
        grouped["reach"].shift(1).groupby(df["unit_id"], sort=False).rolling(3, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    df["reach_trend_prev"] = df["reach_prev"] - grouped["reach"].shift(2)
    df["admin_verified_gap_prev"] = grouped["admin_verified_gap"].shift(1)
    df["refusal_rate_prev"] = refusal_rate.groupby(df["unit_id"], sort=False).shift(1)

    df["rounds_targeted_before"] = grouped.cumcount().astype(float)

    prev_round_index = grouped["round_index"].shift(1)
    df["rounds_since_last_targeted"] = (df["round_index"] - prev_round_index).fillna(0.0)

    # Chronic miss: how often the unit fell below the round's own threshold in
    # the five rounds before this one. Uses the same rule the portal uses.
    below = (df["reach"] < 0.80).astype(float)
    df["chronic_miss_count"] = (
        below.groupby(df["unit_id"], sort=False).shift(1)
        .groupby(df["unit_id"], sort=False).rolling(5, min_periods=1).sum()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    if "mlos_version" in df.columns:
        df["mlos_changed"] = (
            grouped["mlos_version"].diff().fillna(0.0).ne(0.0).astype(float)
        )
    else:
        df["mlos_changed"] = 0.0

    df["distance_to_lga_centroid_km"] = _distance_to_lga_centroid(df, log)

    if "campaign_type" not in df.columns:
        df["campaign_type"] = "unknown"
    df["campaign_type"] = df["campaign_type"].astype("category")

    keep = [
        "unit_id", "grain", "lga_code", "state_code", "round_code", "round_index",
        "round_start", "target_pop", "denominator_basis", "reach",
        *FEATURE_COLUMNS, *CATEGORICAL_COLUMNS,
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy(), log


def _distance_to_lga_centroid(df: pd.DataFrame, log: ProvenanceLog) -> pd.Series:
    """Great-circle distance from each settlement to its LGA centroid, in km.

    Remoteness is one of the few reach predictors that is stable across rounds
    and cheap to compute, so it is worth the trigonometry.
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        log.record(
            "latitude/longitude",
            "Absent. Remoteness is dropped as a predictor.",
            "Hard-to-reach settlements look like easy ones to the reach model.",
        )
        return pd.Series(np.nan, index=df.index)

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    centroid = pd.DataFrame({"lga_code": df["lga_code"], "lat": lat, "lon": lon})
    means = centroid.groupby("lga_code")[["lat", "lon"]].transform("mean")

    lat1, lon1 = np.radians(lat), np.radians(lon)
    lat2, lon2 = np.radians(means["lat"]), np.radians(means["lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return pd.Series(6371.0 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1))), index=df.index)
