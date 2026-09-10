"""Read a panel straight out of the NEOC polio data portal.

The query below is written against the portal schema as of migration 0043:
``ref.settlement_version``, ``ref.campaign_round``, ``ref.target_population``,
``core.campaign_coverage``, ``core.iev_verification``, ``core.afp_indicator``
and ``core.es_sample``. It is a plain read; nothing here writes.

Two portal conventions are respected and must not be quietly dropped.

* Coverage carries its denominator through ``denominator_id``, so the
  denominator basis travels with every row rather than being chosen here.
* ``is_inaccessible`` is true only for settlements labelled Inaccessible.
  Partially Accessible settlements stay in the denominator, because removing
  them shrinks the denominator and inflates coverage.

Row-level security applies to whoever runs this. A state-scoped user gets a
state-scoped panel, and the engine reports on what it was allowed to see.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..contracts import ValidationReport, validate_panel

#: Settlement-grain panel. Bind ``%(round_codes)s`` to a tuple of round codes,
#: or pass ``None`` to take every round.
NEOC_PANEL_SQL = """
with rounds as (
    select id, round_code, campaign_type, start_date,
           row_number() over (order by coalesce(start_date, date '1900-01-01'), round_code)
             as round_index
    from ref.campaign_round
    where %(round_filter)s
),
es as (
    select s.lga_code,
           count(*) filter (
               where sm.result = 'positive'
                 and sm.collection_date > current_date - interval '90 days'
           ) as es_positive_90d
    from core.es_site s
    left join core.es_sample sm on sm.es_site_id = s.id
    group by s.lga_code
),
afp as (
    select distinct on (lga_code) lga_code, npafp_rate, stool_adequacy
    from core.afp_indicator
    order by lga_code, period_month desc
)
select
    c.settlement_sk::text            as unit_id,
    'settlement'                     as grain,
    w.lga_code                       as lga_code,
    l.state_code                     as state_code,
    r.round_code                     as round_code,
    r.round_index                    as round_index,
    r.start_date                     as round_start,
    r.campaign_type                  as campaign_type,
    tp.target_pop                    as target_pop,
    tp.basis::text                   as denominator_basis,
    c.vaccinated                     as admin_vaccinated,
    iev.verified_vaccinated          as verified_vaccinated,
    sv.accessibility_status::text    as accessibility_status,
    sv.security_compromised          as security_compromised,
    sv.is_inaccessible               as is_inaccessible,
    sv.latitude                      as latitude,
    sv.longitude                     as longitude,
    c.mlos_version                   as mlos_version,
    afp.npafp_rate                   as npafp_rate,
    afp.stool_adequacy               as stool_adequacy,
    coalesce(es.es_positive_90d, 0)  as es_positive_90d
from core.campaign_coverage c
join rounds r                     on r.id = c.round_id
join ref.target_population tp     on tp.id = c.denominator_id
join ref.settlement_version sv    on sv.settlement_sk = c.settlement_sk
                                 and sv.mlos_version = c.mlos_version
join ref.ward w                   on w.code = sv.ward_code
join ref.lga  l                   on l.code = w.lga_code
left join core.iev_verification iev on iev.settlement_sk = c.settlement_sk
                                   and iev.round_id = c.round_id
left join afp on afp.lga_code = w.lga_code
left join es  on es.lga_code  = w.lga_code
where c.source_code = %(source_code)s
order by unit_id, round_index
"""

#: Chronic-miss history, used to estimate how sticky reach is between rounds.
NEOC_CHRONIC_MISS_SQL = """
select settlement_sk::text as unit_id,
       lga_code,
       rounds_targeted,
       rounds_sub_threshold
from core.v_chronic_miss
where rounds_targeted >= 2
"""


def load_postgres_panel(
    connection: Any,
    *,
    source_code: str = "etally",
    round_codes: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, ValidationReport]:
    """Read the settlement panel and the chronic-miss history from the portal.

    Args:
        connection: Any DB-API connection that ``pandas.read_sql`` accepts, or a
            SQLAlchemy connectable.
        source_code: Coverage source system to read. The portal keeps sources
            side by side and never merges them, so one must be named.
        round_codes: Rounds to include. ``None`` takes every round.

    Returns:
        A triple ``(panel, chronic_miss, report)``.
    """
    params: dict[str, Any] = {
        "source_code": source_code,
        "round_filter": True if round_codes is None else None,
    }
    if round_codes is None:
        sql = NEOC_PANEL_SQL.replace("%(round_filter)s", "true")
        params.pop("round_filter")
    else:
        sql = NEOC_PANEL_SQL.replace("%(round_filter)s", "round_code in %(round_codes)s")
        params.pop("round_filter")
        params["round_codes"] = tuple(round_codes)

    panel = pd.read_sql(sql, connection, params=params)
    chronic = pd.read_sql(NEOC_CHRONIC_MISS_SQL, connection)

    if not panel.empty:
        panel["round_start"] = pd.to_datetime(panel["round_start"], errors="coerce")
        for column in ("security_compromised", "is_inaccessible"):
            if column in panel.columns:
                panel[column] = panel[column].fillna(False).astype(bool)

    return panel, chronic, validate_panel(panel)
