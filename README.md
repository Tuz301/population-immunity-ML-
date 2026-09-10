# Immunity Engine

[![ci](https://github.com/Tuz301/population-immunity-ML-/actions/workflows/ci.yml/badge.svg)](https://github.com/Tuz301/population-immunity-ML-/actions/workflows/ci.yml)

**Repository:** https://github.com/Tuz301/population-immunity-ML-

How many campaign rounds does a location need to reach population immunity — and
where will no number of rounds do it?

The engine answers both questions for a settlement, a ward or an LGA. It gives a
round count with a stated confidence, and it names the locations where the round
count is the wrong answer to the question.

It is built for the Nigeria cVDPV2 outbreak response. It reads the NEOC polio
data portal schema directly. It works with any panel that meets the data
contract.

---

## What it does that a round count alone does not

**It refuses to give a number when a number would be wrong.** A share of children
in some locations are reached in no round at all. Those children cap population
immunity permanently. The engine finds the cap, compares it against the target,
and returns *infeasible by campaigns* instead of a round count. On a simulated
programme with known truth, it identifies 94% of those locations, and 81% of the
locations it flags really are unreachable.

**It separates two failures that look identical in a spreadsheet.** A location can
miss the target because children cannot be reached, or because births refill the
susceptible stock faster than rounds drain it. The first needs access
negotiation. The second needs a shorter interval or stronger routine
immunisation. Neither needs more rounds at the current interval. The engine names
which one binds.

**It separates fatigue from structure.** Reach falls round on round in most
panels. A shortfall caused by falling reach has an owner and a remedy. A
shortfall caused by unreachable children does not. The engine runs both
scenarios and reports them apart.

**It answers "how often", not only "how many".** Reaching the target once is not
the same as holding it. The engine reports the longest round interval whose
trough still clears the target, and returns nothing when no practical interval
holds it.

**It states what would sharpen the answer.** Every location carries a binding
constraint: the input whose uncertainty drives its round count most. That is
where the next measurement belongs.

---

## Results on a simulated programme

The simulated programme has 960 settlements and 8 reported rounds. Truth is
known, so the round count can be scored. The generator is not the estimator with
noise added: reach drifts downward, some settlements over-report, denominators
are inflated by a factor the engine never sees, accessibility flips between
rounds, and verified counts carry finite-sample noise.

| Measure | Result |
|---|---|
| Round count, mean absolute error | 1.42 rounds |
| Round count, median absolute error | 1 round |
| Predicted within one round | 65% |
| Round count, rank correlation with truth | 0.61 |
| Reconstructed current immunity, mean absolute error | 0.032 |
| Unreachable verdict, recall | 94% |
| Unreachable verdict, precision | 81% |
| Round-sufficiency Brier score | 0.106 (0.25 is uninformative) |
| 80% round-count interval, observed coverage | 93% (target 80%) |
| Reach forecast, out-of-sample error | 0.066 reach points |
| Reach forecast, skill over persistence | +51% |
| Reach forecast, 80% interval coverage | 71% (target 80%) |

Read these numbers as a claim about the estimator, not about Nigeria. They show
that the engine recovers a known answer under known misspecification. Field
accuracy needs the field panel.

Three results are stated rather than tuned away.

The 80% round-count interval covers the truth 93% of the time, so the engine
under-claims what it knows there. Tuning that against the validation truth would
be fitting to the test.

The reach model's own 80% interval covers only 71% of held-out observations, so
it claims more than it knows. The engine prints this as a calibration verdict and
the council raises it as a finding on every recommendation. Read the round count
as a range, not as a confidence statement, until that verdict passes.

The mid-range round-sufficiency probabilities remain a few points
over-confident: where the engine says 55%, the truth is nearer 40%.

Reproduce them:

```bash
python scripts/run_validation.py
```

---

## Install

```bash
git clone https://github.com/Tuz301/population-immunity-ML-.git
cd population-immunity-ML-
pip install -e ".[dev]"
```

Optional extras:

- `[council]` adds the language-model pass of the expert council.
- `[postgres]` adds the NEOC portal adapter.

---

## Use

```bash
# Round requirement per unit, with the escalation table
python -m immunity_engine plan --panel panel.csv

# One unit in full
python -m immunity_engine plan --panel panel.csv --unit LGA1-2

# Spread a fixed round budget across units
python -m immunity_engine allocate --panel panel.csv --budget 40

# Expert review of the highest-priority recommendations
python -m immunity_engine council --panel panel.csv --limit 5

# Score the reach model on held-out rounds of your own panel
python -m immunity_engine backtest --panel panel.csv

# Run against a simulated programme with known truth
python -m immunity_engine validate
```

From Python:

```python
from immunity_engine import EngineConfig, run_pipeline
from immunity_engine.report import escalation_table, programme_summary

plan_set = run_pipeline(panel, EngineConfig(), planning_interval_months=3.0)

print(programme_summary(plan_set, EngineConfig()))
print(escalation_table(plan_set))          # where rounds are the wrong instrument

for plan in plan_set.plans:
    print(plan.headline())
```

Read the NEOC portal directly:

```python
from immunity_engine.adapters import load_postgres_panel

panel, chronic_miss, report = load_postgres_panel(connection, source_code="etally")
```

The adapter is a plain read. Row-level security applies to whoever runs it: a
state-scoped user gets a state-scoped panel.

---

## How it works

Five layers. Only one of them is machine learning.

| Layer | Question | Method |
|---|---|---|
| 1. Reach | How well will a team do here next round? | LightGBM, monotone constraints, conformal intervals |
| 2. Heterogeneity | Which children does a round reach, and are they the same children each time? | Beta reach distribution with an unreachable core; stickiness from the intraclass correlation of reach |
| 3. Stock | What does a round move? | Susceptible stock with births, ageing and campaign pulses |
| 4. Inversion | How many rounds, and can any number do it? | Monte Carlo over the parameter posterior, run to convergence |
| 5. Council | Should this recommendation go forward? | Six mandated seats, deterministic rules first, language model second |

The round count is not regressed. A regression on past round counts learns the
schedule that was run, not the requirement. The engine builds the susceptible
stock, runs it forward, and records the round at which immunity crosses the
threshold, for thousands of parameter draws.

Full detail: [`docs/METHOD.md`](docs/METHOD.md).

---

## The council

Six seats, each owning failure modes that no other seat covers:

| Seat | Owns |
|---|---|
| Field Epidemiologist | Whether the target is right for this place, and whether reaching it interrupts transmission |
| Immunisation Programme Manager | Whether the plan can be executed at the tempo stated |
| Adversarial Biostatistician | Whether the numbers support the confidence attached to them |
| Access and Security Analyst | Whether the children counted on can in fact be reached |
| Systems Analyst | Which stock and which loop the recommendation acts on |
| Data Integrity Auditor | Whether the inputs mean what the engine assumed |

Three rules govern the record:

1. One block inside a seat's own mandate holds the recommendation. The seats do
   not overlap, so a blocking seat is the only one looking at that failure.
2. Overall confidence is the lowest confidence any seat reported, not the mean.
3. Dissent is published with the recommendation, not filed behind it.

The deterministic rule reviewer always runs. Every check it makes is a threshold
this programme has been caught by before, written as code. It needs no network,
no key and no model, so the safety findings of a run never depend on a service
being up.

The language-model pass runs in addition when `ANTHROPIC_API_KEY` is set and
`--use-model` is given. It reads the same evidence packet and reasons about what
a threshold cannot express. It cannot silently overturn a rule finding: both
reach the adjudicator. Seats are polled independently and never see each other's
answers, because a council that converges has stopped doing its job.

A seat that could not be polled is recorded as unreviewed. It is never treated as
assent.

Every recommendation carries a systems block: the stock it moves, the loop it
acts on, the delay in that loop, the leverage tier, and the named trap it could
fall into.

---

## Data contract

Required columns:

`unit_id`, `grain`, `lga_code`, `state_code`, `round_code`, `round_index`,
`round_start`, `campaign_type`, `target_pop`, `denominator_basis`,
`admin_vaccinated`

The engine stops if one is absent.

Optional columns sharpen the answer. Each has a documented fallback, and every
use of a fallback is recorded in the provenance log and printed in the report.
The most important is `verified_vaccinated`. Without it, reach rests on
administrative counts, which the programme measures as 18 percentage points
optimistic nationally. The engine widens its intervals but keeps that bias, and
says so.

Full contract: `src/immunity_engine/contracts.py`.

---

## Limits

1. **Round counts have not been validated against field outcomes.** They cannot
   be, without a location where the counterfactual was run. Every accuracy figure
   above comes from a simulated programme.
2. **Prediction intervals are too wide.** The engine under-claims what it knows.
3. **Spatial transmission is not modelled.** Neighbouring locations are treated
   as independent. A location can reach the target and be reinfected from next
   door.
4. **Clustering inside a location is not modelled.** 95% immunity with the
   remaining susceptibles in one ward is not 95% spread evenly.
5. **Reach beyond about six consecutive rounds is extrapolation.** No location in
   the training data has run more.
6. **The engine produces advice.** It does not authorise a classification or a
   campaign calendar. A named human approver does.

---

## Tests

```bash
python -m pytest -q
```

The tests protect modelling claims, not plumbing. Among them: Gauss-Jacobi
quadrature is exact to machine precision against an independent moment
expansion; stickiness never makes the answer look better; the unreachable core
survives every round; immunity without campaigns settles at routine immunisation
coverage; a higher target never needs fewer rounds; the council cannot be talked
round.
