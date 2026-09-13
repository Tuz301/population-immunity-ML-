# CLAUDE.md

Guidance for coding agents working in this repository.

## What this is

An engine that answers, per settlement / ward / LGA: **how many campaign rounds
are needed to reach population immunity, and where no number of rounds will do
it.** Built for the Nigeria cVDPV2 outbreak response; reads the NEOC polio data
portal schema.

The second half of that sentence is the point of the project. A round count is
easy to produce and easy to produce wrongly. The verdict that campaigns are the
wrong instrument is the output that changes what a programme does.

## Commands

```bash
pip install -e ".[dev]"          # install, with test deps
python -m pytest -q              # 46 tests, ~20s
python -m immunity_engine plan       --draws 2000    # round requirement per unit
python -m immunity_engine allocate   --budget 40     # spread a round budget
python -m immunity_engine council    --limit 5       # expert review
python -m immunity_engine backtest                   # score reach on held-out rounds
python -m immunity_engine validate                   # run against known truth
python scripts/run_validation.py                     # full validation report
python scripts/pipeline_end_to_end.py                # annotated ingestion-to-output walk
```

Omit `--panel` on any command and a simulated programme is generated. That is
the only way to run without real data, and it is how CI exercises the engine.

## Architecture

Five layers. Only one is machine learning; do not add more without a reason.

| Module | Layer | Responsibility |
|---|---|---|
| `contracts.py` | — | Data contract, `EngineConfig`, provenance log |
| `adapters/` | Ingestion | Postgres (NEOC schema), CSV, synthetic generator |
| `features.py` | Features | Leak-free, lag-shifted feature matrix |
| `reach_model.py` | 1. Reach | LightGBM + conformal residual intervals |
| `heterogeneity.py` | 2. Heterogeneity | Beta reach distribution, unreachable core, stickiness |
| `immunity.py`, `vectorised.py` | 3. Stock | Susceptible-stock dynamics (readable / batched) |
| `rounds_required.py` | 4. Inversion | Monte Carlo → round count + feasibility verdict |
| `optimizer.py` | — | Budget allocation across units (greedy, submodular) |
| `council/` | 5. Council | Six mandated seats, rules first, model second |
| `pipeline.py` | — | Wires 1→4 |
| `report.py` | — | Human-readable output |
| `validation.py` | — | Backtest and scoring |

`immunity.py` is the readable reference implementation. `vectorised.py` is the
same model batched across Monte Carlo draws, and is what actually runs.
`tests/test_science.py` holds them to agreement. **If you change one, change
both.**

## Invariants — do not break these

The tests in `tests/test_science.py` protect modelling claims, not plumbing.
Each guards a property that a round count depends on:

1. **Gauss-Jacobi quadrature is exact**, verified to ~1e-15 against an
   independent moment expansion. `(1 - tau*r)^N` is a degree-N polynomial, so
   the integral against the Beta density is exact, not approximate.
2. **Stickiness never makes a place look easier.** Jensen's inequality
   guarantees it; a sign error here under-counts rounds everywhere.
3. **The unreachable core survives every round.** If unlimited rounds ever drive
   susceptibles to zero, the engine recommends campaigns for places that need
   access negotiation.
4. **Without campaigns, immunity settles at routine-immunisation coverage.**
   This is the baseline the history replay starts from.
5. **A higher target never needs fewer rounds.**
6. **Runs are reproducible.** Every figure is Monte Carlo; a seeded run must
   reproduce exactly. CI enforces this.

## Gotchas that cost real time

**LightGBM refuses monotone constraints twice.** Not under a `quantile`
objective, and not alongside a native categorical. Hence the current design: a
constrained squared-error centre model, a separate scale model, and quantiles
rebuilt from a conformal calibration slice; categoricals are one-hot encoded.
Neither refusal is reachable from a unit test of the maths — both only appear
when a real panel goes through the real model. This is why CI has a separate
end-to-end job.

**Seven estimator biases were found during validation.** All are fixed; all are
documented in `docs/METHOD.md` §8. Do not reintroduce them:

- Stickiness (`rho`) from a pass-or-fail label reads far too low (0.38 against a
  true 0.72). Thresholding discards how far below the line a place sits. Use the
  intraclass correlation of the *continuous* reach series.
- Reach spread (`kappa`) must have verification sampling variance removed. With
  an LQAS lot of 60, sampling variance is the same order as the real
  between-settlement variance.
- Intermittent closure is **not** a permanent unreachable core. A settlement
  closed in 6% of rounds is reachable; counting that as 6% permanently
  unreachable dropped precision on the infeasibility verdict to 47%.
- Stickiness is also dragged down by movement it does not cause. It decides
  *which* children a round misses, never *how many*, so round-to-round movement
  in a settlement's own aggregate reach is not evidence about it. Removing that
  movement lifts the estimate from 0.66 to 0.70 against a true 0.72. The
  correction only ever pushes upward, so its direction is known.
- Round-to-round movement measured from **one** reported stream is mostly
  reporting noise: 22% against a true 7%. Feeding that to the simulator makes the
  engine worse, because an inflated shock lets a draw clear the target on one
  good round. Use the covariance of the administrative and verified streams,
  whose errors are independent, and take the median across settlements so that
  places which close and reopen do not set the figure for everywhere else. That
  reads 8.4%.
- The predicted reach spread must have that same round-to-round movement taken
  back out before the inversion holds it as a level. The model states what one
  round will return; the inversion draws a level once, keeps it for every round
  of a draw, and applies the movement separately. Hand it the single-round spread
  and the movement is counted twice. At the median settlement **53% of the reach
  predictive variance is movement**, so the level's own spread was half again too
  wide.
- The uncertainty on reconstructed current immunity must be propagated, not
  asserted. It was a flat 0.02 for every settlement, produced by perturbing the
  reach history alone by one verification lot's sampling error and then flooring
  the result — and the floor was doing all the work. The true error is strongly
  heteroscedastic: near the ceiling a shift in reach barely moves immunity, while
  far from it every input does. Measured against truth the stated figure was
  **2.4x too cautious at the top of the range**. Rebuild the replay under every
  input it depends on, at the dispersions the inversion already states for those
  same quantities.

**Never let the spread model score its own training residuals.** The interval
width is the shape of the standardised residual. Fitting the scale model on a
slice and then dividing that slice's residuals by that model's own predictions
understates them, and every interval built from it runs narrow — 71% coverage at
a nominal 80%. The failure does not show up where it is made: coverage on the
slice that produced it looks correct. The calibration slice is therefore split
again by round, so the shape is read off rows that taught neither model. That
holds all three reported levels within tolerance.

**The mid-range probability gap is a point-estimate defect, not a calibration
defect.** The engine states 55% where 40% happens, through the middle of its
range. Six hypotheses about the spread were implemented and measured; five were
rejected and the sixth, the conformal self-scoring leak, was real but moved the
middle not at all. The question was settled by handing the engine the true
current immunity and changing nothing else: round-count MAE fell 1.37 to 0.78,
and the gap at 0.45, 0.55 and 0.65 went to -0.01, +0.00 and +0.02. The middle is
mis-stated because immunity is reconstructed with error, not because the
probability is built wrongly from it. **Do not attempt to close this gap by
re-tuning the Monte Carlo.** It closes by measuring immunity — a serosurvey or
LQAS that anchors the starting stock — which is also what the engine's own
`binding_constraint` names for these units.

**Scores move about 0.02 of Brier between synthetic seeds.** Measured across
seeds 11, 23 and 37: 0.0933, 0.0883, 0.1081, an sd of 0.010 and a range of 0.020.
Any claimed improvement or regression smaller than that is noise. Interval
coverage is far steadier, at 88.4% +/- 1.4, so it is the better signal to steer
a calibration change by.

**The calibration slice must match the deployment horizon.** A residual measured
one round ahead is smaller than one measured three rounds ahead. Calibrating at
one horizon and deploying at another produces intervals that are too narrow
exactly when they matter.

## House rules

- **No coverage figure without its denominator basis.** The contract enforces
  this; `ref.denominator_basis` travels with every row. Coverage above 100% is a
  denominator defect, never success.
- **Record every assumption.** Optional columns have documented fallbacks; each
  use is written to the `ProvenanceLog` and printed in the report. An assumption
  that is not written down becomes a fact by the time it reaches a decision maker.
- **State miscalibration, do not tune it away.** The round-count interval
  currently over-covers and the reach interval under-covers. Both are reported.
  Tuning either against the validation truth would be fitting to the test.
- **The engine produces advice.** It does not authorise a classification or a
  campaign calendar. Every report says so; keep it that way.
- **Comments explain why, not what.** Match the existing density — the modules
  carry the reasoning behind a modelling choice, not a narration of the code.

## Validation, and what it does not claim

`adapters/synthetic.py` generates a programme with known truth. It is
deliberately **not** the estimator with noise added: reach drifts downward, some
locations over-report, denominators are inflated by an unseen factor,
accessibility flips between rounds, verified counts carry finite-sample noise.
Recovery under that misspecification is the claim.

**Round counts have never been validated against field outcomes** and cannot be
without a location where the counterfactual was run. Every accuracy figure in
this repository is a claim about the estimator, not about Nigeria. Do not let
that qualifier get dropped from a README, a report or a PR body.

`validation.backtest_on_panel` is the one that runs on real data. It scores
reach, not round count, because on a real panel the true round count is never
observed.

## Not modelled

Spatial transmission (neighbours are independent), susceptible clustering within
a location, and reach beyond about six consecutive rounds (extrapolation — no
location in training data has run more).
