# Immunity Engine — complete handover

Everything a new team needs to pick this work up: what the engine is, what it
does, what has been measured, what was learned the hard way, and what is left.

**Repository.** https://github.com/Tuz301/population-immunity-ML-
**Branch.** `claude/campaign-optimization-immunity-yne6vw`, pull request #1, open as a draft.
**Context.** Nigeria cVDPV2 outbreak response. Reads the NEOC polio data portal schema.

Read sections 1 to 3 to understand the engine. Read section 6 before you plan any
work. Section 9 lists the first moves.

---

## 1. The question, and why it is inverted

The engine answers two questions for a settlement, a ward or an LGA.

1. How many campaign rounds does this place need to reach population immunity?
2. Where will no number of rounds do it?

The second question is the point of the project. A round count is easy to produce
and easy to produce wrongly. The verdict that campaigns are the **wrong
instrument** is the output that changes what a programme does.

### The core design decision

The round count is **not regressed**. There is no model that maps features to a
round count directly.

The reason is not statistical taste. A round count is not a property of a place.
It is the answer to a threshold-crossing problem over a stock that births refill,
and in which a bounded share of children can never be reached. A regression on
historical round counts learns the schedule a programme ran. It does not learn
the requirement that programme faced. Those two things differ exactly where the
programme was wrong, which is where the engine is supposed to help.

So the engine inverts instead:

1. Build the susceptible stock for the settlement.
2. Draw the uncertain parameters.
3. Run the stock forward under each draw.
4. Record the round at which immunity first crosses the threshold.
5. Report the distribution of that round.

Where the immunity ceiling sits below the threshold in a draw, no round clears
it. That draw votes for the verdict that campaigns are not the instrument.

---

## 2. The science

Full derivations are in `docs/METHOD.md`. This is the shape of the argument.

### 2.1 The immunity target

The target is the herd-immunity threshold adjusted for schedule failure.

```
Vc(adj) = (1 − 1/R₀) / (1 − ε)
```

With R₀ = 6.0 and ε = 0.126, the target is **95.35%**.

Per-dose take is derived rather than assumed. If a three-dose schedule fails with
probability ε, then `1 − (1 − take)³ = 1 − ε`, which gives take ≈ 0.50. The
engine reconciles the two parameters and reports the residual.

`use_local_r0` is off by default, because the programme's MAP makes Vc(adj) the
single external referent. Local R₀ is reported alongside as a secondary view.

### 2.2 Reach is a distribution, not a rate

A round that "reaches 70%" does not reach 70% of every child equally. Two
structures matter and they are different.

**The unreachable core, π₀.** A share of children are reached in *no* round.
They cap population immunity permanently. The core is built from three
overlapping routes — inaccessibility, refusal and mobility — combined through
their complements, because a settlement that is both insecure and nomadic is one
unreachable population, not two.

**Concentration, κ.** Among reachable children, reach follows a Beta
distribution. Low κ means the same children are reached repeatedly and the rest
are missed repeatedly.

**Stickiness, ρ.** This decides *which* children a round misses, never *how
many*. It is estimated from the intraclass correlation of the continuous reach
series. This distinction is load-bearing and is the source of two of the seven
biases in section 5.

The integral of `(1 − τ·r)^N` against the Beta density is computed by
Gauss-Jacobi quadrature. Because the integrand is a degree-N polynomial, the
quadrature is **exact**, not approximate — verified to about 1e-15 against an
independent moment expansion.

### 2.3 The susceptible stock

Children age into the target band, births refill the susceptible pool between
rounds, routine immunisation protects a share of newborns, and each campaign
round applies a pulse that depends on reach, take and stickiness.

`immunity.py` is the readable reference implementation. `vectorised.py` is the
same model batched across Monte Carlo draws, and is what actually runs.
`tests/test_science.py` holds the two to agreement. **Change one, change both.**

### 2.4 Two ceilings, and why both are reported

**The core-limited ceiling** is the highest immunity a closed cohort could reach,
limited only by the unreachable core.

**The schedule ceiling** is the highest immunity this round interval ever
reaches, once repeated rounds and births balance. It sits below the core ceiling,
and it is the one that decides feasibility.

The gap between the two separates an **access problem** from a
**round-frequency problem**. Those have different owners and different remedies.
Reporting only one would send the response to the wrong place.

### 2.5 Fatigue against structure

Reach falls round on round in most panels. The engine runs the inversion twice.

- The **planning run** projects the measured decline forward. Its round count is
  what this programme should expect.
- The **structural run** holds reach at today's level. Only this run may return
  the verdict that campaigns are the wrong instrument.

A shortfall caused by teams tiring is a shortfall someone can act on. Telling a
programme its access is hopeless when its problem is fatigue sends the response
to the wrong place entirely.

---

## 3. The code

### Architecture

Five layers. **Only one is machine learning.** Do not add more without a reason.

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
| `pipeline.py` | — | Wires layers 1 to 4 |
| `report.py` | — | Human-readable output |
| `validation.py` | — | Backtest and scoring |

### The reach model, and why it is built this way

LightGBM refuses monotone constraints twice: not under a `quantile` objective,
and not alongside a native categorical. The model must never forecast *better*
reach because insecurity rose or because a settlement was missed more often.
Those signs are known, and a planning model that gets them backwards is worse
than no model.

Hence the current design: a constrained squared-error **centre** model, a
separate **scale** model, and quantiles rebuilt from a conformal calibration
slice. Categoricals are one-hot encoded.

Validation is **forward in time**. Random k-fold on a panel like this scores a
model on rounds it has already seen in neighbouring settlements, and reports a
skill the field will never see.

### The council

Six seats, each owning failure modes no other seat covers: Field Epidemiologist,
Immunisation Programme Manager, Adversarial Biostatistician, Access and Security
Analyst, Systems Analyst, Data Integrity Auditor.

Three rules govern the record.

1. One block inside a seat's own mandate holds the recommendation. The seats do
   not overlap, so a blocking seat is the only one looking at that failure.
2. Overall confidence is the **lowest** any seat reported, not the mean.
3. Dissent is published with the recommendation, not filed behind it.

The deterministic rule reviewer always runs and needs no network, no key and no
model. The language-model pass runs in addition when `ANTHROPIC_API_KEY` is set
and `--use-model` is given. It cannot silently overturn a rule finding: both
reach the adjudicator. Seats are polled independently and never see each other's
answers, because a council that converges has stopped doing its job. A seat that
could not be polled is recorded as unreviewed, never as assent.

### Commands

```bash
pip install -e ".[dev]"
python -m pytest -q                                  # 53 tests

python -m immunity_engine plan     --draws 2000      # round requirement per unit
python -m immunity_engine allocate --budget 40       # spread a round budget
python -m immunity_engine council  --limit 5         # expert review
python -m immunity_engine backtest                   # score reach on held-out rounds
python -m immunity_engine validate                   # run against known truth

python scripts/run_validation.py                     # full validation report
python scripts/pipeline_end_to_end.py                # annotated ingestion-to-output walk
```

Omit `--panel` on any command and a simulated programme is generated. That is the
only way to run without real data, and it is how CI exercises the engine.

### Data contract

**Required.** `unit_id`, `grain`, `lga_code`, `state_code`, `round_code`,
`round_index`, `round_start`, `campaign_type`, `target_pop`,
`denominator_basis`, `admin_vaccinated`. The engine stops if one is absent.

**The optional column that changes the answer.** `verified_vaccinated`. Without
it, reach rests on administrative counts, which the programme measures as 18
percentage points optimistic nationally. More important, the engine separates
campaign movement from reporting noise using the covariance of the two streams.
With one stream only, that estimate reads 22% against a true 7%, and feeding it
to the simulator makes the engine measurably **worse**.

**Also valuable.** `verified_sample_n`, `accessibility_status`, `refusal_count`,
`births_per_month`, `ri_coverage`, `nomadic_share`, `latitude`, `longitude`,
`r0_local`, `security_compromised`, `is_inaccessible`, `mlos_version`,
`teams_deployed`, `npafp_rate`, `stool_adequacy`, `es_positive_90d`.

Every optional column has a documented fallback. Every use of a fallback is
written to the `ProvenanceLog` and printed in the report. An assumption that is
not written down becomes a fact by the time it reaches a decision maker.

---

## 4. What is measured today

All figures come from `adapters/synthetic.py`: 960 settlements, 8 reported
rounds. The generator is **not** the estimator with noise added. Reach drifts
downward, some settlements over-report, denominators carry an unseen inflation
factor, accessibility flips between rounds, and verified counts carry
finite-sample noise. Recovery under that misspecification is the claim.

| Measure | Result | Target |
|---|---|---|
| Round count, mean absolute error | 1.37 rounds | — |
| Round count, median absolute error | 1 round | — |
| Predicted within one round | 66% | — |
| Round count, rank correlation with truth | 0.63 | — |
| Reconstructed current immunity, mean absolute error | 0.032 | — |
| Unreachable verdict, recall | 95% | — |
| Unreachable verdict, precision | 80% | — |
| Round-sufficiency Brier score | 0.109 | 0.25 is uninformative |
| 80% round-count interval, observed coverage | 88% | 80% |
| Reach forecast, held-out error | 0.0657 reach points | — |
| Reach forecast, skill over persistence | +51.5% | — |
| Reach 90% interval coverage | 90% | 90% |
| Reach 80% interval coverage | 82% | 80% |
| Reach 50% interval coverage | 53% | 50% |
| Panel-only backtest, error and 80% coverage | 0.0667 / 86% | — / 80% |

The reach calibration verdict reads **"calibrated within tolerance"**, so the
round-count intervals derived from it may be read as stated rather than as
indicative. That verdict failed before this branch.

**Scores move about 0.02 of Brier between synthetic seeds.** Measured across
seeds 11, 23 and 37: 0.0933, 0.0883, 0.1081 — an sd of 0.010 and a range of
0.020. Any claimed improvement smaller than that is noise. Interval coverage is
far steadier at 88.4% ± 1.4 and is the better signal to steer a calibration
change by.

### The qualifier that must not be dropped

**No round count has been validated against a field outcome.** It cannot be,
without a location where the counterfactual was run. Every figure above describes
the estimator. None describes Nigeria. That sentence belongs on every briefing,
slide and paper that carries these numbers.

`validation.backtest_on_panel` is the one that runs on real data. It scores
**reach**, not round count, because on a real panel the true round count is never
observed.

---

## 5. What was learned the hard way

Seven estimator biases were found during validation. All are fixed. All are
documented in `docs/METHOD.md` §8. **Do not reintroduce them.**

1. **Stickiness from a pass-or-fail label reads far too low** — 0.38 against a
   true 0.72. Thresholding discards how far below the line a place sits. Use the
   intraclass correlation of the *continuous* reach series.
2. **Reach spread must have verification sampling variance removed.** With an
   LQAS lot of 60, sampling variance is the same order as the real
   between-settlement variance.
3. **Intermittent closure is not a permanent unreachable core.** A settlement
   closed in 6% of rounds is reachable. Counting that as 6% permanently
   unreachable dropped precision on the infeasibility verdict to 47%.
4. **Stickiness is dragged down by movement it does not cause.** It decides
   which children a round misses, never how many. Removing that movement lifts
   the estimate from 0.66 to 0.70 against a true 0.72. The correction only ever
   pushes upward, so its direction is known in advance.
5. **Round-to-round movement measured from one stream is mostly reporting
   noise** — 22% against a true 7%. An inflated shock lets a draw clear the
   target on one good round. Use the covariance of the administrative and
   verified streams, whose errors are independent, at the **median** across
   settlements so that places which close and reopen do not set the figure for
   everywhere else. That reads 8.4%.
6. **The predicted reach spread counts that movement twice.** The model states
   what one round will return. The inversion draws a level once, keeps it for
   every round of a draw, and applies the movement separately. At the median
   settlement **53% of the reach predictive variance is movement**.
7. **The uncertainty on reconstructed immunity must be propagated, not
   asserted.** It was a flat 0.02 for every settlement, and a floor was doing all
   the work. The true error is strongly heteroscedastic. The stated figure was
   2.4× too cautious at the top of the range and far too confident at the bottom.

### Two further invariants

**Never let the spread model score its own training residuals.** Fitting the
scale model on a slice and then dividing that slice's residuals by that model's
own predictions understates them. Every interval built from it runs narrow — 71%
coverage at a nominal 80%. The failure does not show up where it is made:
coverage on the producing slice looks correct. The calibration slice is therefore
split again by round, so the shape is read off rows that taught neither model.

**A verdict and the figure printed beside it must agree.** The infeasibility
verdict fires when at least half the draws put the ceiling below the target, so
the ceiling reported alongside it must be below the target too. Summaries that
interpolate between draws break this exactly at the boundary. Report medians of
draw arrays through `_lower_median` — a draw that happened, not the midpoint of
two that did. This reached CI as a test that passed locally and failed on the
runner, because a unit must land on P = 0.500 exactly for it to show, which two
of twelve seed and draw-count combinations do.

### The protected invariants

`tests/test_science.py` guards modelling claims, not plumbing:

1. Gauss-Jacobi quadrature is exact to about 1e-15.
2. Stickiness never makes a place look easier (Jensen's inequality).
3. The unreachable core survives every round.
4. Without campaigns, immunity settles at routine-immunisation coverage.
5. A higher target never needs fewer rounds.
6. Runs are reproducible. CI enforces this.
7. A verdict and the figure printed beside it agree.

---

## 6. The finding that sets the agenda

**Read this before planning any work.**

The engine states 55% where 40% happens, through the middle of its probability
range. Six hypotheses about the probability calculation were implemented and
measured. Five were rejected. The sixth was real, was fixed, and moved the middle
not at all.

| Hypothesis | Result |
|---|---|
| Forecast error grows with horizon | Flat, ratio 1.00 → 1.03. **Rejected** |
| Within-unit variance separates by mean–variance signature | Negative variance component. **Rejected** |
| Monte Carlo parameter spreads are overstated | Brier identical at three settings. **Rejected** |
| The probability answers feasibility, not timing | Brier worse, 0.1266 vs 0.1055. **Rejected** |
| The interval shape is read off rows the spread model trained on | **Confirmed and fixed** |
| Round-to-round movement is counted twice | **Confirmed and fixed**, middle unmoved |

The cause was then isolated directly. The engine received the **true** current
immunity. Nothing else changed.

| Measure | As built | With true immunity |
|---|---|---|
| Round-count MAE | 1.37 | **0.78** |
| Brier score | 0.109 | **0.091** |
| Gap at stated 0.45 | −0.05 | **−0.01** |
| Gap at stated 0.55 | −0.09 | **+0.00** |
| Gap at stated 0.65 | −0.11 | **+0.02** |

The middle of the range is wrong because **immunity is reconstructed with
error**. It is not wrong because the probability is built incorrectly from that
immunity.

The error is concentrated where the decision is hardest:

| Band of the immunity estimate | Mean estimate | True error SD |
|---|---|---|
| Lowest fifth | 0.669 | **0.1485** |
| Highest fifth | 0.958 | 0.0082 |

Eighteen times larger in the settlements that need the most rounds. Near the
ceiling, repeated rounds have saturated immunity and the reconstruction has
little room to be wrong. Far from the ceiling, the denominator, the unreachable
core and per-dose take all move the answer.

**The binding constraint on this engine is the measurement of current immunity.
It is not the model.**

---

## 7. The open work, ranked

Full detail in `docs/RESEARCH_AGENDA.md`.

### Priority 1 — three items that are one programme of work

**1. Measure current immunity directly.** A serosurvey, or LQAS with finger-mark
or recall history, sited in low-immunity settlements. The open question for a
scientist: how to size and site it so it narrows the error **where the error is**,
not where the population is. Proportional-to-population sampling lands in
well-covered settlements and buys almost nothing.

**2. Obtain a real field panel.** One state, eight rounds minimum, satisfying the
data contract, with `verified_vaccinated`. This replaces the reach figures with
field figures. It does not validate the round count.

**3. Design the round-count validation.** The counterfactual was never run
anywhere, so history cannot validate a round count. The one honest route is a
**pre-registered prospective forecast**: seal predictions with a date and a
commit hash before the rounds run, compare after. Open questions: what outcome
measure is falsifiable when immunity is not directly observed after the rounds
either; how many settlements are needed to detect a calibration gap of 8 to 11
points; how to select settlements without making the test easy.

These three share a design. A serosurvey built for item 1 can serve as the
outcome measure for item 3, and both depend on the panel in item 2. Designing
them together costs far less than designing them in sequence.

### Priority 2 — audit every asserted dispersion

The immunity fix found a defect of a particular shape: **an uncertainty asserted
as a constant rather than propagated from its inputs.** The same audit has been
run on two more parameters. Both fail the same way. **Neither is fixed.**

| Parameter | Stated spread | Measured error SD | Ratio |
|---|---|---|---|
| Unreachable core, π₀ | 0.0158 at the median unit | 0.1053 (0.2274 in the low band) | ~7× (~14×) |
| Denominator inflation | 0.08 | 0.2325, with a −0.0770 bias | ~3× |

π₀ matters most: it sets the ceiling, and the ceiling drives the verdict that
campaigns are the wrong instrument — the most consequential thing this engine
says. The fix pattern is known: propagate the sampling error behind each of the
three routes into the core through `unreachable_core`.

**Triangulate the denominator.** `target_pop` is a reported figure with no second
source. 407 of 960 simulated settlements reported coverage above 100% in at least
one round, and the engine treats those denominators as inflated by at least that
excess — a floor, not an estimate. The scientific question is which independent
source has an error genuinely **uncorrelated** with the microplan error. Two
sources that share a bias triangulate nothing.

### Priority 3 — known modelling gaps

1. **Spatial transmission is not modelled.** Neighbours are independent. A
   settlement can reach the target and be reinfected from next door. Closing this
   changes what "reaching the target" means for a single unit.
2. **Susceptible clustering inside a unit is not modelled.** 95% immunity with the
   remaining susceptibles in one ward is not 95% spread evenly.
3. **The drift floor is an assumption.** Fatigue is held to plateau at 60% of
   current reach. No panel here is long enough to measure where it really does.
4. **Reach beyond about six consecutive rounds is extrapolation.**
5. **Per-dose take is one number for the whole programme.** It does not vary by
   vaccine type, cold chain performance or season.
6. **Local R₀ is available but off by default.** Turning it on is a decision, not
   a data problem.

### Two open questions about the reach model

Neither is a defect. Both need a real panel.

**One feature dominates.** `accessibility_ordinal` carries 83.2% of feature
importance. On a real panel, a single feature at that weight needs checking for
leakage, and for whether the model has become a lookup table on access status.

**The surveillance features carry no weight.** `npafp_rate`, `stool_adequacy` and
`es_positive_90d` all score 0.00%. In the simulated programme they may be noise
by construction, so this says nothing about the field. On a real panel the
question is live: does surveillance quality predict campaign reach? If not, drop
them from the model and keep them for the report.

---

## 8. Guard rails

These exist because the failure modes are attractive and hard to detect later.

**Do not close the mid-range gap by fitting to validation truth.** It is a
measurement defect, not a tuning defect. Six hypotheses about the probability
calculation were tested and none was it. A parameter tuned until the reliability
diagram looks straight produces a model that scores well here and fails in the
field.

**Do not claim a Brier movement smaller than 0.02.** That is the seed-to-seed
range. Steer calibration work by interval coverage instead.

**State miscalibration, do not tune it away.** Both intervals currently run wide.
Both are reported.

**No coverage figure without its denominator basis.** Coverage above 100% is a
denominator defect, never success.

**Record every assumption** in the `ProvenanceLog`.

**The engine produces advice.** It does not authorise a classification or a
campaign calendar. A named human approver does. Every report says so. Keep it
that way.

---

## 9. First moves for a new session

1. **Install and run the validation.** `pip install -e ".[dev]"` then
   `python scripts/run_validation.py`. Confirm the figures in section 4. If they
   differ by more than the seed noise in section 4, something has changed and
   that is the first thing to understand.
2. **Read `docs/METHOD.md` §8 and §12.** Sections 5 and 7 of this document
   summarise them; the method document carries the derivations.
3. **Walk the pipeline once.** `python scripts/pipeline_end_to_end.py` prints an
   annotated ingestion-to-output pass.
4. **Pick a lane.** The π₀ dispersion in section 7 is the largest defect that can
   be closed with code alone, and the fix pattern is already proven on immunity.
   Everything above it in priority needs data or a study design.
5. **Before changing any estimator,** check whether a test in
   `tests/test_science.py` already encodes the property you are about to alter.
   Seven invariants are protected there, and each one exists because it was
   broken once.

### If you are asked to improve the headline scores

Read section 6 first, then section 8. The two most tempting moves — tuning the
mid-band and chasing a Brier improvement of a few thousandths — are both ruled
out by measurement, and the reasons are recorded so that they do not have to be
rediscovered.

---

## 10. Document map

| File | Holds |
|---|---|
| `README.md` | What the engine is, for a first-time reader |
| `CLAUDE.md` | Working guidance for coding agents: architecture, invariants, gotchas |
| `docs/METHOD.md` | The derivations, the parameters, the known limits |
| `docs/RESEARCH_AGENDA.md` | The open work in full, with the data requests |
| `docs/HANDOVER.md` | This document |
| `tests/test_science.py` | The modelling claims, as executable assertions |
| `scripts/run_validation.py` | The full validation report |
| `scripts/pipeline_end_to_end.py` | An annotated walk from ingestion to output |
