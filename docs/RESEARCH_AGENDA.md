# What is not done yet

A handover note for scientific review. It lists the work the engine still needs,
in the order that the evidence supports.

Status as of commit `ddea081`, branch `claude/campaign-optimization-immunity-yne6vw`.

---

## 1. Where the engine stands

The engine answers one question per settlement: how many campaign rounds reach
the immunity target, with what confidence, and when no number of rounds will do
it. It inverts a susceptible-stock model. It does not regress on past round
counts, because that learns the schedule a programme ran and not the requirement
it faced.

Two things are true at once, and both must travel with any result.

The estimator recovers known truth under misspecification. On a simulated
programme of 960 settlements, the round count has a mean absolute error of 1.37
rounds. The unreachable-settlement flag has a recall of 95%. All three reported
reach intervals cover within tolerance.

**No round count has been validated against a field outcome.** Every accuracy
figure in this repository describes the estimator. None describes Nigeria. That
qualifier cannot be dropped from a briefing, a paper or a slide.

---

## 2. The one finding that sets the agenda

The engine states 55% where 40% happens, through the middle of its probability
range. Six hypotheses about the probability calculation were implemented and
measured. Five were rejected. The sixth was real, was fixed, and moved the middle
not at all.

The cause was then isolated by a direct test. The engine received the true
current immunity for each settlement. Nothing else changed.

| Measure | As built | With true immunity |
|---|---|---|
| Round-count MAE | 1.37 | **0.78** |
| Brier score | 0.109 | **0.091** |
| Gap at stated 0.45 | −0.05 | **−0.01** |
| Gap at stated 0.55 | −0.09 | **+0.00** |
| Gap at stated 0.65 | −0.11 | **+0.02** |

The middle of the range is wrong because immunity is reconstructed with error.
It is not wrong because the probability is built incorrectly from that immunity.

**The binding constraint on this engine is the measurement of current immunity.**
It is not the model. Every item below is ranked against that finding.

---

## 3. Priority 1 — Measure current immunity directly

Immunity is not measured at settlement grain anywhere in the data. The engine
rebuilds it by replaying the observed rounds through the stock model. The error
in that reconstruction is severe, and it is worst where the decision is hardest.

| Band of the estimate | Mean estimate | True error SD |
|---|---|---|
| Lowest fifth | 0.669 | **0.1485** |
| Highest fifth | 0.958 | 0.0082 |

The error is eighteen times larger in the settlements that need the most rounds.
Near the ceiling, repeated rounds have saturated immunity and the reconstruction
has little room to be wrong. Far from the ceiling, the denominator, the
unreachable core and per-dose take all move the answer.

### What is needed

A direct measurement of immunity that anchors the starting stock in the
low-immunity settlements. Two candidates, in order of strength.

1. **Serosurvey.** A seroprevalence sample in a purposive set of low-immunity
   LGAs. This measures immunity itself rather than a proxy for it.
2. **LQAS with finger-mark or recall history.** Weaker, cheaper, and already part
   of programme practice in many states.

### The question for a scientific reviewer

How large must the sample be, and how must sites be selected, so that the
measurement narrows the reconstruction error in the bottom band rather than
confirming the top band where the engine is already accurate? A survey that
samples proportional to population will land mostly in well-covered settlements
and will buy very little. The value is concentrated where the error is.

---

## 4. Priority 1 — A real field panel

Every figure in this repository comes from `adapters/synthetic.py`. That
generator is deliberately not the estimator with noise added. Reach drifts
downward, some locations over-report, denominators carry an unseen inflation
factor, accessibility flips between rounds, and verified counts carry
finite-sample noise. Recovery under that misspecification is the claim.

It is still simulated.

### What is needed

One state. Eight rounds minimum. The panel must satisfy the data contract in
`src/immunity_engine/contracts.py`.

**Required columns.** `unit_id`, `grain`, `lga_code`, `state_code`, `round_code`,
`round_index`, `round_start`, `campaign_type`, `target_pop`,
`denominator_basis`, `admin_vaccinated`.

**The optional column that matters most.** `verified_vaccinated`. Without it,
reach rests on administrative counts, which the programme measures as 18
percentage points optimistic nationally. More important, the engine separates
campaign movement from reporting noise using the covariance of the
administrative and verified streams. With one stream only, that estimate reads
22% against a true 7%, and feeding it to the simulator makes the engine
measurably worse.

**Also valuable.** `verified_sample_n`, `accessibility_status`, `refusal_count`,
`births_per_month`, `ri_coverage`, `nomadic_share`, `latitude`, `longitude`.

With that panel, `validation.backtest_on_panel` scores the reach model on rounds
it has never seen, using no simulated truth. That replaces the reach figures with
field figures. It does not validate the round count. See section 8.

---

## 5. Priority 2 — Audit every asserted dispersion

The immunity fix in commit `9eb98ff` found a defect of a particular shape. The
uncertainty on a quantity was **asserted as a constant** rather than derived from
what is known about its inputs. The stated figure was a flat 0.020 for every
settlement. The true error was strongly heteroscedastic.

The same audit has now been run on two more parameters. Both fail the same way.
**Neither is fixed.**

### The unreachable core, `pi_zero`

| Quantity | Value |
|---|---|
| Stated spread in the Monte Carlo | 0.30 × π₀ + 0.005, which is **0.0158** at the median unit |
| Measured error SD, all units | **0.1053** |
| Measured error SD, low-immunity band | **0.2274** |

The stated uncertainty is about seven times too small across the programme, and
about fourteen times too small in the band that matters. The core sets the
ceiling. A wrong ceiling drives the verdict that campaigns are the wrong
instrument, which is the most consequential thing this engine says.

### The denominator inflation factor

| Quantity | Value |
|---|---|
| Stated spread, `EngineConfig.denominator_inflation_sd` | **0.08** |
| Measured error SD | **0.2325** |
| Measured error bias | **−0.0770** |

The spread is about three times too small. The estimate also carries a bias: the
engine under-states how inflated a reported denominator is. The direction of the
consequence for the round count has not been traced and should be.

### What is needed

A principled estimator for each, in the pattern used for immunity: propagate the
uncertainty from the inputs the quantity is built from, rather than asserting a
figure. For `pi_zero` that means the sampling error behind each of the three
routes into the core — inaccessibility, refusal and mobility — carried through
`unreachable_core`. For the denominator it means a triangulation, which is the
next item.

---

## 6. Priority 2 — Triangulate the denominator

`target_pop` is a reported figure with a provenance stamp. The engine treats the
stamp seriously and carries an inflation factor with uncertainty. It has no
second source to check the figure against.

407 of 960 settlements in the simulated programme reported administrative
coverage above 100% in at least one round. The engine treats those denominators
as inflated by at least that excess. That is a floor, not an estimate.

### What is needed

An independent estimate of the child cohort per settlement, from a source that
does not share an error with the microplan. Candidates: satellite building
footprints with an occupancy model, the Master List of Settlements with its
version history, birth-cohort projection from census with a growth model, or a
household enumeration in a sample of settlements.

The scientific question is which of these has an error that is **independent** of
the microplan error. Two sources that share a bias do not triangulate anything.

---

## 7. Priority 3 — Modelling gaps that are known and unclosed

These are recorded in `docs/METHOD.md` §12. They are listed here with what each
would take.

1. **Spatial transmission is not modelled.** Two settlements that share a border
   are independent in this engine. A settlement can reach the target and be
   reinfected from next door. Closing this needs a contact or movement kernel and
   `latitude`/`longitude` on the panel. It changes what "reaching the target"
   means for a single unit, so it is a change to the question, not only to the
   arithmetic.
2. **Susceptible clustering inside a unit is not modelled.** Population immunity
   of 95% with the remaining susceptibles in one ward is not the same as 95%
   spread evenly. The council raises this when stickiness is high, but the engine
   does not quantify it.
3. **The drift floor is an assumption.** Fatigue is held to plateau at 60% of
   current reach. No panel here runs long enough to measure where it really
   plateaus. The figure is a guard against manufacturing infeasibility, not a
   measurement.
4. **Reach beyond about six consecutive rounds is extrapolation.** No location in
   the training data has run more.
5. **Per-dose take is one number for the whole programme.** It is derived, not
   assumed: 1 − (1 − take)³ = 1 − schedule failure. It does not vary by vaccine
   type, cold chain performance, or season.
6. **Local R₀ is available but off by default.** `use_local_r0` is False, because
   the programme's MAP makes Vc(adj) the single external referent. Where local R₀
   is estimated, a unit-specific target is defensible and would change which
   settlements read as infeasible.

---

## 8. Priority 1 — A validation design for the round count

This is the hardest item and it is not a data request. It is a study design.

The round count cannot be validated against history, because the counterfactual
was never run. No settlement has a record of what would have happened under a
different number of rounds.

### The one honest path

A **prospective, pre-registered forecast**. Lock the engine's predictions for a
defined set of settlements before the next rounds run. Publish the predictions,
sealed, with the date and the commit hash. Compare after the rounds complete.

This tests calibration, which is the claim the engine actually makes. It does not
need a counterfactual. It needs discipline about the sealing.

### The questions for a scientific reviewer

- What is the outcome measure, given that immunity is not directly observed after
  the rounds either? A post-campaign serosurvey in the forecast settlements is the
  clean answer and the expensive one.
- How many settlements are needed to detect a calibration gap of the size the
  engine currently shows, which is 8 to 11 points in the middle band?
- How are settlements selected without making the test easy? Sampling only
  high-immunity settlements would produce a flattering result that means nothing.

---

## 9. Two guard rails

These exist because the failure mode is attractive and would be hard to detect
later.

**Do not close the mid-range gap by fitting to validation truth.** The gap is a
measurement defect, not a tuning defect. Six hypotheses about the probability
calculation were tested and none of them was it. A parameter tuned until the
reliability diagram looks straight would produce a model that scores well on this
simulated programme and fails in the field. This instruction is recorded in
`CLAUDE.md`.

**Scores move about 0.02 of Brier between synthetic seeds.** Measured across
seeds 11, 23 and 37: 0.0933, 0.0883, 0.1081. The standard deviation is 0.010 and
the range is 0.020. Any claimed improvement smaller than that is noise. Interval
coverage is far steadier, at 88.4% ± 1.4, and is the better signal to steer a
calibration change by. Both figures are recorded in `CLAUDE.md`.

---

## 10. Two open questions about the reach model

Neither is a defect. Both need a real panel to answer.

**One feature dominates.** `accessibility_ordinal` carries 83.2% of the reach
model's feature importance. On the simulated programme that is by construction.
On a real panel, a single feature at that weight needs checking for leakage, and
for whether the model has become a lookup table on access status.

**The surveillance features carry no weight.** `npafp_rate`, `stool_adequacy` and
`es_positive_90d` all score 0.00% importance. In the simulated programme they may
simply be noise by construction, so this result says nothing about the field. On
a real panel the question is live: does surveillance quality predict campaign
reach? If it does not, the columns should be dropped from the model and kept only
for the report.

---

## 11. Documentation debt

`docs/METHOD.md` §12 item 2 was stale and is now corrected in the same commit as
this note. It stated that the round-count 80% interval covers about 93% and the
reach 80% interval covers about 71%. Both figures were superseded on this branch.
The round-count interval now covers 88% and the reach interval covers 82%. The
two no longer miss in opposite directions; both run wide.

No other documentation debt is known. `README.md` and `docs/METHOD.md` should be
re-read against the figures in section 2 before either is circulated outside the
team.

---

## 12. Summary in one table

| # | Item | Blocks deployment | Needs |
|---|---|---|---|
| 1 | Measure current immunity | **Yes** | Serosurvey or LQAS, weighted to low-immunity settlements |
| 2 | Real field panel | **Yes** | One state, eight rounds, with `verified_vaccinated` |
| 3 | Round-count validation design | **Yes** | A pre-registered prospective forecast |
| 4 | `pi_zero` uncertainty, ~7× understated | No | A propagated estimator |
| 5 | Denominator uncertainty, ~3× understated and biased | No | An independent second source |
| 6 | Spatial transmission | No | A movement kernel and coordinates |
| 7 | Susceptible clustering inside a unit | No | Sub-unit data or a clustering model |
| 8 | Drift floor of 0.60 | No | A longer panel |
| 9 | Local R₀ per LGA | No | A decision, not data |
| 10 | ~~Fix stale METHOD.md §12~~ | No | Done in this commit |

Items 1 to 3 are one programme of work, not three. A serosurvey designed for item
1 can serve as the outcome measure for item 3, and both depend on the panel in
item 2. Designing them together costs far less than designing them in sequence.

---

## What to bring to a scientific review

The three questions that most need an expert answer, ahead of any further code:

1. How should a serosurvey be sized and sited so that it narrows the immunity
   error where the error actually is, rather than where the population is?
2. What outcome measure makes a pre-registered round-count forecast falsifiable,
   given that immunity is not directly observed after the rounds either?
3. Which independent source of a child denominator has an error that is genuinely
   uncorrelated with the microplan error?
