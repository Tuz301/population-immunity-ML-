# Method

This document describes how the engine calculates the number of campaign rounds.
Read it before you use a number that the engine produces.

## 1. The question

A programme asks: *how many campaign rounds does this location need to reach
population immunity?*

The engine does not answer this with a regression. A regression on past round
counts learns the schedule that the programme ran. It does not learn the
requirement.

The engine answers it by inversion. It builds a model of the susceptible
children in a location. It runs the model forward. It records the round at which
immunity first crosses the threshold. It repeats this for thousands of parameter
draws. The result is a distribution of round counts, not a single number.

## 2. The immunity target

The target is the critical vaccination coverage, adjusted for vaccine failure:

```
Vc(adj) = (1 - 1/R0) / (1 - epsilon)
```

At `R0 = 6.0` and `epsilon = 0.126`, `Vc(adj) = 95.3%`.

The engine can also use a local reproduction number for each location. This is
off by default. A single external target prevents a programme from benchmarking
each round against the last one.

### Reconciliation of the two vaccine parameters

The programme states a schedule failure `epsilon = 0.126`. The engine also needs
a per-dose take `tau`. The two must describe the same vaccine:

```
1 - (1 - tau)^d = 1 - epsilon
```

With `d = 3` doses, `epsilon = 0.126` implies `tau = 0.4987`. The engine uses
`tau = 0.500`. The residual is 0.1 percentage points of schedule efficacy.

`EngineConfig.reconcile_take()` reports this residual. Change one parameter and
the residual changes. The report shows it.

## 3. Reach as a distribution, not a rate

A campaign round does not reach a fixed share of children at random. It reaches
the same children each time. This is the reason that the tenth round does much
less than the first.

The engine gives each child a per-round reach probability `r`:

- With probability `pi_zero`, the child is in the unreachable core. Then `r = 0`
  in every round. Security-closed settlements, permanent refusals and
  never-enumerated mobile populations are in this group.
- Otherwise `r` has a Beta distribution with mean `mu` and concentration
  `kappa`. A low `kappa` means unequal reach between children.

Between rounds, a share `rho` of children keep their reach probability. A share
`1 - rho` draw a new one. `rho = 1` is full stickiness. `rho = 0` is the naive
independent model.

The susceptible share after `N` identical rounds is:

```
s(N) = pi_zero + (1 - pi_zero) * [ rho * E[(1 - tau*r)^N] + (1 - rho) * (1 - tau*mu)^N ]
```

`(1 - tau*r)^N` is a polynomial of degree `N` in `r`. Gauss-Jacobi quadrature
integrates it against the Beta density exactly. The engine does not approximate
this integral.

The `pi_zero` term does not decay. It sets a ceiling that no round count passes.

## 4. The susceptible stock

Children are a stock. Doses are a flow. The engine tracks the stock.

Between rounds:

```
dN/dt = B - N/A
dS/dt = B*(1 - v) - S/A
```

`B` is births per month. `A` is the width of the target age band in months. `v`
is the share of newborns that routine immunisation protects. Both equations have
a closed-form solution, so the engine does not integrate numerically.

At a campaign round, the susceptible mass at reach node `r` is multiplied by
`(1 - tau*r)`.

Two consequences follow.

1. Without campaigns, population immunity settles at `v`. This is the level a
   location returns to when campaigns stop. The engine uses it as the baseline
   for the history replay in section 6.
2. With campaigns at a fixed interval, immunity settles into a sawtooth. The peak
   is immunity just after a round. The trough is immunity just before the next
   one. The trough decides whether transmission restarts between rounds.

## 5. Two ceilings, and why both are reported

**Core-limited ceiling.** The highest immunity a closed cohort could reach. It is
limited only by the unreachable core.

**Schedule ceiling.** The highest immunity this round interval ever reaches, once
rounds and births balance. Births refill the stock between rounds, so this
ceiling is lower.

The engine reports both. The gap between them separates two different problems:

- The core-limited ceiling is below the target. Children cannot be reached. Send
  an access, negotiation or enumeration action.
- The core-limited ceiling clears the target but the schedule ceiling does not.
  Rounds are too far apart, or routine immunisation is too weak. Shorten the
  interval, or strengthen routine immunisation.

Neither problem is solved by adding rounds at the current interval.

## 6. Current immunity

Immunity at settlement grain is not measured. The engine reconstructs it.

1. Start each location at the immunity that routine immunisation alone sustains.
2. Apply each observed round, with the reach that round achieved.
3. Let births refill the stock between rounds, at the observed round spacing.

The starting point is not a free parameter. It is the no-campaign equilibrium
from section 4.

Uncertainty comes from replaying three times, at the measured reach and at plus
and minus one sampling standard error of the verification lot.

On the simulated programme, the mean absolute error of this reconstruction is
about 0.03.

## 7. The reach model

The engine predicts one quantity with machine learning: the reach of the next
round in each location.

**Target.** Verified reach, from independent verification. Where no verification
exists, administrative coverage multiplied by the programme-wide
verified-to-administrative ratio. The engine records this substitution.

**Method.** LightGBM with monotone constraints, plus a conformal residual model.

LightGBM does not accept monotone constraints under a quantile objective. The
engine therefore separates the centre from the spread:

1. A constrained squared-error model gives the centre.
2. A second model gives the conditional spread.
3. The standardised residual distribution comes from a calibration slice that the
   centre model never saw.
4. Quantiles are rebuilt from the three.

The calibration slice is the same depth as the scoring holdout. A residual
measured one round ahead is smaller than one measured three rounds ahead.
Calibrating at one horizon and deploying at another gives intervals that are too
narrow.

**Monotone constraints.** Reach cannot increase because insecurity increased, or
because the location was missed more often. These signs are known. The engine
enforces them. It costs some fit.

**Validation.** Forward in time. Random k-fold scores a model on rounds it has
already seen in neighbouring settlements.

## 8. Parameters estimated from the panel

| Parameter | Meaning | Estimator |
|---|---|---|
| `pi_zero` | Unreachable core | Persistent closure, permanent refusal and unenumerated mobile population, combined through their complements |
| `kappa` | Spread of reach between children | Method of moments on between-settlement spread, minus verification sampling variance |
| `rho` | Round-to-round stickiness | Intraclass correlation of reach, with round effects removed and sampling variance subtracted |
| `drift` | Change in reach per round | Within-unit slope of reach on round index |
| `inflation` | Denominator inflation | Peak administrative coverage above 100% |

Three of these estimators required correction during development. Each
correction is recorded here because each changed the answer materially.

**Stickiness from a pass-or-fail label is biased low.** A threshold discards how
far below the line a location sits. Two locations can both fail every round while
one is at 79% and the other at 30%. Measured on the pass-or-fail label, `rho`
came out at 0.38 against a true value of 0.72. Measured on the continuous reach
series, it came out at 0.66.

**Spread must have measurement noise removed.** Verification uses a small lot.
With a lot of 60, the sampling variance is of the same order as the real
between-settlement variance. Without the correction, `kappa` came out at 3.0
against a true value of 8.0. The engine then reported far more inequality between
children than exists, and gave up on hard-to-reach children too early.

**Intermittent closure is not a permanent core.** A location closed in 6% of
rounds is reachable. It was not reached every time. Counting that 6% as
permanently unreachable capped its immunity below the target and produced a false
"campaigns cannot do this" verdict. Precision on that verdict fell to 47%. Only
locations closed in every round, or all but one, now count toward `pi_zero`.
Precision returned to 83%.

## 9. Fatigue against structure

Reach falls round on round in most panels. The engine measures the fall and
projects it.

A shortfall caused by falling reach is not the same as a shortfall caused by
children who cannot be reached. The first has an owner and a remedy. The second
does not.

The engine therefore runs two simulations:

- **Planning run.** Projects the measured decline. This gives the round count.
- **Structural run.** Holds reach at today's level. This gives the feasibility
  verdict.

Only the structural run can return the verdict that campaigns are the wrong
instrument. When the structural run clears the target and the planning run does
not, the engine sets `fatigue_risk` instead. The message is then: the target is
reachable, but only if reach stops falling.

## 10. What the engine outputs

For each location:

- `rounds_median` and `rounds_assured`, the round count at the planning
  confidence level.
- `probability_by_round`, the chance that each round count is enough. This is the
  full answer before it is summarised.
- `feasibility`, one of feasible, feasible beyond budget, infeasible by
  campaigns, insufficient data.
- `immunity_ceiling_median` and `core_ceiling_median`.
- `holding_interval_months`, the longest interval that holds the target.
- `binding_constraint`, the input whose uncertainty drives the round count most.

The last item directs the next measurement. It is the answer to "what should we
spend the next data pound on".

## 11. Validation

Three claims are tested separately. Do not read a good result on one as a good
result on the others.

**Discrimination.** Does the engine rank locations correctly?

**Calibration.** When the engine says four rounds will do it with 80%
confidence, does that happen 80% of the time?

**Recovery under misspecification.** Does the answer degrade gracefully when the
world does not match the model?

The simulated programme in `adapters/synthetic.py` supports the third claim. It
is not the estimator with noise added. It differs from the estimator's
assumptions in five ways: reach drifts downward; some locations over-report;
denominators are inflated by a factor the engine never sees; accessibility flips
between rounds; verified counts carry finite-sample noise.

Results on that programme are in the README.

`backtest_on_panel` supports the first two claims on a real panel. It scores
reach, not the round count. On a real panel the true round count is never
observed, because the counterfactual was not run.

## 12. Known limits

1. **Round counts have not been validated against field outcomes.** They cannot
   be, without a location where the counterfactual was run. Every accuracy figure
   in this repository comes from a simulated programme.
2. **Both intervals still run wide, in the same direction.** On the simulated
   programme, the 80% round-count interval covers the truth about 88% of the
   time, and the reach model's 80% interval covers about 82% of held-out
   observations. The engine under-claims what it knows in both. Both are
   reported. Neither is tuned against the validation truth, because that would be
   fitting to the test. Round-count coverage is steady across seeds at 88.4% plus
   or minus 1.4, so it is the figure to steer a calibration change by; the Brier
   score moves about 0.02 between seeds and cannot carry a smaller claim.
3. **The engine does not model spatial transmission.** Two locations that share a
   border are treated as independent. A location can reach the target and still
   be reinfected from next door.
4. **Susceptible clustering is not modelled inside a location.** Population
   immunity of 95% with the remaining susceptibles in one ward is not the same as
   95% spread evenly. The council raises this when stickiness is high.
5. **The drift floor is an assumption.** Fatigue is held to plateau at 60% of
   current reach. No panel here is long enough to measure where it really
   plateaus.
6. **Reach beyond about six consecutive rounds is extrapolation.** No location in
   the training data has run more.
