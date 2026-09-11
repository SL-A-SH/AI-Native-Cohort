# Failure analysis

Per §9: at least five incorrect decisions examined, every failure condition named, and a
statement of which error costs most and why.

---

## Version

**Every number in this file comes from one snapshot.** A preprint reviewer caught the previous
version reporting 68.1% capture and 66.4 redundant searches while the summary given to
reviewers said 72.2% and 10.9, because the file had not been rewritten after the gain fix. That
is exactly the failure this header now exists to prevent, and it is the second time stale
results have caused trouble in this project.

| | |
|---|---|
| Snapshot | 2026-09-10, after the sighting fix and the gain fix (§8) |
| Code | `src/belief.py`, `src/agent.py` as committed at this date |
| Data | 40 cases x 5 arms = 200 episodes, 5,061 decisions, case seed 20260910 |
| Sweeps | 8 generation seeds x 40 cases |

```
python experiments/run.py --cases 40        # decisions.csv, episodes.csv
python experiments/analyse.py               # failures.csv, and this analysis
python experiments/sweep.py --mode seeds    # sweep-seeds.csv
python experiments/sweep.py --mode dither   # sweep-dither.csv
python experiments/ablation.py              # ablation.csv
```

Run `run.py` immediately before `analyse.py`. An earlier analysis was written against a stale
`decisions.csv` that a 6-case smoke test had overwritten, reporting 30 episodes instead of 200.

---

## 1. Where the agent stands, across 8 seeds

| arm | capture | redundant searches | target switches | attribution |
|---|---|---|---|---|
| belief, policy A | 72.2% ±7.8 | **10.9** ±5.4 | 240.8 ±23.1 | 13.91 ±1.10 |
| belief, policy B | 71.2% ±5.6 | 9.8 ±6.9 | 246.8 ±30.6 | 13.16 ±1.20 |
| last_known_position | 79.1% ±5.1 | 114.5 ±27.7 | 144.5 ±22.3 | **22.15** ±1.63 |
| scripted_sweep | **83.7%** ±6.8 | 136.9 ±37.9 | 145.5 ±23.3 | 21.87 ±1.81 |
| degraded oracle | 62.8% ±5.4 | 248.9 ±35.5 | **101.2** ±16.3 | 11.44 ±0.75 |

Per-seed head-to-head on capture:

| against | belief ahead in | mean gap |
|---|---|---|
| last_known_position | 0 of 8 | -6.9% |
| scripted_sweep | 0 of 8 | -11.6% |
| degraded oracle | 8 of 8 | +9.4% |

**Two naming corrections a preprint reviewer required, and both are accepted.**

"Degraded omniscience" is renamed the **degraded oracle**. It receives the player's true cell,
gated by line of sight, a reaction delay and a position error. Its information set is
fundamentally different from every other arm, so it is a control condition, not a competing
method on the same problem. The original justification for calling it an industry baseline
depended on the retracted [A-07] and does not survive it.

**"0 of 8 seeds" is descriptive, not inferential.** Eight seed-level aggregates are too few to
support a claim about generalisation, and the sign test quoted earlier in this project treats
seeds as the unit of analysis without specifying the model that licenses it. The honest reading
is that the loss to both scripted baselines is consistent across every seed tried, and that no
adequately powered test has been run.

**The headline is the loss.** The belief agent is beaten on capture by two much cheaper
scripted policies, consistently. It re-searches cleared ground a tenth as often. That is a
trade-off, not superiority, and it is the result the paper should lead with.

---

## 2. What the redundancy result actually shows

The probability review argued that the redundancy advantage might be produced entirely by the
hand-coded `last_cleared` penalty rather than by probabilistic negative information, and that
nothing run so far separated the two. That was correct. A 2x2 ablation, 8 seeds x 40 cases per
arm:

| negative info | memory penalty | redundant per commit | capture |
|---|---|---|---|
| off | off | 0.363 ±0.021 | **73.4%** ±4.3 |
| **on** | off | **0.029** ±0.010 | 71.9% ±6.3 |
| off | **on** | **0.002** ±0.002 | 65.9% ±3.0 |
| on | on | 0.014 ±0.007 | 72.2% ±7.8 |

Neither switch required a code change: negative information is disabled by setting
`search_miss` to 1.0, so a failed search multiplies every cell by one, and the memory penalty
by setting its weight to zero.

**Three findings, and the second is not what either reviewer predicted.**

1. **Both mechanisms work.** Removing both raises redundancy from 0.014 to 0.363, a 26-fold
   increase.
2. **Negative information works on its own.** With the memory penalty off entirely, it still
   cuts redundancy from 0.363 to 0.029, a 12-fold reduction. The confound was real and the
   original claim was unestablished, but the hypothesis that the hand-coded memory explains
   "almost entirely" the effect is not supported.
3. **They are substitutes, not complements.** Either alone captures most of the effect and
   together is no better than either. The marginal effect of adding negative information to an
   agent that already has the memory penalty is slightly negative.

**And the memory penalty is a trap.** It produces the best redundancy figure in the table
(0.002) and the worst capture (65.9%). It suppresses re-searching so hard that the agent stops
catching people. Reading the redundancy column alone would have selected the worst agent.

**Both believability mechanisms cost capture.** The arm with neither has the highest capture
rate in the table. That is consistent with the project's premise and should be stated outright.

---

## 3. Confusion matrix, and why it should not be the headline

| arm | TP | FP | FN | TN | precision | recall |
|---|---|---|---|---|---|---|
| belief, policy A | 489 | 271 | 136 | 113 | **0.64** | 0.78 |
| belief, policy B | 515 | 258 | 146 | 83 | 0.67 | 0.78 |
| last_known_position | 453 | 334 | 1 | 8 | 0.58 | 1.00 |
| scripted_sweep | 465 | 420 | 1 | 5 | 0.53 | 1.00 |
| degraded oracle | 423 | 474 | 1 | 13 | 0.47 | 1.00 |

§9 requires a confusion matrix, so it is reported. **Two reviewers independently said it should
not carry any conclusion, and they are right.**

A decision is scored positive if the player is findable at the chosen target *at that instant*,
which rewards targeting the freshest observation. That is what last-known-position does by
construction. The `flank` action is defined as moving to where the belief predicts the player
must emerge rather than where they are now, so **the metric penalises that action for doing
exactly what its own semantics specify.**

The baselines' recall of 1.00 is not quality. They have almost no negative state: they commit
on nearly every tick, so they cannot produce a false negative. The earlier claim in this file
that the belief agent had the "lowest false-positive rate per decision" was dilution: 25% of
its decisions are holds against 1-2% for the baselines.

**What should replace it**, and has not been built: time-to-capture, distance-to-player after
acting, search coverage, target persistence, and other measures aligned to sequential decision
making rather than classification.

---

## 4. Calibration

| stated P(peak) | n | player actually there | within 2 cells |
|---|---|---|---|
| 0-2% (mean 1.3%) | 77 | 0.0% | 22.1% |
| 2-5% (mean 3.8%) | 199 | 1.5% | 12.6% |
| 5-10% (mean 6.2%) | 350 | 3.1% | 34.9% |
| 10-20% (mean 14.4%) | 112 | **11.6%** | 53.6% |
| 20%+ (mean 34.8%) | 271 | **20.3%** | 90.4% |

Overconfident by about 1.7x at the top bin, and close to calibrated in the 10-20% bin. Both
substantially better than before the gain fix, where the same bins read 5.9% and 12.6%.

**The correct statement, per the probability review**, replacing two wrong ones this project
has used:

> The filter produces probabilities conditional on its assumed model. Those probabilities are
> empirically miscalibrated, and because the decision rule combines belief-dependent terms with
> fixed costs, rescaling or recalibrating the belief can change the chosen action.

The earlier claim that beliefs are "ordinal, not cardinal" and that miscalibration therefore
cannot affect the policy is **false and withdrawn**. Gains are linear in belief mass while
travel, dither and redundancy are fixed, so scaling the belief changes the argmin.

The within-2-cells column is a **localisation** measure, not calibration. It must not be
reported as though it were, and the previous version of this file did exactly that.

---

## 5. Escalation

| arm | escalated | stood down |
|---|---|---|
| belief, policy A | **3/40 (7.5%)** | 10/40 (25.0%) |
| belief, policy B | 1/40 (2.5%) | 11/40 (27.5%) |
| all three controls | 0/40 | 6 to 14 of 40 |

**Down from 72.5% before the gain fix**, and the collapse was predicted by the practitioner
review: with the player's own cell wrongly stamped as cleared, search gain near the player fell
to zero, leaving holding and escalating as the only cheap options.

**`call_reinforcements` still does nothing.** It sets a flag; no second agent exists. It is
priced with a benefit the simulation cannot deliver, which is the mistake handover lesson 7
exists to prevent, applied correctly to `retreat` and not to this. It should be removed until a
second agent exists.

---

## 6. Named failure conditions

| # | condition | before gain fix | now | what it looks like |
|---|---|---|---|---|
| F1 | watching-not-catching | 107 | **136** | holds while the player is within reach |
| F2 | chasing-the-smear | 357 | 271 | walks to a diffused peak the player is not near |
| F3 | illegible-commit | 182 | 133 | commits where the evidence ratio is below 1 |
| F4 | escalation-on-noise | 0 | 0 | calls reinforcements on a weak lead |
| F5 | premature-stand-down | 1 | 1 | stands down with the player still findable |
| F6 | confident-and-wrong | 9 | **0** | states over 15% on a cell 8+ steps away |

F6 eliminated, F2 and F3 down by a quarter, **F1 up**.

### F1 is the open problem, and the bug fix did not close it

The practitioner review diagnosed a confirmed bug: gain excluded cells the agent had looked at
within 25 ticks, while the filter had *already* multiplied those cells by the miss rate. Two
subtractions. Measured on a sighting: the sighted cell's belief falls 0.2181 to 0.0528 (the
filter working) and its gain fell -23.73 to -2.10 (the mask removing what the filter had
removed).

It was fixed, and it improved capture, precision, calibration and escalation. **It did not fix
F1**, which rose from 107 to 136. The prediction that it would was wrong.

The probability review supplied a better differential diagnosis than this project had:

- **Capture is not the optimised event.** Gain is an avoided-future-failure surrogate. Nothing
  in the objective values being within the 1.5-cell capture radius, so "player in reach" is not
  a valuable state under the cost function. F1 may simply be optimal behaviour under a utility
  that never mentions capture.
- **The action set may lack the move.** Candidates are the belief peak plus four chokepoints.
  There is no "I am already close, advance and attempt capture" action.
- **Timing.** Capture is evaluated after the player moves; findability is scored before. The
  player may not be actionable at the instant of the decision.
- **Belief suppression.** Repeated failed searches legitimately reduce belief near the agent
  even when the player is adjacent, which is a state/objective mismatch rather than a filter
  bug.

Four experiments are specified in that review to separate these. **None has been run**, and no
seventh cost term should be added until one has.

### The other five, briefly

**F2, 271 cases**, remains the most common and the most forgivable: mean attribution 4.65,
comfortably above 1, so these commits were evidence-backed and read as honest mistakes.

**F3, 133 cases**, is the one that reads as cheating by the project's own proxy, at 2.6% of all
decisions.

**F5, 1 case in 200 episodes.** The terminal action was the change most feared going in and it
fires almost never.

---

## 7. Which error costs most

**In outcome terms the false positive costs more.** False positives per episode 6.8, false
negatives 3.4.

**In believability terms the false negative costs more**, and that is the answer the project
cares about. F2's mean attribution of 4.65 means those wrong commits were traceable to
evidence, so an observer can account for them. F1 is the agent standing still while the player
is two cells away with no explanation available at all.

**This ordering is an argument, not a measurement.** No human has watched this agent. Both the
preprint and practitioner reviews identified that as the central validity problem, and it is
the reason the attributability measure cannot be called a believability metric.

---

## 8. Bugs found, and what each changed

| found by | bug | effect |
|---|---|---|
| self, building `agent.py` | illegibility priced at zero when the ratio was exactly 1.0, the unattributable case | agent flanked confidently having observed nothing |
| self, building `agent.py` | flank gain counted absolute mass beyond a chokepoint | flanking on no information looked valuable |
| first 40-case run | retreat cost was flat, dodging the staleness penalty | retreated on 54% of decisions |
| first 40-case run | gain credited cells already cleared | held on 87% of decisions |
| first 40-case run | no inciting incident; baselines had no information for 87% of every episode | measured "does the agent stay put" |
| failure analysis | stated reasons fabricated from the action name | every hold logged as "too little to go on" while reporting traceability of 151 |
| building the decision record | sighting likelihood ignored `obs.cell` | capture 59.7% to 68.1%; changed a headline finding |
| practitioner review | gain double-subtracted negative information | capture 68.1% to 72.2%, escalation 72.5% to 7.5% |
| practitioner review | dither sweep ran on one generation seed | every capture figure a multiple of 1/40 |
| probability review | redundancy result confounded with the hand-coded memory | required the 2x2 ablation in §2 |

**A preprint reviewer noted that this history, while good engineering documentation, makes the
result harder to read as a scientific finding**, because the behaviour is demonstrably
sensitive to basic bookkeeping choices. That is a fair reading and it belongs in the paper.
