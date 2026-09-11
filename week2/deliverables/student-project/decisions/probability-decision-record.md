# Probability decision record

Per §10. One decision recorded in full, then one new piece of evidence walked all the way
through to a new action.

Reproduce every number below with:

```
python experiments/decision_case.py
```

The script prints this record. Nothing here is typed in by hand.

---

## Audit data

| | |
|---|---|
| Case | 10 of 40, from `experiments/run.py --cases 40` |
| Tick | 29 |
| Recorded | 2026-09-10 |
| Repository version | `bdb05ae` plus the sighting-model fix described in `results/failure-analysis.md` §7 |
| Data version | `results/decisions.csv`, 5,061 decisions, 200 episodes, case-generation seed 20260910 |
| Model version | `src/belief.py` `PARAMS`: stay 0.55, sighting_hit 0.95, **sighting_sigma 1.0**, sighting_floor 0.01, sound_sigma 2.2, sound_floor 0.04, search_miss 0.15. **Every value `[ASSUMED]`, none fitted.** |
| Policy version | Policy A, uniform diffusion. `src/agent.py` `COSTS`: miss_per_tick 1.0, travel 0.35, exposure 0.25, illegibility 6.0, redundancy 8.0, dither 4.0, horizon 40, reinforcement_call 12.0 |
| Runtime | Python 3.13.2, numpy 2.5.2 |

**Why this case.** Chosen by scanning every tick of the first twelve cases for one where the
agent is genuinely torn rather than already certain. The first candidate was case 0 tick 3, the
clearest example of failure F1, but it made a useless record: the agent had already sighted the
player twice, held 99.6% of its belief in one region, and no new evidence would have moved it. A
record of a decision the agent was already sure about demonstrates nothing §10 asks for.

---

## 1. The situation

| | |
|---|---|
| Guard post | (11, 3) |
| Guard now | (6, 16), Mid corridor |
| Player actually at | **(2, 17), NE room** — hidden from the agent, used only for scoring |
| Walking distance between them | 7 steps |

**Evidence received, in order.** This is everything the agent has.

| tick | event | where |
|---|---|---|
| 0 | sound | (9, 22) — the opening noise |
| 19 | sound | (9, 16) |
| 21 | sighting | (7, 16) |
| 26 | sound | (4, 16) |

Three ticks have passed since the last observation, so the belief has diffused three times
since anything was learned.

## 2. The hidden state

The hidden state is the player's cell, one of 201. That is not a legible table, so cells are
grouped into eight named regions **for reporting only**. The agent never sees these regions and
never reasons over them. `experiments/decision_case.py` asserts they partition the free cells
exactly; that assertion caught two errors in my own region definitions before this was written.

### The prior

| region | cells | P(player here) |
|---|---|---|
| NW room | 24 | 6.1% |
| N corridor | 23 | 5.8% |
| **NE room** | 25 | **51.3%** |
| W rooms | 21 | 3.6% |
| **Mid corridor** | 28 | **22.9%** |
| Centre | 22 | 9.8% |
| SE room | 24 | 0.2% |
| S corridor | 34 | 0.2% |
| **TOTAL** | **201** | **100.0%** |

Entropy 0.798 of maximum. Peak cell (4, 17) at 9.2%, traceability 11.9.

The NE room leads at 51.3%, and **the player is in the NE room**, so the agent's leading
hypothesis is correct. Note the gap between the region total and the peak cell: 51.3% spread
across 25 cells, with no single cell above 9.2%. The agent knows roughly where you are and not
at all precisely.

### The two states the decision turns on

| state | P |
|---|---|
| **H_near** — player within reach of where the guard stands | 16.4% |
| **H_gone** — player anywhere else | 83.6% |

**Evidence for both states, per §10.** For H_gone: nothing observed for three ticks, and the
last three observations trace a path moving away northward, (9, 16) to (7, 16) to (4, 16). For
H_near: the guard is standing one cell from where the tick 21 sighting happened, 22.9% of the
belief is in the region it occupies, and a player doubling back is not excluded. The belief
makes H_gone five times more likely; the evidence for H_near is real but weak.

## 3. The available actions and their costs

Every option the agent priced, ranked. Negative is better: gain terms are credited as
failure-ticks that will not have to be paid later.

| action | target | total | breakdown |
|---|---|---|---|
| **search** | **(4, 17)** | **-12.94** | travel +3.00, gain -20.44, illegibility +0.50, dither +4.00 |
| search | (4, 16) | -12.04 | travel +2.40, gain -18.73, illegibility +0.29, dither +4.00 |
| hold_position | — | +1.07 | miss +1.07 |
| retreat | (7, 16) | +1.42 | miss +1.07, travel +0.35 |
| flank | (7, 16) | +4.63 | travel +0.60, illegibility +0.03, dither +4.00 |
| flank | (6, 17) | +4.78 | travel +0.60, illegibility +0.18, dither +4.00 |
| flank | (6, 18) | +5.47 | travel +1.20, illegibility +0.27, dither +4.00 |
| search | (6, 17) | +12.46 | travel +0.60, illegibility +0.18, **redundancy +7.68**, dither +4.00 |
| search | (6, 18) | +13.15 | travel +1.20, illegibility +0.27, **redundancy +7.68**, dither +4.00 |
| search | (6, 19) | +13.93 | travel +1.80, illegibility +0.45, **redundancy +7.68**, dither +4.00 |
| resume_post | (11, 3) | +36.65 | forfeit +36.65 |

## 4. The policy and the threshold

**The policy is argmin over expected cost.** There is no threshold on belief, deliberately:
hand-set thresholds were beaten by cost-derived boundaries on the previous project, and the
discrepancy sat in its decision record for days unnoticed.

The one boundary that exists is **derived rather than chosen**. The illegibility term is priced
off the unattributable fraction `1/ratio`, where a ratio of 1.0 means the belief in a cell
equals what the agent would have believed having observed nothing at all. That 1.0 is not
tuned; it is where the two beliefs are equal.

## 5. The decision and its reason

**Chosen: `search -> (4, 17)` at -12.94.**

The agent's own recorded reason:

> search; -> (4, 17); at -12.94; drawn by gain -20.44; held back by dither +4.00; next best
> search -12.04; H 0.80; peak traceability 11.9

**The margin is 0.90**, over a target one cell away from the chosen one. This is close to a
tie, and recording it as such matters: a reader seeing only "the agent chose (4, 17)" would
infer more confidence than the comparison supports.

The decision is **correct**. The player is at (2, 17) and the agent is heading into that room.

---

## 6. New evidence: a sound at (6, 20)

One new observation, chosen deliberately to **contradict the leading hypothesis**. The Mid
corridor was second at 22.9%, so a noise inside it tests whether the agent updates against its
own lead rather than merely confirming it.

**This sound could not have come from the player**, who is at (2, 17), seven steps away in a
different region. It is a false positive: a decoy, a door settling, another actor. The agent
has no way to know that, which is the point of the exercise.

### The likelihood, per region

`P(hear this | player in region)`, from `belief.PARAMS`: `sound_sigma` 2.20, `sound_floor`
0.04, with distance measured **through** the map rather than across it, so a cell two steps
away through a wall is as unlikely as the long walk that actually reaches it.

| region | prior | mean likelihood | prior x L | posterior |
|---|---|---|---|---|
| NW room | 6.1% | 0.04 | 0.0025 | 2.9% |
| N corridor | 5.8% | 0.04 | 0.0023 | 2.8% |
| NE room | 51.3% | 0.04 | 0.0206 | **24.8%** |
| W rooms | 3.6% | 0.04 | 0.0015 | 1.8% |
| Mid corridor | 22.9% | **0.21** | 0.0522 | **62.7%** |
| Centre | 9.8% | 0.04 | 0.0040 | 4.8% |
| SE room | 0.2% | 0.04 | 0.0001 | 0.1% |
| S corridor | 0.2% | 0.04 | 0.0001 | 0.1% |
| **TOTAL** | **100.0%** | | **0.0833** | **100.0%** |

`P(evidence) = 0.0833`, the normalising constant: prior times likelihood, summed over every
cell.

The likelihood sits flat at the floor of 0.04 everywhere except the Mid corridor, because a
sound at (6, 20) is many steps through the map from every other region. Only the Mid corridor
holds cells close enough to explain it.

### The posterior, and the two states

| state | before | after |
|---|---|---|
| H_near | 16.4% | **59.4%** |
| H_gone | 83.6% | 40.6% |

Entropy fell from 0.798 to 0.662. Peak moved from (4, 17) at 9.2% to (6, 19) at 17.1%, and peak
traceability rose from 11.9 to 151.3.

### The new decision

| action | target | total | breakdown |
|---|---|---|---|
| **hold_position** | — | **+1.00** | miss +1.00 |
| retreat | (7, 16) | +1.35 | miss +1.00, travel +0.35 |
| flank | (7, 16) | +4.62 | travel +0.60, illegibility +0.02, dither +4.00 |
| flank | (6, 17) | +4.64 | travel +0.60, illegibility +0.04, dither +4.00 |
| flank | (4, 16) | +6.98 | travel +2.40, illegibility +0.58, dither +4.00 |
| search | (6, 17) | +12.32 | travel +0.60, illegibility +0.04, **redundancy +7.68**, dither +4.00 |
| search | (6, 18) | +12.91 | travel +1.20, illegibility +0.03, **redundancy +7.68**, dither +4.00 |
| search | (6, 19) | +13.52 | travel +1.80, illegibility +0.04, **redundancy +7.68**, dither +4.00 |

**Chosen: `hold_position` at +1.00. The action type changed, from `search` to standing still.**

---

## 7. What this case shows, including the parts I did not plan

**The update is arithmetically correct and it made the agent worse.** Before the sound its
leading hypothesis was the NE room at 51.3%, and the player was in the NE room. After the
sound the NE room fell to 24.8% and the agent abandoned a target it was walking toward.

Nothing went wrong in the filter. The prior was reasonable, the likelihood is the one the model
declares, the posterior follows, the action follows from the posterior. **Correct Bayesian
updating on a false observation produces a confidently wrong belief**, and this agent has no
mechanism to doubt an observation: every event it receives is treated as having come from the
player.

This is the adversarial-observation weakness flagged in `research-file.md` under "Can the
player manipulate the evidence with a thrown object or a noise decoy?", now demonstrated on a
real decision with real numbers rather than argued in the abstract. A player who understands
the guard can steer it with one thrown object, and the machinery will cooperate precisely
because it is working correctly.

**The second unplanned finding: the new belief made the agent freeze.** The sound put 62.7% of
the belief in the region the guard was already standing in, and the response was to stand
still. Every cell nearby carries `redundancy +7.68` because the guard has already looked at
them, so searching costs twelve while holding costs one. This is failure **F1** from
`results/failure-analysis.md`, 107 occurrences across the run, and this table is its clearest
single instance: **the agent now believes the player is close, and that belief is exactly what
prices every move toward them out of reach.**

The redundancy term is doing what it was built for. It is also the reason the agent has no
affordable way to close the last few cells, because looking at a cell and going to a cell are
priced as if they were the same act, and only the second one catches anyone.

**Three smaller things worth recording:**

- **Both decisions were near-ties**: 0.90 between two targets one cell apart, then 0.35 between
  holding and retreating. An agent this evenly balanced switches on very little, which is the
  mechanism behind the dithering measured in the failure analysis.
- **Entropy fell while accuracy fell.** The agent became more confident and less correct in the
  same step. Section 3 of the failure analysis shows confidence and correctness are barely
  related in this agent at all.
- **The dither penalty was charged and did not prevent the switch.** +4.00 against a gain of
  -20.44 was never going to bind, which is the same conclusion the weight sweep reaches across
  its entire range.
