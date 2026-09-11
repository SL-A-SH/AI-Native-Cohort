# NPC Tactical Belief Agent

An NPC that maintains a probability distribution over a hidden player's position on a grid,
updates it from noisy observations, and chooses one of six actions by minimising an expected
cost. Three of the cost terms price how the behaviour *reads* rather than how accurate it is.

**The headline result is a loss.** Across 8 case-generation seeds the agent is beaten on capture
rate by two much cheaper scripted controls in every seed, while re-searching cleared ground
about a tenth as often. A factorial ablation shows the mechanism responsible for that advantage
is not the one the project assumed. Details in `paper/preprint.pdf`.

**This is a course preprint.** It has not been submitted to IJCAI or any other venue and no
claim of review or acceptance is made.

---

## Reproducing every number

Python 3.13, numpy 2.5, matplotlib 3.11. From this directory:

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install numpy matplotlib

python experiments/run.py --cases 40         # -> results/decisions.csv, results/episodes.csv
python experiments/analyse.py                # -> results/failures.csv, prints the failure analysis
python experiments/sweep.py --mode seeds     # -> results/sweep-seeds.csv   (8 seeds, ~90s)
python experiments/sweep.py --mode dither    # -> results/sweep-dither.csv  (8 seeds x 6 weights, ~9min)
python experiments/ablation.py               # -> results/ablation.csv      (2x2 factorial, ~6min)
python experiments/decision_case.py          # prints the worked case in decisions/
```

**Run `run.py` immediately before `analyse.py`.** `analyse.py` reads whatever is in
`results/decisions.csv` and does not check it is current. An earlier analysis was written
against a stale file that a 6-case smoke test had overwritten; it reported 30 episodes instead
of 200 and was only caught because the episode count looked wrong.

Figures and video:

```bash
python src/day3_tuning_figure.py       # -> figures/day3-the-knob.png
python src/day4_search_figure.py       # -> figures/day4-failed-search.png
python src/day4_animation.py           # -> figures/day4-failed-search.mp4 + .gif   (needs ffmpeg)
python src/day5_attribution_figure.py  # -> figures/day5-attribution.png
python src/day6_dither_video.py        # -> figures/day6-dither.mp4 + .gif          (needs ffmpeg)
```

The paper (needs a LaTeX toolchain; [Tectonic](https://tectonic-typesetting.github.io) works
without a system install):

```bash
cd paper && tectonic -X compile main.tex && mv main.pdf preprint.pdf
```

Everything is deterministic. Given the same seeds you get the same numbers; there is no
untracked randomness.

---

## What lives where

```
README.md                     this file
research-file.md              problem, terms, sources, questions, AI errors (three of them)
discussion-record.md          nine human answers, each with one recorded outcome
review-record.md              three AI reviews and the eight self-reviews during the build
decisions/
  probability-decision-record.md   one decision in full, prior -> likelihood -> posterior -> action
paper/
  main.tex, references.bib, preprint.pdf, figures/, ijcai26.sty, named.bst
src/
  world.py                    the map, distances, line of sight, the simulated player
  belief.py                   the filter: predict, update, attributability
  agent.py                    six actions, the cost function, the decision rule
  baselines.py                three controls
  day*_*.py                   figure and video scripts
experiments/
  run.py                      the main experiment; hides the label at decision time
  analyse.py                  confusion matrix, calibration, escalation, named failures
  sweep.py                    seed sweep and dither sweep
  ablation.py                 the 2x2 that separates negative information from the memory penalty
  decision_case.py            generates the probability decision record
data/cases.csv                the 40 scenarios, so they can be inspected without running the generator
results/                      all output, plus failure-analysis.md
notes/                        handover, Unreal port notes, the three AI review prompts
social/                       the LinkedIn and X drafts
figures/                      every figure and video
```

---

## The experiment in one paragraph

Forty scenarios spread deliberately across five guard posts and four bands of starting distance
(4 to 38 steps, `data/cases.csv`), not sampled at random. Each episode opens with one noise from
the player's starting cell and runs to capture, stand-down, or 80 ticks. Every scenario is
replayed identically for all five arms, so the comparison is paired. The player walks to
randomly chosen goals with a 15% chance of a random step and **never reacts to the agent**,
which is the single biggest limitation of the whole study.

**The label is hidden at decision time.** The player's true position is used for exactly three
things: deciding what observations to emit, supplying the degraded oracle, and scoring
afterwards. No argument to the agent's `decide()` carries it, and `world.Player` appears in no
signature in `agent.py`.

---

## Headline numbers

8 case-generation seeds x 40 scenarios, mean ± SD across seeds:

| arm | capture | redundant searches | attribution |
|---|---|---|---|
| belief (policy A) | 72.2% ±7.8 | **10.9** ±5.4 | 13.9 |
| belief (policy B) | 71.2% ±5.6 | 9.8 ±6.9 | 13.2 |
| last-known-position | 79.1% ±5.1 | 114.5 ±27.7 | 22.2 |
| scripted sweep | **83.7%** ±6.8 | 136.9 ±37.9 | 21.9 |
| degraded oracle | 62.8% ±5.4 | 248.9 ±35.5 | 11.4 |

The 2x2 ablation, which is the result that most changed our reading:

| negative info | memory penalty | redundant/commit | capture |
|---|---|---|---|
| off | off | 0.363 | **73.4%** |
| on | off | 0.029 | 71.9% |
| off | on | **0.002** | 65.9% |
| on | on | 0.014 | 72.2% |

Either mechanism alone produces most of the effect; together is no better than either. The
hand-coded memory gives the best redundancy figure and the worst capture rate, so reading that
column alone selects the worst agent. Both believability mechanisms cost capture.

---

## Things a reader should know before trusting this

- **No human has judged this agent.** The dependent variable is how behaviour reads to an
  observer and we measure a proxy. This is the central validity limitation.
- **One hand-made map.** Chokepoints, points of interest, distances and visibility all derive
  from a single grid whose corridors are one cell wide.
- **Roughly fifteen assumed parameters**, none fitted. Not fitting them means we did not
  optimise against our own results; it does not make them justified.
- **The primary metric is biased.** Scoring a decision by whether the player is findable at the
  chosen target *at that instant* favours policies that chase the freshest observation, and
  penalises `flank` for doing what its own definition specifies.
- **Two actions are never chosen**, and a third is priced with a benefit the simulation cannot
  deliver because no second agent exists.
- **The belief is overconfident** by about 1.7x at the top of its range. Do not read its numbers
  as probabilities.
- **Ten bugs were found during the build**, four of which changed a headline number. Behaviour
  here is sensitive to bookkeeping choices.

---

## AI use

Large language models were used to draft and refactor code, propose the experimental design,
draft the paper, and review the work. Three structured reviews (practitioner, probability,
preprint) were commissioned before writing, with the prompts kept in
`notes/review-prompts.md`. They found the double-counted negative information, the single-seed
sweep, the confound requiring the ablation, and the prior art that retracted the project's
original motivation.

AI also introduced errors. `research-file.md` records three, including two false claims about
what shipped games do, the second of which survived four apparent human confirmations before
being checked against the literature. Every number reported here was produced by the code in
this repository and verified against its output.
