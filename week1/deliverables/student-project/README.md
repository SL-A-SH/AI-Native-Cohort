# Purchase Integrity Agent — Week 1 student project

A probability-based decision agent for in-app purchase entitlement grants, designed,
tested and documented under the Week 1 cohort assignment.

**Problem statement.** The agent observes a validated in-app purchase receipt together
with the buying account's history, device signals, and current session behaviour. It must
approve, question, stop, or examine the entitlement grant because the purchase's true
nature is not known.

The store has already approved the *charge*; the agent decides the *entitlement grant* —
and, after a discussion with practitioners, a second decision at refund-request time.

> **New to this repository? Read `notes/START-HERE.md` first** — it gives a reading order
> with an explanation of what each file is for.

## Repository map

| Path | What it is |
|---|---|
| `research-file.md` | The research file: problem, objective, technical terms, search queries, communities, references, AI prompts and recorded AI errors |
| `discussion-record.md` | Every Reddit/X contribution, the human answers, and the 16 numbered design changes (DC-1…DC-16) they caused |
| `review-record.md` | The three AI reviews (practitioner, probability, preprint) with accept/reject reasoning |
| `decisions/probability-decision-record.md` | One decision recorded in full: evidence, hidden states, beliefs, event, actions, costs, policy, decision, audit data — then a new evidence item and the posterior update |
| `src/simulator.py` | The agent and the experiment as a plain script |
| `experiments/purchase-integrity-simulation.ipynb` | The same content as an executed notebook — read this one |
| `experiments/EXPERIMENT-README.md` | How to reproduce the experiment, the test design, and the headline results |
| `data/cases.csv` | The 50 simulated cases, with the hidden `true_state` used only for scoring |
| `results/results.csv` | Every prediction and action: posteriors, P(hostile), actions and realized costs for all four policies |
| `paper/` | IJCAI-style preprint: `main.tex`, `references.bib`, `figures/`, `preprint.pdf` |
| `social/` | The LinkedIn post and X thread announcing the preprint |
| `notes/` | Working notes not required by the assignment: reading guide, a plain-language retelling of the discussions, and the original project handoff |

## The agent in brief

* **Input** — receipt context: account age, prior purchases, refund history, device age,
  session type, hour, amount, product type.
* **Hidden states** — H1 legitimate · H2 stolen card · H3 account takeover · H4 friendly
  fraud (family member) · H5 refund abuse.
* **Belief** — a posterior over those five, updated from priors by Bayes rule.
* **Actions** — approve (act) · question (step-up auth) · examine (send to a human) ·
  stop (refuse). At the refund surface, a single `refundPreference` output.
* **Costs** — an action × state matrix conditioned on product type, including the
  reputational cost of a wrong refusal.
* **Policy** — a hard-rule gate, then a value-of-information gate, then the
  lowest-expected-cost action. Thresholds are derived from the cost matrix, not eye-tuned.
* **Feedback** — refund and CONSUMPTION_REQUEST notifications; chargebacks arrive weeks
  later, so labels are delayed and selective.
* **Human reasoning function** — *identify uncertainty*: the agent abstains when its
  belief has not separated.

## Reproducing the experiment

```bash
pip install numpy pandas
python src/simulator.py
```

Or open `experiments/purchase-integrity-simulation.ipynb` and Run All. The seed is fixed
(`SEED = 17`), so every number is deterministic. See `experiments/EXPERIMENT-README.md`
for the full instructions, including the Windows UTF-8 note.

## Headline result

Total decision cost across 50 cases: **B0 161.96 → P1 92.88 → P2 92.22 → P3 88.72**,
where B0 is the de facto industry baseline ("grant everything, ban on refund") and P3
derives its thresholds from the cost matrix. The entire margin comes from two hostile
cases; with none in a batch, the baseline wins — which is why the evaluation is
cost-based rather than accuracy-based.

## Honesty notes

* **Every numeric parameter is `[ASSUMED]`.** Priors, likelihoods, and costs are design
  assumptions informed by practitioner discussion; no public IAP-fraud base rates were
  found. The simulation tests the decision machinery, not real-world detection rates.
* **The data is self-simulated.** Cases are generated from the same likelihood tables the
  agent uses, so the belief update is correctly specified by construction.
* **AI use** is stated in `research-file.md` and in the preprint's AI-use statement;
  recorded AI errors are kept in the research file rather than deleted.
