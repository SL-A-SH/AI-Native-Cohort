# Purchase Integrity Agent — Simulation (Week 1)

A minimal, fully auditable simulation of an entitlement-decision agent for in-app
purchases. The agent maintains a belief over five hidden states (legitimate, stolen
card, account takeover, friendly fraud, refund abuse) and selects among approve /
question / examine / stop at grant time, plus a second decision surface — composing
the `refundPreference` field of Apple's CONSUMPTION_REQUEST response (design changes
DC-1…DC-11 in `../discussion-record.md`).

**Every numeric parameter is [ASSUMED]** — priors, likelihood tables, and costs are
design assumptions informed by practitioner discussions, not measured field data. The
simulation tests the decision machinery, not real-world detection rates.

## Files

| File | Contents |
|---|---|
| `purchase-integrity-simulation.ipynb` | The executed notebook: model, policies, metrics, failure analysis, findings |
| `simulator.py` | Same content as a plain script (jupytext percent format) — the notebook is generated from this |
| `cases.csv` | The 50 simulated cases: evidence columns + hidden `true_state` (never read by any policy) |
| `results.csv` | All predictions and actions: per-case posteriors, P(hostile), actions and realized costs for all three policies, refund-stage outputs |

## How to reproduce

1. Requirements: Python ≥ 3.10 with `numpy` and `pandas`
   (`pip install numpy pandas`). To re-execute the notebook itself, also
   `pip install jupytext nbclient ipykernel`.
2. Run the script directly:

   ```
   python simulator.py
   ```

   or regenerate + execute the notebook:

   ```
   jupytext --to ipynb simulator.py -o purchase-integrity-simulation.ipynb
   jupyter execute purchase-integrity-simulation.ipynb   # or nbclient
   ```

   **Windows note.** `jupyter execute` opens the notebook with the system code
   page (cp1252 on most Windows installs), which fails on the UTF-8 characters
   in the markdown cells:

   ```
   UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d ...
   ```

   This is an encoding default, not a corrupt file. Enable Python's UTF-8 mode:

   ```powershell
   $env:PYTHONUTF8=1                                      # PowerShell
   jupyter execute purchase-integrity-simulation.ipynb
   ```

   ```
   python -X utf8 -m nbclient purchase-integrity-simulation.ipynb   # one-shot equivalent
   ```

   `python simulator.py` is unaffected (it reads no text files), and opening the
   notebook in VS Code or JupyterLab and choosing **Run All** also works, since
   both read as UTF-8.

   The shipped `.ipynb` already contains the outputs from the authoring run, so
   re-execution verifies reproducibility rather than being required to read the
   results.

3. The random seed is fixed (`SEED = 17`) at the top of the script. Every number in
   the notebook — case draws, posteriors, metrics, the refund-stage draws — is
   deterministic given that seed. Changing the seed changes the sampled cases but not
   the model, policies, or cost matrix.

## Experiment design (assignment section 9 checklist)

- 50 simulated cases, composition 45/1/1/2/1 across H1–H5 (approximates the [ASSUMED]
  92/2/2/3/1% priors while guaranteeing every state appears). The `true_state` label
  is hidden from all policies — it is used only for scoring afterwards.
- Four policies compared: **B0** baseline (always approve + reactive ban — the de
  facto industry policy per r/reactnative), **P1** eye-tuned threshold bands on
  P(hostile)=P(H2)+P(H3), **P2** = P1 + the DC-6 evidence-hierarchy constraint, and
  **P3** = the cost-derived policy: a hard-rule gate, then a value-of-information gate
  (buy a human review only when VoI exceeds its fee), then the lowest-expected-cost
  action. P3 contains no hand-set bands — every boundary follows from the cost matrix.
- An **evidence planner** ranks every permitted check by information gain in bits
  (Shannon entropy over the five states), including a deliberate zero-gain control.
- Metrics beyond accuracy: action × state confusion tables, intervention
  precision/recall, FP/FN counts plus *effective* FN (fraud not actually blocked),
  human-review rate, total decision cost, calibration table + Brier score.
- Six incorrect/degraded decisions examined and named (F1–F6 in the notebook),
  including the highest-cost error and why it is the highest.

## Headline results (seed 17)

Total decision cost: **B0 161.96 → P1 92.88 → P2 92.22 → P3 88.72.** The entire margin
comes from the two hostile cases; with no hostile cases in a batch the baseline wins —
itself the finding that motivates cost-sensitive evaluation.

Cost-derived thresholds strictly dominate the eye-tuned bands: P3 is cheapest, has the
fewest false positives (1 vs 2), the best intervention precision (0.75 vs 0.67), and the
same hostile recall (1.0) and human-review rate (2%). For a $19.99 consumable the derived
boundaries are approve <8.2%, question to 71.5%, stop above — against eye-tuned
5%/25%/60%. The cost ratio is only ~1.75:1, which is the quantitative case for the
practitioners' "don't gate at grant time" advice (DC-12).

Value of information justified a human review in exactly 1 of 50 cases — including a
"no" on C045, the most genuinely ambiguous case in the set. The refund-abuse case evaded
both decision surfaces (failures F1 and F6), producing the main outstanding change: give
consumption magnitude a likelihood at refund time.

**Sensitivity to the contested cost (DC-16).** A practitioner disputes the chargeback
penalty on hostile approvals ("your worst case is capped at the revenue anyway, since
Apple is the merchant of record"). Setting it to zero moves the stop gate from 71.5% to
81.4% on a $19.99 consumable and to 91.6% on a $4.99 one, where the ratio inverts to 0.40
— a wrong denial costing more than twice a wrong approval. The ranking does not move
(B0 131.96 > P1 92.88 > P2 88.47 > P3 80.97) and one action in 50 changes. The only cell
still carrying an adverse ratio at zero penalty is the **large consumable** (1.43, gate at
75.7%): high value, zero recoverability — the one corner where blocking can pay.

Each of the six named failures is tagged with the architecture layer that was wrong
(L0 costs, L1 worlds, L2 what is usual here, L3 answer shape, L4 action rule), so a
repair has an address.

**Version note.** v0.2 replaced the hand-set `examine` cost row with the
value-of-information treatment (pay the fee, learn the state, act optimally), so P1/P2
totals differ from the v0.1 run (99.47/96.72 → 92.88/92.22). The ranking is unchanged.
