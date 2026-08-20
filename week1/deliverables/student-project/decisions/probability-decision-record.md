# Probability Decision Record — Case C045

**What this is.** A complete, replayable audit trail of one entitlement decision made by the Purchase Integrity Agent. Every number below can be recomputed from the referenced data, likelihood tables, and policy version — nothing is a black-box judgment (DC-4). All numeric parameters are **[ASSUMED]** design values, not measured field data.

**Case selection.** C045 from the simulated batch (`simulation/cases.csv`, seed 17). Its true hidden state is withheld from the agent and from this record — at decision time the correct state is not known. (Ground truth exists in `cases.csv` for later scoring; it was not consulted here.)

## Decision record

| Item | Required information |
|---|---|
| **Evidence** | Validated receipt for a **$19.99 consumable** gem pack. Account age 30–365 days (mid); **many prior purchases (>5)**; **no refund history**; purchase from a **device new to the account (<7 days)**; **direct-to-store session** (app opened, straight to purchase, no play first); daytime. |
| **Hidden states** | H1 legitimate purchase; H2 stolen card on the store account; H3 account takeover (ATO); H4 friendly fraud (family member with real access); H5 refund abuse (consume-then-refund). |
| **Beliefs** | Posterior after Bayesian update of the [ASSUMED] priors (92 / 2 / 2 / 3 / 1%) with all evidence above, via the v0.1 likelihood tables: **H1 37.6%, H2 3.5%, H3 31.0%, H4 25.9%, H5 2.1%** (sum = 100%). The belief has *not* separated: no state reaches 50%. |
| **Event** | The policy acts on **P(hostile) = P(H2) + P(H3) = 34.5%** — "any fraud" is deliberately not the event, because `question` is useless against H4/H5 (a real cardholder passes any challenge). Override event P(H5) > 30%: **not triggered** (2.1%). |
| **Actions** | approve = grant entitlement now; question = step-up re-auth before grant; examine = hold grant, human review queue; stop = deny + flag. |
| **Costs** | [ASSUMED] cost matrix, consumable column (nothing is recoverable by revocation — DC-7). Expected cost per action at the current belief: **approve 16.19, question 8.67, examine 11.64, stop 8.30**. Wrong approve on H2/H3 ≈ $19.99 + $15 chargeback/account-health penalty; wrong stop on H1 ≈ half the sale + reputational cost (DC-10); question against H4/H5 = full approve loss *plus* friction. |
| **Policy** | v0.1 bands on P(hostile): <5% approve · 5–25% question · 25–60% examine · >60% stop; override: never `question` when P(H5) > 30%; **DC-6 constraint:** device/session evidence alone cannot escalate past `question` — `examine`/`stop` require at least one account-level risk item (new account, no purchase history, or refund history). |
| **Decision** | Band says **examine** (34.5% ∈ 25–60%). C045 has **no account-level risk item** — its risk comes entirely from the new device + direct-to-store session — so DC-6 caps the action to **`question`: step-up re-auth before granting.** Reason: the belief is genuinely mixed (37.6% legitimate), practitioners rejected blocking on device/session signals alone, and question at least screens the hostile 34.5% while costing a legitimate buyer only friction. |
| **Audit data** | Record time: 2026-08-18 (IST). Data version: `simulation/cases.csv`, seed 17, sim v0.1. Model version: likelihood tables v0.1, 8 evidence variables, [ASSUMED]. Policy version: v0.1 bands + DC-6 (policy "P2"), design-change log through DC-12 (`discussion-record.md`). Code: `simulation/simulator.py`. |

## New evidence arrives

**Step 1 — prior.** The belief above becomes the prior: H1 37.6%, H2 3.5%, H3 31.0%, H4 25.9%, H5 2.1%.

**Step 2 — new evidence (E1).** Account security log shows the **account email address was changed ~30 minutes before this purchase.** This is an *account-level* evidence item.

**Step 3 — likelihoods [ASSUMED].** P(E1 | state): H1 0.01 (legitimate users rarely change email minutes before buying); H2 0.10 (a store-account card thief doesn't need to touch email); **H3 0.60** (locking out the owner is the classic ATO move); H4 0.02 (family member has no reason); H5 0.03.

**Step 4 — posterior.** Multiply and normalize (unnormalized: .00376 / .00348 / .18588 / .00517 / .00062; sum .19891):

**H1 1.9%, H2 1.8%, H3 93.4%, H4 2.6%, H5 0.3%** (sum = 100%). P(hostile) = **95.2%**.

**Step 5 — compare with threshold.** 95.2% > 60% ⇒ `stop` band. DC-6 no longer caps: E1 is account-level evidence. Expected costs at the new belief: approve 33.75, question 8.83, examine 11.56, **stop 0.46** — the band decision and the cost minimum agree.

**Step 6 — new action.** **`stop`: deny the entitlement, flag the account for owner-recovery flow.** One evidence item moved the same case from "screen politely" to "deny" — carried by a likelihood ratio of 60:1 (H3 vs H1), which is the record's point: the *reason* for the escalation is written down and checkable.

## Comparable cases and evidence for both sides

**Comparable recent cases.** Case #017 (prior worked example, project handoff): 14-month account, new device, direct-to-store at 03:12, followed by an email change → posterior 95.4% hostile → stop — consistent with this record's 95.2% on the same evidence class. Within this batch, C009 (old account, many purchases, new device but *normal play* session, daytime) stayed in the question band and resolved toward the family pattern — the session type and account tenure separate the two.

**Evidence searched for the safe state (H1/H4):** many prior purchases and zero refund history (the strongest legitimacy signals per practitioner discussion DC-6); daytime purchase; amount typical for the account's history; new devices happen innocently (phone upgrades, family members — commenter 2).

**Evidence searched for the unsafe state (H2/H3):** device new to a *healthy established* account plus a direct-to-store session with no play — the specific ATO signature practitioners flagged; E1 (email change immediately before purchase) is the discriminating item, worth more than every device signal combined.

## Known weaknesses of this record

1. At the initial belief, the [ASSUMED] cost matrix's myopic minimum was `stop` (8.30) while the eye-tuned bands chose examine/question (8.67) — the bands and the cost matrix disagree inside the 25–60% band. Deriving thresholds *from* the cost matrix is queued for v0.2. (Myopic expected cost also undervalues `examine`, whose payoff is information — the analyst resolves the state.)
2. Naive-Bayes combination double-counts correlated evidence (new device, direct session): the 34.5% initial P(hostile) is probably overconfident.
3. Every likelihood and cost is [ASSUMED]; the record's value is that each one is *visible* and individually challengeable.
