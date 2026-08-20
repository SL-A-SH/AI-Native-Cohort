# %% [markdown]
# # Purchase Integrity Agent — Simulation Notebook (Week 1)
#
# **Problem.** The agent observes a validated in-app purchase receipt together with the
# buying account's history, device signals, and current session behaviour. It must
# approve, question, stop, or examine the entitlement grant because the purchase's true
# nature is not known. The store already approved the *charge*; the agent decides the
# *entitlement grant*.
#
# **Assignment mapping (section 8).**
#
# | Part | This notebook |
# |---|---|
# | Input | receipt context: account age, prior purchases, refund history, device age, session type, hour, amount, product type |
# | Hidden state | H1 legitimate, H2 stolen card, H3 account takeover, H4 friendly fraud, H5 refund abuse |
# | Belief | posterior over H1–H5 via Bayes rule (naive-Bayes evidence combination) |
# | Action | approve (act), question (ask a question / step-up auth), examine (send to a human), stop (refuse) |
# | Cost | per action × state cost matrix, conditioned on product type (DC-7), incl. reputational cost (DC-10) |
# | Policy | thresholds on the event P(hostile) = P(H2)+P(H3) with overrides (DC-6); in P3, boundaries derived from the cost matrix instead of eye-tuned |
# | Feedback | refund / CONSUMPTION_REQUEST notifications; chargebacks arrive weeks later (selective labels — see limitations) |
#
# **Architecture mapping (course layers L0–L5 / agent versions V0–V5).**
#
# | Layer | Hands upward | Where it lives here |
# |---|---|---|
# | L0 what counts as failure | the cost of being wrong | §4 cost matrix + §6 harness (labels hidden at decision time) |
# | L1 what could be true | a list of possible worlds | §0 five hidden states, mutually exclusive, sum to 1 |
# | L2 what is usually true here | a share for each world | §1 priors + likelihood tables, §3 Bayesian update |
# | L3 what shape the answer takes | a typed answer, not a paragraph | posterior vector, P(hostile), `refundPreference` enum — no prose anywhere in the decision path |
# | L4 what to do about it | one action | §5 policies; §5b derives the boundaries; §5d the V5 state machine |
# | L5 why it did that | a trace you can read | `results.csv` + `decisions/probability-decision-record.md` |
#
# **Human reasoning function (one, per the assignment): identify uncertainty.**
# The agent abstains when its belief has not separated — at grant time via the
# `examine` action, and at the consumption surface by deliberately sending
# "no preference" on a flat posterior (DC-9, practitioner-validated).
#
# **Every numeric parameter in this notebook is [ASSUMED]** — priors, likelihoods, and
# costs are design assumptions informed by practitioner discussions (see
# discussion-record.md, DC-1…DC-11), not measured field data. No public IAP-fraud base
# rates were found. The simulation tests the *policy machinery*, not real-world rates.

# %%
from pathlib import Path

import numpy as np
import pandas as pd

# Paths resolve against the project root whether this runs as src/simulator.py or as
# experiments/purchase-integrity-simulation.ipynb — both sit one level below the root.
try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:                      # notebook: no __file__
    ROOT = Path.cwd().parent
DATA_DIR, RESULTS_DIR = ROOT / "data", ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 17
rng = np.random.default_rng(SEED)

STATES = ["H1_legit", "H2_stolen_card", "H3_ato", "H4_friendly", "H5_refund_abuse"]
PRIORS = np.array([0.92, 0.02, 0.02, 0.03, 0.01])  # [ASSUMED] — no public base rates found
HOSTILE = ["H2_stolen_card", "H3_ato"]              # the event the policy acts on
FRAUD   = STATES[1:]                                # any non-legit state

assert abs(PRIORS.sum() - 1.0) < 1e-12

# %% [markdown]
# ## 1. Evidence model — [ASSUMED] likelihood tables
#
# Each evidence variable is discrete. `P(value | state)` tables below encode the design
# assumptions. Account-level variables (account age, prior purchases, refund history)
# are distinguished from device/session-level variables (device age, session type, hour)
# because DC-6 forbids escalation beyond `question` on device/session evidence alone.
#
# Known weakness (logged before this simulation was written): naive-Bayes combination
# double-counts correlated evidence — a new device, a direct-to-store session, and a
# night-time hour are correlated in reality but treated as independent here.

# %%
# P(value | state) per variable; rows follow STATES order.
LIK = {
    "account_age": {          # new (<30d), mid (30–365d), old (>365d)
        "values": ["new", "mid", "old"],
        "p": np.array([
            [0.15, 0.35, 0.50],   # H1
            [0.70, 0.20, 0.10],   # H2 stolen card: prefers fresh accounts
            [0.05, 0.25, 0.70],   # H3 ATO: takes over established accounts
            [0.10, 0.30, 0.60],   # H4 family member on an established account
            [0.50, 0.35, 0.15],   # H5 refund abusers cycle newer accounts
        ]),
    },
    "prior_purchases": {      # none, few (1–5), many (>5)
        "values": ["none", "few", "many"],
        "p": np.array([
            [0.25, 0.45, 0.30],
            [0.80, 0.15, 0.05],
            [0.10, 0.40, 0.50],
            [0.20, 0.40, 0.40],
            [0.40, 0.45, 0.15],
        ]),
    },
    "refund_history": {       # yes, no  (account-level)
        "values": ["yes", "no"],
        "p": np.array([
            [0.05, 0.95],
            [0.05, 0.95],
            [0.05, 0.95],
            [0.15, 0.85],
            [0.60, 0.40],     # prior refunds are the H5 tell
        ]),
    },
    "device_age": {           # new (<7d on this account), known
        "values": ["new", "known"],
        "p": np.array([
            [0.10, 0.90],
            [0.85, 0.15],
            [0.80, 0.20],
            [0.50, 0.50],     # family member's own device often "new to the account"
            [0.30, 0.70],
        ]),
    },
    "session_type": {         # direct_to_store (app open → straight to purchase), normal_play
        "values": ["direct", "normal"],
        "p": np.array([
            [0.10, 0.90],
            [0.90, 0.10],
            [0.75, 0.25],
            [0.40, 0.60],
            [0.55, 0.45],
        ]),
    },
    "hour": {                 # night (00:00–05:59 local), day
        "values": ["night", "day"],
        "p": np.array([
            [0.10, 0.90],
            [0.45, 0.55],
            [0.50, 0.50],
            [0.15, 0.85],
            [0.20, 0.80],
        ]),
    },
    "amount": {               # small ($4.99), medium ($19.99), large ($49.99)
        "values": ["small", "medium", "large"],
        "p": np.array([
            [0.40, 0.40, 0.20],
            [0.10, 0.30, 0.60],
            [0.10, 0.30, 0.60],
            [0.30, 0.40, 0.30],
            [0.15, 0.40, 0.45],
        ]),
    },
    "product_type": {         # consumable, non_consumable, subscription
        "values": ["consumable", "non_consumable", "subscription"],
        "p": np.array([
            [0.55, 0.25, 0.20],
            [0.70, 0.20, 0.10],
            [0.70, 0.20, 0.10],
            [0.60, 0.25, 0.15],
            [0.90, 0.07, 0.03],  # DC-7: consume-then-refund is consumable-specific
        ]),
    },
}
for var, spec in LIK.items():
    assert np.allclose(spec["p"].sum(axis=1), 1.0), var

ACCOUNT_LEVEL_VARS = ["account_age", "prior_purchases", "refund_history"]
AMOUNT_VALUE = {"small": 4.99, "medium": 19.99, "large": 49.99}

# %% [markdown]
# ## 2. Case generation (N = 50)
#
# Composition is fixed to approximate the priors while guaranteeing every hidden state
# appears at least once: 45 / 1 / 1 / 2 / 1 (= 90/2/2/4/2%, priors are 92/2/2/3/1%).
# Evidence is sampled from the state's likelihood tables. **The `true_state` column is
# never read by any policy** — it is used only afterwards for scoring, satisfying the
# "hide the correct label when the agent makes a decision" requirement.

# %%
COMPOSITION = {"H1_legit": 45, "H2_stolen_card": 1, "H3_ato": 1, "H4_friendly": 2, "H5_refund_abuse": 1}
assert sum(COMPOSITION.values()) == 50

rows = []
for state, n in COMPOSITION.items():
    si = STATES.index(state)
    for _ in range(n):
        row = {"true_state": state}
        for var, spec in LIK.items():
            row[var] = rng.choice(spec["values"], p=spec["p"][si])
        rows.append(row)

cases = pd.DataFrame(rows).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
cases.insert(0, "case_id", [f"C{i:03d}" for i in range(1, len(cases) + 1)])
cases["amount_usd"] = cases["amount"].map(AMOUNT_VALUE)
cases.to_csv(DATA_DIR / "cases.csv", index=False)
cases.head(10)

# %% [markdown]
# ## 3. Belief update
#
# Posterior ∝ prior × ∏ P(evidence | state), normalized. Explicit and replayable:
# every posterior can be recomputed from the tables above (DC-4 — a practitioner
# independently demanded "a rule I can audit and replay rather than something
# generative").

# %%
EVIDENCE_VARS = list(LIK.keys())

def posterior(row) -> np.ndarray:
    p = PRIORS.copy()
    for var in EVIDENCE_VARS:
        vi = LIK[var]["values"].index(row[var])
        p = p * LIK[var]["p"][:, vi]
    p = p / p.sum()
    assert abs(p.sum() - 1.0) < 1e-9
    return p

post = np.vstack([posterior(r) for _, r in cases.iterrows()])
results = cases[["case_id", "true_state", "amount", "amount_usd", "product_type"]].copy()
for i, s in enumerate(STATES):
    results[f"P_{s}"] = post[:, i]
results["P_hostile"] = post[:, 1] + post[:, 2]
results["P_fraud_any"] = post[:, 1:].sum(axis=1)

# account-level risk flags for DC-6
results["account_risk"] = (
    (cases["account_age"] == "new")
    | (cases["prior_purchases"] == "none")
    | (cases["refund_history"] == "yes")
)

# %% [markdown]
# ## 4. Cost model — [ASSUMED], conditioned on product type (DC-7) with reputational cost (DC-10)
#
# Units ≈ USD of expected loss. `recoverable = 0.7` for non-consumables/subscriptions
# (entitlement revocation recovers most value, DC-7); `0.0` for consumables.
# `question` is useless against H4/H5 — the real cardholder passes any challenge —
# so its cost there equals approving *plus* the friction (this is why the policy
# thresholds act on P(hostile), not P(any fraud)).

# %%
# DC-16 [ASSUMED, CONTESTED]. A practitioner states the developer's worst case is capped
# at the revenue "since Apple is the merchant of record" — i.e. the platform, not the
# developer, absorbs card-network chargeback fees. The original matrix added a flat
# penalty on top of the item value for hostile states. Because the whole gate/no-gate
# argument turns on this number, it is a switch rather than a constant, and §5e re-derives
# every boundary with it set to zero. [VERIFY against Apple's developer terms before
# citing the merchant-of-record claim.]
CHARGEBACK_PENALTY = 15.0

def analyst_fee(amt: float) -> float:
    """Full price of one human review: analyst time + the delay it imposes. [ASSUMED]"""
    return 8.0 + 1.0 + 0.02 * amt

def cost(action: str, state: str, amt: float, product: str) -> float:
    rec = 0.0 if product == "consumable" else 0.7
    loss = amt * (1 - rec)
    approve_c = {
        "H1_legit": 0.0,
        "H2_stolen_card": loss + CHARGEBACK_PENALTY,   # chargeback fee + account-health penalty
        "H3_ato":         loss + CHARGEBACK_PENALTY,
        "H4_friendly":    0.6 * loss + 2.0,  # Apple grants most of these refunds anyway
        "H5_refund_abuse": loss + 5.0,   # abuse handling, not a chargeback — DC-16 leaves this alone
    }[state]
    if action == "approve":
        return approve_c
    friction = 1.5 + 0.05 * amt          # step-up friction + abandonment expectation
    if action == "question":
        return {
            "H1_legit": friction,
            "H2_stolen_card": 0.15 * approve_c,      # re-auth blocks ~85%
            "H3_ato":         0.25 * approve_c,      # attacker may control email/2FA
            "H4_friendly":    approve_c + friction,  # passes — question is useless vs H4
            "H5_refund_abuse": approve_c + friction, # passes — useless vs H5
        }[state]
    if action == "examine":
        # v0.2 change (course Ch.5 alignment): `examine` is no longer a hand-set cost
        # row. It is priced as "pay the review fee, learn the state, then take the best
        # action for that state" — the standard value-of-information treatment. This is
        # what makes `examine` derivable rather than asserted.
        # [ASSUMED] the analyst is perfect. That is optimistic: a real reviewer is
        # imperfect and slow, so this OVERSTATES the value of examine (see limitations).
        return analyst_fee(amt) + min(
            cost(a, state, amt, product) for a in ("approve", "question", "stop")
        )
    if action == "stop":
        return {
            "H1_legit": 0.5 * amt + 10.0,  # lost sale + reputational (DC-10)
            "H2_stolen_card": 0.0,
            "H3_ato": 0.0,
            "H4_friendly": 3.0,
            "H5_refund_abuse": 0.0,
        }[state]
    raise ValueError(action)

# %% [markdown]
# ## 5. Policies
#
# * **B0 — baseline (DC-8):** always approve, ban reactively on refund. This is the de
#   facto industry policy per r/reactnative ("detect refund → instaban"). The ban's
#   value (preventing *repeat* abuse) is outside this per-transaction simulation — a
#   disclosed limitation that favours the baseline less than reality would.
# * **P1 — thresholds v0.1:** bands on P(hostile)=P(H2)+P(H3): <5% approve, 5–25%
#   question, 25–60% examine, >60% stop. Override: if P(H5)>30%, never `question`
#   (the abuser passes any challenge) — escalate to `examine`.
# * **P2 — v0.1 + evidence hierarchy (DC-6):** as P1, but device/session signals alone
#   cannot push past `question`: `examine`/`stop` require at least one account-level
#   risk flag (new account, no prior purchases, or refund history).

# %%
def policy_band(p_hostile: float) -> str:
    if p_hostile < 0.05:
        return "approve"
    if p_hostile < 0.25:
        return "question"
    if p_hostile < 0.60:
        return "examine"
    return "stop"

def policy_P1(row) -> str:
    a = policy_band(row["P_hostile"])
    if a == "question" and row["P_H5_refund_abuse"] > 0.30:
        a = "examine"
    return a

def policy_P2(row) -> str:
    a = policy_P1(row)
    if a in ("examine", "stop") and not row["account_risk"]:
        a = "question"      # DC-6 cap: no escalation on device/session evidence alone
    return a

results["action_B0"] = "approve"
results["action_P1"] = results.apply(policy_P1, axis=1)
results["action_P2"] = results.apply(policy_P2, axis=1)

# %% [markdown]
# ## 5b. Where the thresholds should actually be — derived from the cost matrix
#
# P1/P2's bands (5% / 25% / 60%) were eye-tuned; the decision record flagged that they
# disagree with the cost matrix. A threshold is not a taste — it is the belief level at
# which the cheaper action changes, and it follows from the cost ratio by algebra.
#
# For the canonical case (a $19.99 consumable, hostile mass split evenly between H2 and
# H3, remaining mass on H1):
#
# * `approve` costs `p × (amount + 15)` — the entitlement plus chargeback/account-health
#   penalty, unrecoverable because a consumable cannot be revoked.
# * `stop` costs `(1 − p) × (0.5 × amount + 10)` — half the lost sale plus the
#   reputational term (DC-10).
#
# Setting them equal gives the approve/stop indifference point directly. The cell below
# solves all three boundaries numerically and sweeps the whole belief range.

# %%
ACT_NOW = ("approve", "question", "stop")   # actions available without buying information

def belief_from_p(p_hostile: float) -> np.ndarray:
    """Canonical belief: p split evenly across H2/H3, the rest on H1. [ASSUMED shape]"""
    b = np.zeros(5)
    b[0] = 1 - p_hostile
    b[1] = b[2] = p_hostile / 2
    return b

def expected_cost(action: str, belief: np.ndarray, amt: float, product: str) -> float:
    return float(sum(belief[i] * cost(action, s, amt, product) for i, s in enumerate(STATES)))

def voi(belief: np.ndarray, amt: float, product: str) -> float:
    """Value of resolving the hidden state: best-now cost minus best-under-perfect-info."""
    best_now = min(expected_cost(a, belief, amt, product) for a in ACT_NOW)
    perfect = float(sum(belief[i] * min(cost(a, s, amt, product) for a in ACT_NOW)
                        for i, s in enumerate(STATES)))
    return best_now - perfect

CANON_AMT, CANON_PROD = 19.99, "consumable"
sweep_rows = []
for p in [0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.20, 0.25, 0.40, 0.60, 0.70, 0.75, 0.90, 1.0]:
    b = belief_from_p(p)
    row = {"P(hostile)": p}
    for a in ACT_NOW:
        row[a] = round(expected_cost(a, b, CANON_AMT, CANON_PROD), 2)
    row["VoI"] = round(voi(b, CANON_AMT, CANON_PROD), 2)
    row["fee"] = round(analyst_fee(CANON_AMT), 2)
    row["cheapest now"] = min(ACT_NOW, key=lambda a: expected_cost(a, b, CANON_AMT, CANON_PROD))
    row["buy review?"] = "yes" if row["VoI"] > row["fee"] else "no"
    sweep_rows.append(row)
sweep = pd.DataFrame(sweep_rows).set_index("P(hostile)")
print(sweep)

# locate the boundaries to 0.1% resolution
grid = np.arange(0, 1.0005, 0.001)
cheapest = [min(ACT_NOW, key=lambda a: expected_cost(a, belief_from_p(p), CANON_AMT, CANON_PROD))
            for p in grid]
bounds = [(round(float(grid[i]), 3), cheapest[i - 1], cheapest[i])
          for i in range(1, len(grid)) if cheapest[i] != cheapest[i - 1]]
print("\nCost-derived boundaries (canonical $19.99 consumable):")
for p, a, b_ in bounds:
    print(f"  {a} -> {b_} at P(hostile) = {p:.1%}")
print(f"\nEye-tuned bands for comparison: approve <5%, question 5-25%, examine 25-60%, stop >60%")

# %%
def boundaries(amt: float, product: str) -> list:
    """Belief levels where the cheapest action changes, to 0.1% resolution."""
    g = np.arange(0, 1.0005, 0.001)
    best = [min(ACT_NOW, key=lambda a: expected_cost(a, belief_from_p(p), amt, product)) for p in g]
    return [(round(float(g[i]), 3), best[i - 1], best[i])
            for i in range(1, len(g)) if best[i] != best[i - 1]]

def describe_gate(amt, product):
    bs = boundaries(amt, product)
    stop_at = next((p for p, _, to in bs if to == "stop"), None)
    return "never stops" if stop_at is None else f"stop above {stop_at:.1%}"

# %% [markdown]
# ## 5c. What should the agent ask next? — information gain (course Ch.4)
#
# P1/P2 receive all evidence at once and never choose what to look at. A real agent buys
# evidence one item at a time, so it needs to rank the *permitted* checks before spending
# one. The ranking is entropy-based: `H = −Σ pᵢ log₂ pᵢ` measured in bits, and a check's
# **information gain** is `H(before) − Σ_answer P(answer) × H(belief | answer)` — the
# leftover uncertainty after each possible answer, weighted by how often that answer
# occurs (the agent does not get to choose the answer it receives).
#
# Two additions to the evidence set for this section:
#
# * `email_changed` — the account-security check used in the probability decision record
#   (P(change | state) = 0.01 / 0.10 / 0.60 / 0.02 / 0.03, [ASSUMED]). It is *not* part of
#   the routinely-observed 8 variables; it must be bought.
# * `coin_flip` — a deliberate zero-gain control (identical likelihood under every state),
#   the course's "favourite colour" tool. Any planner that ranks it above zero is broken.

# %%
def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

CHECKS = {var: LIK[var]["p"] for var in EVIDENCE_VARS}
CHECKS["email_changed"] = np.array([[0.01, 0.99],   # P(changed | H1), P(not changed | H1)
                                    [0.10, 0.90],
                                    [0.60, 0.40],
                                    [0.02, 0.98],
                                    [0.03, 0.97]])
CHECKS["coin_flip"] = np.array([[0.5, 0.5]] * 5)    # zero-gain control

def info_gain(belief: np.ndarray, table: np.ndarray) -> float:
    """Expected reduction in entropy (bits) from running a check with this likelihood table."""
    h_before = entropy(belief)
    h_after = 0.0
    for j in range(table.shape[1]):
        joint = belief * table[:, j]
        p_answer = joint.sum()
        if p_answer <= 0:
            continue
        h_after += p_answer * entropy(joint / p_answer)
    return h_before - h_after

def rank_checks(belief: np.ndarray) -> pd.DataFrame:
    rows = [{"check": k, "gain (bits)": round(info_gain(belief, t), 3)} for k, t in CHECKS.items()]
    return (pd.DataFrame(rows).sort_values("gain (bits)", ascending=False)
            .reset_index(drop=True))

print(f"Prior entropy: {entropy(PRIORS):.3f} bits\n")
print("Ranking from the priors — what to look at first, before anything is known:")
print(rank_checks(PRIORS).to_string(index=False))

# The planner re-ranks after every answer. C045 is the decision-record case: an
# established account, new device, direct-to-store session — belief already updated on
# all 8 routine variables, still unresolved.
c045 = results[results.case_id == "C045"].iloc[0]
b045 = np.array([c045[f"P_{s}"] for s in STATES])
print(f"\nC045 belief: {dict(zip(STATES, b045.round(3)))}")
print(f"C045 entropy: {entropy(b045):.3f} bits — the belief has not separated")
print("\nRanking at C045's belief (routine variables already spent):")
print(rank_checks(b045).to_string(index=False))
print(f"\nVoI of a human review at C045: {voi(b045, c045['amount_usd'], c045['product_type']):.2f} "
      f"vs fee {analyst_fee(c045['amount_usd']):.2f}")

# %% [markdown]
# ## 5d. Policy P3 — the course's V5 state machine
#
# Hard rule first, then buy information only if it pays, then act on the cheapest option.
# No band appears anywhere in this policy: every boundary comes from the cost matrix.
#
# 1. **Hard-rule gate (DC-6):** without account-level evidence, `examine` and `stop` are
#    not available — device/session signals alone may not escalate.
# 2. **Ask gate (VoI):** if a human review is permitted and `VoI > analyst_fee`, examine.
# 3. **Act gate:** otherwise take the lowest-expected-cost action among the allowed ones.

# %%
def policy_P3(row) -> str:
    belief = np.array([row[f"P_{s}"] for s in STATES])
    amt, prod = row["amount_usd"], row["product_type"]
    allowed = list(ACT_NOW) if row["account_risk"] else ["approve", "question"]   # gate 1
    if row["account_risk"] and voi(belief, amt, prod) > analyst_fee(amt):         # gate 2
        return "examine"
    return min(allowed, key=lambda a: expected_cost(a, belief, amt, prod))        # gate 3

results["action_P3"] = results.apply(policy_P3, axis=1)

for pol in ["B0", "P1", "P2", "P3"]:
    results[f"cost_{pol}"] = results.apply(
        lambda r: cost(r[f"action_{pol}"], r["true_state"], r["amount_usd"], r["product_type"]), axis=1
    )

# %% [markdown]
# ## 5e. DC-16 — does the gate survive if the platform absorbs the chargeback?
#
# The gate/no-gate argument turns on one [ASSUMED] number. A practitioner states the
# developer's worst case is *capped at the revenue* because Apple is the merchant of
# record, which would make the flat penalty on hostile approves wrong. If that is true,
# a wrong approve costs no more than a wrong denial, and the arithmetic — not the
# practitioners — retires grant-time gating.
#
# The cell below re-derives every boundary and re-runs all four policies with
# `CHARGEBACK_PENALTY = 0`, then restores the original value.

# %%
_saved_penalty = CHARGEBACK_PENALTY
sens_rows = []
for penalty in (15.0, 5.0, 0.0):
    CHARGEBACK_PENALTY = penalty
    for amt, prod in [(4.99, "consumable"), (19.99, "consumable"), (49.99, "consumable"),
                      (49.99, "non_consumable")]:
        c_approve = cost("approve", "H2_stolen_card", amt, prod)
        c_stop = cost("stop", "H1_legit", amt, prod)
        sens_rows.append({
            "penalty": penalty, "amount": amt, "product": prod,
            "wrong approve (H2)": round(c_approve, 2),
            "wrong stop (H1)": round(c_stop, 2),
            "cost ratio": round(c_approve / c_stop, 2),
            "gate": describe_gate(amt, prod),
        })
sensitivity = pd.DataFrame(sens_rows)
print(sensitivity.to_string(index=False))

# Re-run every policy end-to-end under the zero-penalty model.
CHARGEBACK_PENALTY = 0.0
alt = results.copy()
alt["action_P3"] = alt.apply(policy_P3, axis=1)
alt_totals = {}
for pol in ["B0", "P1", "P2", "P3"]:
    alt_totals[pol] = round(float(alt.apply(
        lambda r: cost(r[f"action_{pol}"], r["true_state"], r["amount_usd"], r["product_type"]),
        axis=1).sum()), 2)
print("\nTotal decision cost with CHARGEBACK_PENALTY = 0:", alt_totals)
print("Actions P3 takes under the zero-penalty model:", dict(alt["action_P3"].value_counts()))
print("Cases where P3's action changes:", int((alt["action_P3"] != results["action_P3"]).sum()))

CHARGEBACK_PENALTY = _saved_penalty   # restore for everything downstream
assert CHARGEBACK_PENALTY == 15.0

# %% [markdown]
# ### DC-16 result — the gate retreats but does not vanish, and the ranking does not move
#
# **The threshold is highly sensitive to the contested number.** Removing the penalty
# moves the stop boundary for the canonical $19.99 consumable from 71.5% to **81.4%**,
# for a $4.99 consumable from 72.9% to **91.6%**, and for a $49.99 non-consumable to
# **91.2%**. Blocking survives only as a near-certainty action. For a $4.99 consumable the
# cost ratio inverts outright — 0.40, meaning a wrong denial costs *more than twice* a
# wrong approval — so on small purchases the arithmetic forbids blocking almost regardless
# of belief. That is the quantitative form of "too much UX risk" (DC-12).
#
# **The corner survives exactly where the practitioner conceded it might.** At zero
# penalty the *only* cell still carrying a meaningfully adverse ratio is the large
# consumable: 1.43, gate at **75.7%** — the lowest threshold in the zero-penalty block.
# Large consumables are the last place where blocking can pay, because they are the only
# combination of high value and zero recoverability. The corner named in the discussion is
# real in the arithmetic, and it is the *only* one.
#
# **The policy ranking is insensitive to the contested number.** Under the zero-penalty
# model the totals become B0 131.96, P1 92.88, P2 88.47, **P3 80.97** — the same order,
# with P3 still cheapest and the derived-threshold policy still beating both eye-tuned
# variants. Exactly one case out of 50 changes action. So the disputed $15 changes *where
# the gate sits*, not *which policy wins*: the conclusion "derive thresholds from costs,
# and gate almost nothing" holds under both cost models, which is the more useful result
# to report than either model alone.


# %% [markdown]
# ## 6. Metrics
#
# Accuracy alone is banned by the assignment; we report action×state confusion tables,
# precision/recall of intervention against fraud, FP/FN counts, human-review rate,
# total decision cost, and calibration of P(hostile).

# %%
def confusion(pol):
    t = pd.crosstab(results[f"action_{pol}"], results["true_state"])
    return t.reindex(index=["approve", "question", "examine", "stop"], columns=STATES, fill_value=0)

for pol in ["B0", "P1", "P2", "P3"]:
    print(f"\n=== {pol} — action × true state ===")
    print(confusion(pol))

# %%
summary = []
for pol in ["B0", "P1", "P2", "P3"]:
    act = results[f"action_{pol}"]
    fraud = results["true_state"] != "H1_legit"
    hostile = results["true_state"].isin(HOSTILE)
    intervened = act != "approve"
    # question does not actually stop H4/H5 (they pass the challenge) → effective miss
    effective_block = intervened & ~(act.eq("question") & results["true_state"].isin(["H4_friendly", "H5_refund_abuse"]))
    tp = int((intervened & fraud).sum())
    fp = int((intervened & ~fraud).sum())
    fn = int((~intervened & fraud).sum())
    eff_fn = int((fraud & ~effective_block).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    summary.append({
        "policy": pol,
        "FP (legit intervened)": fp,
        "FN (fraud approved)": fn,
        "effective FN (fraud not blocked)": eff_fn,
        "precision (intervention)": round(prec, 3) if prec == prec else None,
        "recall (fraud)": round(rec, 3),
        "hostile recall": round(float((intervened & hostile).sum() / hostile.sum()), 3),
        "human-review rate": round(float((act == "examine").mean()), 3),
        "total cost": round(float(results[f"cost_{pol}"].sum()), 2),
        "mean cost/case": round(float(results[f"cost_{pol}"].mean()), 3),
    })
summary_df = pd.DataFrame(summary).set_index("policy")
summary_df

# %%
# Calibration of P(hostile): bins aligned with the policy bands + Brier score.
bins = [0, 0.05, 0.25, 0.60, 1.0]
lab = ["<5%", "5–25%", "25–60%", ">60%"]
cal = results.assign(bin=pd.cut(results["P_hostile"], bins, labels=lab, include_lowest=True),
                     hostile=results["true_state"].isin(HOSTILE).astype(int))
cal_table = cal.groupby("bin", observed=False).agg(
    n=("hostile", "size"),
    mean_predicted=("P_hostile", "mean"),
    empirical_hostile_rate=("hostile", "mean"),
).round(3)
brier = float(((cal["P_hostile"] - cal["hostile"]) ** 2).mean())
print(cal_table)
print(f"\nBrier score (hostile event): {brier:.4f}")
print("Caveat: with 50 cases and 2 hostile cases, per-bin rates are noisy — reported for method, not significance.")

# %% [markdown]
# ## 7. Second decision surface — the consumption response (DC-2, DC-3, DC-9)
#
# For approved cases that later trigger a CONSUMPTION_REQUEST, the agent decides exactly
# one judgment field, `refundPreference` (DC-3), via three practitioner-validated rules
# (DC-9): near-zero consumption ⇒ prefer grant (never fight — a one-star review costs
# more than a small refund, DC-10); flat posterior ⇒ deliberately abstain (the reject
# option); reason = UNINTENDED_PURCHASE ⇒ raise the decline threshold (usually
# somebody's kid, maps to H4). If stored consent is absent, no consumption payload can
# be sent at all (DC-5). The output is advisory — Apple decides regardless (DC-11).

# %%
P_REFUND_REQ = {"H1_legit": 0.04, "H2_stolen_card": 0.5, "H3_ato": 0.5, "H4_friendly": 0.85, "H5_refund_abuse": 0.95}
P_REASON_UNINTENDED = {"H1_legit": 0.10, "H2_stolen_card": 0.05, "H3_ato": 0.05, "H4_friendly": 0.85, "H5_refund_abuse": 0.05}

def refund_stage(r):
    if r[f"action_P2"] != "approve":
        return pd.Series({"refund_request": False})
    s = r["true_state"]
    if rng.random() >= P_REFUND_REQ[s]:
        return pd.Series({"refund_request": False})
    reason = "UNINTENDED_PURCHASE" if rng.random() < P_REASON_UNINTENDED[s] else "UNSATISFIED_WITH_PURCHASE"
    consumed = {"H1_legit": 0.10 * rng.random(),
                "H2_stolen_card": rng.random(),
                "H3_ato": rng.random(),
                "H4_friendly": 0.6 * rng.random(),
                "H5_refund_abuse": 0.7 + 0.3 * rng.random()}[s]
    consent = bool(rng.random() < 0.85)   # [ASSUMED] consent capture rate (DC-5)

    # refund-time belief: grant-time posterior updated with the reason evidence (DC-1)
    p = np.array([r[f"P_{st}"] for st in STATES])
    lik = np.array([P_REASON_UNINTENDED[st] if reason == "UNINTENDED_PURCHASE"
                    else 1 - P_REASON_UNINTENDED[st] for st in STATES])
    p = p * lik
    p = p / p.sum()
    p_abuse = p[STATES.index("H5_refund_abuse")] + p[1] + p[2]

    if not consent:
        pref = "NO_PAYLOAD (no stored consent — DC-5)"
    elif consumed < 0.05:
        pref = "PREFER_GRANT (near-zero consumption, DC-9.1)"
    elif p.max() < 0.50:
        pref = "NO_PREFERENCE (flat posterior — deliberate abstention, DC-9.2)"
    else:
        decline_thr = 0.75 if reason == "UNINTENDED_PURCHASE" else 0.50   # DC-9.3
        if p_abuse > decline_thr:
            pref = "PREFER_DECLINE"
        elif p[0] + p[3] > 0.60:
            pref = "PREFER_GRANT"
        else:
            pref = "NO_PREFERENCE (flat posterior — deliberate abstention, DC-9.2)"
    return pd.Series({"refund_request": True, "cr_reason": reason,
                      "consumed_pct": round(consumed, 2), "consent": consent,
                      "refund_preference": pref})

stage2 = results.apply(refund_stage, axis=1)
results = pd.concat([results, stage2], axis=1)
cols = ["case_id", "true_state", "cr_reason", "consumed_pct", "consent", "refund_preference"]
results.loc[results["refund_request"] == True, cols]

# %%
results.to_csv(RESULTS_DIR / "results.csv", index=False)
print(f"Saved results/results.csv with {len(results)} cases; all predictions and actions for B0/P1/P2/P3 recorded.")

# %% [markdown]
# ## 7b. Seed sweep — is the ranking a result, or one lucky draw?
#
# The headline comparison rests on 50 cases at a single seed, of which only two are
# hostile. A program-committee review called this out: re-draw the seed and the margin
# moves, so a point estimate on $n=2$ is not evidence of a ranking. This section runs the
# whole pipeline over 300 seeds and reports the *distribution* of the margin instead.
#
# It also answers a second question the single-seed run could not: seed 17 happened to
# draw all five non-legitimate cases as consumables, which is exactly what the product-split
# conclusion needs in order to be true. Over 300 seeds we can see how often a hostile case
# is generated on a *revocable* product, and what that does to the comparison.

# %%
def run_seed(seed: int) -> dict:
    """Re-run generation, belief update, policies and scoring for one seed."""
    r = np.random.default_rng(seed)
    rows = []
    for state, n in COMPOSITION.items():
        si = STATES.index(state)
        for _ in range(n):
            row = {"true_state": state}
            for var, spec in LIK.items():
                row[var] = r.choice(spec["values"], p=spec["p"][si])
            rows.append(row)
    cs = pd.DataFrame(rows)
    cs["amount_usd"] = cs["amount"].map(AMOUNT_VALUE)

    post = np.vstack([posterior(row) for _, row in cs.iterrows()])
    df = cs[["true_state", "amount_usd", "product_type"]].copy()
    for i, s in enumerate(STATES):
        df[f"P_{s}"] = post[:, i]
    df["P_hostile"] = post[:, 1] + post[:, 2]
    df["account_risk"] = ((cs["account_age"] == "new") | (cs["prior_purchases"] == "none")
                          | (cs["refund_history"] == "yes"))

    df["action_B0"] = "approve"
    df["action_P1"] = df.apply(policy_P1, axis=1)
    df["action_P2"] = df.apply(policy_P2, axis=1)
    df["action_P3"] = df.apply(policy_P3, axis=1)

    out = {"seed": seed}
    for pol in ("B0", "P1", "P2", "P3"):
        out[pol] = float(df.apply(lambda x: cost(x[f"action_{pol}"], x["true_state"],
                                                 x["amount_usd"], x["product_type"]), axis=1).sum())
    hostile = df[df.true_state.isin(HOSTILE)]
    nonleg = df[df.true_state != "H1_legit"]
    out["hostile_revocable"] = int((hostile.product_type != "consumable").sum())
    out["nonleg_revocable"] = int((nonleg.product_type != "consumable").sum())
    out["fp_P3"] = int(((df.action_P3 != "approve") & (df.true_state == "H1_legit")).sum())
    return out

sweep_seeds = list(range(1, 301))
sw = pd.DataFrame([run_seed(s) for s in sweep_seeds])
sw["margin_B0_P3"] = sw["B0"] - sw["P3"]
sw["margin_P2_P3"] = sw["P2"] - sw["P3"]

print("Total decision cost across 300 seeds:")
print(sw[["B0", "P1", "P2", "P3"]].describe().loc[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(2))

print("\nP3 vs the reactive baseline (B0 - P3):")
m = sw["margin_B0_P3"]
print(f"  mean {m.mean():.2f}   median {m.median():.2f}   sd {m.std():.2f}"
      f"   5th pct {m.quantile(0.05):.2f}   95th pct {m.quantile(0.95):.2f}")
print(f"  P3 cheaper than B0 in {(m > 0).mean():.1%} of seeds; seed 17 margin was 73.24")

print("\nRanking stability (share of seeds):")
print(f"  P3 <= every other policy : {(sw[['B0','P1','P2']].min(axis=1) >= sw['P3']).mean():.1%}")
print(f"  P3 <= P1 and P2          : {((sw.P1 >= sw.P3) & (sw.P2 >= sw.P3)).mean():.1%}")
print(f"  P3 strictly worst        : {(sw[['B0','P1','P2']].max(axis=1) < sw['P3']).mean():.1%}")

print("\nThe sampling artefact the review identified:")
print(f"  seeds where every non-legitimate case is a consumable: {(sw.nonleg_revocable == 0).mean():.1%}")
print(f"  seeds with at least one hostile case on a revocable product: {(sw.hostile_revocable > 0).mean():.1%}")
rev = sw[sw.hostile_revocable > 0]
print(f"  mean B0-P3 margin, seeds with a revocable hostile case: {rev.margin_B0_P3.mean():.2f}"
      f"  (vs {sw[sw.hostile_revocable == 0].margin_B0_P3.mean():.2f} without)")

print("\nPairwise win rates:")
print(f"  P3 <= P1: {(sw.P3 <= sw.P1).mean():.1%}    P3 <= P2: {(sw.P3 <= sw.P2).mean():.1%}"
      f"    P2 <= P1: {(sw.P2 <= sw.P1).mean():.1%}")
print(f"  mean cost of the DC-6 hard rule (P2 - P1): {(sw.P2 - sw.P1).mean():+.2f}")
print(f"  seed 17's margin sits at the {(sw.margin_B0_P3 < 73.24).mean():.0%} percentile of the distribution")

# %% [markdown]
# ### What the sweep changes
#
# **The headline survives; the fine ranking does not.** P3 is cheaper than the reactive
# baseline in **99.3%** of 300 seeds (mean margin 79.53, 5th–95th percentile 21.75–141.00),
# and seed 17's margin of 73.24 sits at the 45th percentile — close to typical, not
# cherry-picked. But P3 is the cheapest of all four policies in only **61.0%** of seeds. The
# single-seed claim "the cost-derived policy is cheapest" should therefore be stated as a
# tendency with a win rate, not as a property.
#
# **The DC-6 hard rule costs money on average, and the single-seed run hid it.** At seed 17,
# P2 (92.22) edged P1 (92.88), so the paper reported the practitioner-derived constraint as
# a small saving. Over 300 seeds the ordering reverses: mean P1 77.74 against P2 83.87, so
# the rule costs **+6.13 units on average**, and P2 beats P1 in only 49.0% of seeds. This is
# the honest version of a result the paper had backwards. The constraint is still defensible
# — it exists to protect legitimate users from being escalated on device evidence alone,
# which is a fairness argument, not a cost argument — but it should be reported as a price
# paid deliberately, not as a free improvement.
#
# **The product-split conclusion stops being a sampler artefact and becomes a finding.**
# Seed 17 drew all five non-legitimate cases as consumables, which is exactly the condition
# the claim needs; that draw occurs in only **14.3%** of seeds, so the single-seed evidence
# was close to circular. Across all 300 seeds, **54.7%** contain at least one hostile case
# on a revocable product, and in those seeds the mean margin over the baseline falls to
# **68.95** against **92.28** where every hostile case is a consumable. Gating is worth
# about a quarter less when the product can be revoked after the fact — which is the
# direction DC-7 and DC-12 predicted, now measured over cases that actually contain
# revocable hostile purchases rather than asserted from a batch that never generated one.

# %% [markdown]
# ## 8. Failure examination — six named failures, each addressed to a layer
#
# Generated from the actual run (seed 17). Each failure is tagged with the architecture
# layer that was actually wrong, so a repair has an address instead of a vibe: L0 costs,
# L1 possible worlds, L2 what is usual here, L3 answer shape, L4 the action rule.
#
# **F1 — The invisible abuser (C034, H5, approved by all four policies). Layer: L4.**
# P(hostile)=3.0%, P(H5)=8.1%. The policy's event is P(H2)+P(H3), so refund abuse with an
# unremarkable short history crosses no boundary at all — not the eye-tuned bands and not
# the cost-derived ones, because both act on the same event. *Failure condition:
# hostile-only event blindness — a threshold cannot see the state it was not pointed at.*
# The event, not the threshold, is the defect. Fix: a second decision variable on P(H5),
# or accept the grant-time miss and catch it at the consumption surface (see F6).
#
# **F2 — The futile challenge (C009, H4 questioned; cost 36.0 vs 32.0 for approving).
# Layer: L0.** Old account, many purchases, but a new device lifted P(hostile) to 16% →
# `question`. The family member is a real cardholder and passes any challenge, so the
# policy paid the full approve-loss *plus* friction. The cost matrix does encode this
# (question against H4/H5 = approve cost + friction), which is why cost-derived P3 still
# chose question here: the belief mass on H4 was not large enough for the correct answer
# to win. *Failure condition: challenge-proof risk mass.* Fix at L0/L4: a no-question
# override on P(H4)+P(H5), not only P(H5)>30%.
#
# **F3 — The clean-account takeover (C045, H3; P1 examined, P2 and P3 both capped to
# question). Layer: L4 (hard rule).** A mid-age account with many purchases carries no
# account-level risk flag, so the DC-6 gate suppressed escalation — exactly the blind
# spot predicted when the constraint was adopted, since takeovers of healthy accounts
# look clean at account level. The evidence planner (§5c) makes the cost of that gate
# legible: at C045's belief the top-ranked purchasable check is `email_changed`
# (0.33 bits), the very check that resolved the case in the decision record — and the
# gate forbids acting on what it would reveal. *Failure condition: evidence-hierarchy
# blind spot.* Fix: exempt high-P(H3) cases, or promote "new device + established
# account" to an account-level flag.
#
# **F4 — Friction on the faithful (C039, legit, questioned by every policy). Layer: L2.**
# A direct-to-store session with no purchase history pushed P(hostile) to 7.5%, over both
# the eye-tuned 5% and the derived 8.2%... marginally. The cost is small (4.00) but this
# is the boundary where naive-Bayes double-counting of correlated device/session evidence
# inflates posteriors. The defect is in the belief, not the threshold.
# *Failure condition: correlated-evidence inflation at the approve/question boundary.*
#
# **F5 — The night-owl gift (C050, H4, $4.99 consumable) — FIXED by cost-derived
# thresholds. Layer: L4.** P1/P2 challenged this purchase because 5.2% > their 5% band;
# challenging a $4.99 purchase can never pay for itself, and P3 approves it (6.74 → 4.99).
# C040 (legit, $4.99 non-consumable) is the same repair (1.75 → 0.00). *Failure
# condition: amount-blind thresholds.* This failure is retained in the record because it
# demonstrates the repair: a boundary derived from the cost ratio moves with the stake,
# an eye-tuned band does not.
#
# **F6 — Belief-blind grant at the refund surface (C034 again, stage 2). Layers: L1/L3.**
# The same H5 case later filed a refund request with **99% consumed** — the literal
# consume-then-refund signature — and the agent sent PREFER_GRANT, because the refund-time
# update uses only `consumptionRequestReason`; consumption magnitude is read solely by the
# near-zero rule and never given a likelihood. *Failure condition: an observed quantity
# with no shape and no place in the world model.* Fix: add P(consumed_pct | state) at the
# refund-time update. Discovered by the simulation, not anticipated.

# %%
fraud = results["true_state"] != "H1_legit"
fails = results[
    (fraud & results["action_P3"].eq("approve"))                                            # F1: fraud approved
    | (fraud & results["action_P3"].eq("question")
             & results["true_state"].isin(["H4_friendly", "H5_refund_abuse"]))              # F2: futile challenge
    | (fraud & (results["action_P1"] != results["action_P3"]))                              # F3: gate downgrade
    | (~fraud & (results["action_P1"].ne("approve")                                         # F4/F5: friction on legit
                 | results["action_P2"].ne("approve")
                 | results["action_P3"].ne("approve")))
].copy()
show = ["case_id", "true_state", "P_hostile", "P_H5_refund_abuse", "account_risk",
        "action_P1", "action_P2", "action_P3", "cost_B0", "cost_P1", "cost_P2", "cost_P3"]
detail = fails[show].merge(cases[["case_id"] + EVIDENCE_VARS], on="case_id")
print(f"{len(detail)} incorrect or degraded decisions examined:")
detail

# %% [markdown]
# ## 9. Findings
#
# 1. **Every deliberating policy beats the reactive baseline on total decision cost —
#    161.96 (B0) vs 92.88 (P1) vs 92.22 (P2) vs 88.72 (P3) — but the entire margin comes
#    from the two hostile cases.** In a batch with no stolen-card/ATO cases the baseline
#    wins outright, because it never pays friction or review costs. This confirms the
#    pre-registered expectation that "always approve" is hard to beat at a 92% legitimate
#    prior, and it is the argument for cost-sensitive evaluation: accuracy would score B0
#    at 90% while it silently eats every fraud loss.
# 2. **Deriving the thresholds from the cost matrix strictly dominates the eye-tuned
#    bands.** P3 is cheapest (88.72), has the fewest false positives (1 vs 2), the best
#    intervention precision (0.75 vs 0.67), the same hostile recall (1.0) and the same
#    human-review rate (2%). It repairs both amount-blind failures — C050 and C040, small
#    purchases that P1/P2 challenged for less than the challenge cost. The derived
#    boundaries for a $19.99 consumable are **approve below 8.2%, question up to 71.5%,
#    stop above** — against eye-tuned 5% / 25% / 60%. The eye-tuned bands were roughly
#    right at the bottom and badly wrong at the top: they stop at 60% where the cost ratio
#    justifies stopping only past 71.5%.
# 3. **The cost ratio here is ~1.75:1, not 2000:1 — which is the quantitative case for
#    the practitioners' "don't gate" advice (DC-12).** A wrong approve on a $19.99
#    consumable costs 34.99 (item + chargeback penalty); a wrong stop costs 19.99 (half
#    the lost sale + reputational term). Compare a payment-fraud setting where the ratio
#    justifies a 0.05% threshold. Low-value digital goods simply do not support aggressive
#    blocking, and the arithmetic says so before any practitioner does.
# 3b. **DC-16 sensitivity: the disputed cost changes the threshold, not the winner.**
#    A practitioner disputes the flat chargeback penalty ("your worst case is capped at
#    the revenue anyway, since Apple is the merchant of record"). Setting it to zero moves
#    the stop gate from 71.5% to 81.4% ($19.99 consumable) and to 91.6% ($4.99), and
#    inverts the small-purchase ratio to 0.40 — a wrong denial then costs more than twice a
#    wrong approval. But the policy ranking is unchanged (B0 131.96 > P1 92.88 > P2 88.47 >
#    P3 80.97) and exactly one of 50 actions differs. The one cell that keeps an adverse
#    ratio at zero penalty is the **large consumable** (1.43, gate at 75.7%) — the corner
#    the practitioner conceded might exist, and the only one the arithmetic supports.
# 4. **Value of information rarely justifies a human review.** With review priced at 9.40
#    (analyst + delay), VoI exceeded the fee in exactly one of 50 cases. At C045 — the
#    genuinely ambiguous takeover — VoI was 7.52 against a 9.40 fee: *the agent should not
#    buy the review*, even though the case is exactly the kind a human would want to see.
#    That is an uncomfortable, honest result: review is a luxury at these stakes, and any
#    "escalate when unsure" instinct must be priced before it is trusted.
# 5. **The most informative cheap signals are the ones practitioners forbid acting on.**
#    From the priors, `session_type` and `device_age` rank highest (0.095 bits each) —
#    and DC-6 exists precisely to stop the agent escalating on them. The planner
#    quantifies the tension the discussions produced: informative ≠ actionable.
# 6. **The planner independently selects the check the decision record used.** At C045's
#    belief (1.84 bits, nothing above 38%), `email_changed` ranks first at 0.33 bits —
#    the same evidence that drove the posterior to 93.4% ATO in
#    `decisions/probability-decision-record.md`. The zero-gain control (`coin_flip`)
#    scores exactly 0.000, as it must.
# 7. **Nominal recall overstates protection; effective blocking is the honest metric.**
#    Three of five fraud cases were not actually blocked under any policy (effective FN =
#    3): two H4s pass their challenge, one H5 was approved. `question` protects only
#    against H2/H3.
# 8. **The consumption surface needs consumption magnitude as evidence** (F6): the one H5
#    case sailed through both surfaces despite a 99%-consumed refund request.
# 9. **Calibration is monotone in this run** (mean predicted 0.5% / 8.6% / 39.8% vs
#    empirical 0% / 0% / 100% hostile per band, Brier 0.0155) — the method is in place,
#    but with two hostile cases this demonstrates procedure, not calibration quality.
#
# **Highest-cost error: approving hostile fraud on a large consumable** (realized by
# B0 on C046, a stolen-card large consumable: 64.99 units — the worst single-case cost
# in the run). It is the highest because it is *irreversible* (a consumable cannot be
# revoked, unlike a non-consumable or subscription — DC-7), it carries platform
# penalties beyond the item value (chargeback fee, account-health standing), and in
# production stolen-card fraud arrives in bursts, so one miss predicts more. The
# runner-up — stopping a legitimate whale — did not occur in this run but is bounded
# by DC-6 and the reputational term (DC-10) in the cost matrix.
#
# **Queued for v0.3:** challenge-proof-mass override P(H4)+P(H5) (F2); a second decision
# variable on P(H5) (F1); `consumed_pct` likelihood at the refund-time update (F6); DC-6
# exemption for high-P(H3) (F3); an imperfect-analyst model so VoI stops assuming review
# resolves the state completely; sequential evidence purchase driven by the §5c planner
# rather than receiving all eight variables at once.
#
# ## Limitations
#
# 1. **Self-simulated data:** evidence is generated from the same likelihood tables the
#    agent uses, so the belief update is exactly correctly specified — real posteriors
#    would be worse. The simulation validates the decision machinery, not detection power.
# 2. **[ASSUMED] everything numeric:** priors (92/2/2/3/1), likelihoods, and costs are
#    design assumptions; no public IAP-fraud base rates were found.
# 3. **Correlated evidence double-counting:** naive Bayes treats device/session/hour as
#    independent; they are not. Known before the run; visible in over-confident posteriors.
# 4. **Selective labels:** in production, blocked-but-legit is never observed and
#    chargebacks reveal false negatives weeks late. The simulator sees all labels; the
#    field never does. Evaluation design must account for this.
# 5. **Single-transaction scope:** the baseline's reactive ban prevents *repeat* abuse,
#    which a per-transaction simulation cannot credit. The baseline is stronger in
#    reality than it is here.
# 6. **Small N:** 50 cases, 5 fraud cases. Metrics are demonstrations of method, not
#    statistically significant comparisons.
# 7. **Perfect-analyst assumption (v0.2):** `examine` is priced as "pay the fee, learn the
#    true state, act optimally". A real reviewer is imperfect, slow, and queued, so this
#    *overstates* the value of a human review — and even so, VoI justified review in only
#    one of 50 cases. An imperfect-analyst model would make review rarer still. The
#    v0.1 hand-set `examine` cost row was replaced by this derivation, so P1/P2 totals
#    differ from the v0.1 run (99.47/96.72 → 92.88/92.22); the ranking is unchanged.
# 8. **No queue state and no sequential evidence purchase.** The agent receives all eight
#    evidence variables at once; the §5c planner ranks checks but does not yet drive
#    acquisition, and review capacity/delay are not modelled as system state.
