"""Failure analysis, calibration and escalation rates, per §9.

    python experiments/analyse.py

Reads `results/decisions.csv` and `results/episodes.csv` and prints everything the paper's
Failure Analysis and Results sections need. Writes `results/failures.csv`, which holds the
individual bad decisions named in the write-up so a reader can find each one by case and tick
without re-running anything.

Nothing here re-simulates. It reads the recorded decisions, so the numbers cannot drift from
what the agents actually did.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results"

import world  # noqa: E402

COMMITTED = ("search", "flank", "call_reinforcements")


def load(name: str) -> list[dict]:
    with open(RESULTS / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(value: str, default=float("nan")) -> float:
    return float(value) if value not in ("", None) else default


def cell(row: dict, prefix: str):
    r, c = row.get(f"{prefix}_r", ""), row.get(f"{prefix}_c", "")
    return (int(r), int(c)) if r != "" and c != "" else None


def rule(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def confusion_by_arm(rows: list[dict]) -> None:
    """The two error types, per arm.

    A false positive is committing while the player is not findable at the target: pressing on
    when they are long gone, which reads as flailing. A false negative is standing down while
    they *are* findable: the agent had them and let go, which reads as stupid.
    """
    rule("Confusion matrix by arm (positive = committed to a pursuit action)")
    print(f"{'arm':26s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'TN':>6s} "
          f"{'prec':>6s} {'rec':>6s} {'FP rate':>8s}")
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[f"{r['agent']}|{r['policy']}"].append(r)

    for arm, rs in by_arm.items():
        tp = sum(1 for r in rs if r["committed"] == "1" and r["findable"] == "1")
        fp = sum(1 for r in rs if r["committed"] == "1" and r["findable"] == "0")
        fn = sum(1 for r in rs if r["committed"] == "0" and r["findable"] == "1")
        tn = sum(1 for r in rs if r["committed"] == "0" and r["findable"] == "0")
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"{arm:26s} {tp:6d} {fp:6d} {fn:6d} {tn:6d} "
              f"{prec:6.2f} {rec:6.2f} {fp / len(rs):8.1%}")


def calibration(rows: list[dict]) -> None:
    """Is the belief agent's stated confidence worth anything?

    When it says a cell holds 12% of the probability, is the player there 12% of the time? Only
    the belief agent can be asked this: the baselines report no distribution, so their
    `peak_belief` column is empty by construction rather than by omission.

    Two readings, because an exact cell match on a 201 cell map is a hard test and a reader
    should see both. "Exact" is the honest calibration question. "Within 2" is the practical
    one, since an agent standing two cells away has effectively found you.
    """
    rule("Calibration of the belief agent's peak (policy A)")
    belief_rows = [r for r in rows
                   if r["agent"] == "belief" and r["policy"].startswith("A")
                   and r["peak_belief"] not in ("", "nan")]

    bins = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.01)]
    print(f"{'stated P(peak)':>16s} {'n':>6s} {'exact':>8s} {'within 2':>10s}")
    for low, high in bins:
        group = [r for r in belief_rows if low <= num(r["peak_belief"]) < high]
        if not group:
            continue
        exact = sum(1 for r in group if cell(r, "peak") == cell(r, "true"))
        near = 0
        for r in group:
            p, t = cell(r, "peak"), cell(r, "true")
            if p and t and world.distances_from(p)[t] <= 2:
                near += 1
        mid = np.mean([num(r["peak_belief"]) for r in group])
        print(f"  {low:.0%}-{high:.0%} (mean {mid:.1%}) {len(group):6d} "
              f"{exact / len(group):8.1%} {near / len(group):10.1%}")
    print("\nRead the 'exact' column against the stated probability. Where observed exceeds\n"
          "stated the agent is underconfident; where it falls short it is overconfident.")


def escalation(episodes: list[dict]) -> None:
    """Human-review rate, per §9, which for this agent is the escalation action.

    `call_reinforcements` is the analogue: it hands the problem to a second actor rather than
    resolving it alone. It is the only action with a fixed, large, once-per-episode price.
    """
    rule("Escalation rate (the human-review analogue)")
    by_arm = defaultdict(list)
    for e in episodes:
        by_arm[f"{e['agent']}|{e['policy']}"].append(e)
    for arm, es in by_arm.items():
        called = sum(1 for e in es if e["called_reinforcements"] == "1")
        gave_up = sum(1 for e in es if e["gave_up"] == "1")
        print(f"{arm:26s} escalated {called:3d}/{len(es)} ({called / len(es):5.1%})   "
              f"stood down {gave_up:3d}/{len(es)} ({gave_up / len(es):5.1%})")


def failure_conditions(rows: list[dict]) -> list[dict]:
    """Name the failure conditions and pull one concrete example of each.

    §9 asks for at least five incorrect decisions examined and each failure condition named.
    These are the conditions the data actually exhibits, not the ones predicted in advance, and
    two of them contradict predictions recorded in `research-file.md`.
    """
    rule("Named failure conditions, belief agent policy A")
    belief_rows = [r for r in rows
                   if r["agent"] == "belief" and r["policy"].startswith("A")]

    found: list[dict] = []

    def record(name: str, matches: list[dict], note: str) -> None:
        if not matches:
            print(f"{name:28s}      0 occurrences")
            return
        worst = matches[0]
        print(f"{name:28s} {len(matches):6d} occurrences   "
              f"e.g. case {worst['case']} tick {worst['tick']}")
        print(f"{'':28s}        {note}")
        for m in matches:
            found.append({"failure": name, "case": m["case"], "tick": m["tick"],
                          "agent": m["agent"], "policy": m["policy"],
                          "action": m["action"],
                          "target_r": m["target_r"], "target_c": m["target_c"],
                          "true_r": m["true_r"], "true_c": m["true_c"],
                          "attribution": m["attribution"],
                          "peak_belief": m["peak_belief"], "entropy": m["entropy"],
                          "reason": m["reason"]})

    # F1. Stood down while the player was in reach. The "looks stupid" error.
    record("F1 watching-not-catching",
           [r for r in belief_rows if r["committed"] == "0" and r["findable"] == "1"],
           "held or withdrew while the player was within reach of where it stood")

    # F2. Committed to a target the player was nowhere near. The "looks like flailing" error.
    record("F2 chasing-the-smear",
           [r for r in belief_rows if r["committed"] == "1" and r["findable"] == "0"],
           "walked to a diffused peak with the player nowhere near it")

    # F3. Acted on belief it could not account for.
    record("F3 illegible-commit",
           [r for r in belief_rows
            if r["committed"] == "1" and r["attribution"] not in ("", "nan")
            and num(r["attribution"]) < 1.0],
           "committed where the evidence ratio was below 1: unaccountable from outside")

    # F4. Escalated on a weak lead. Expensive and highly visible.
    record("F4 escalation-on-noise",
           [r for r in belief_rows
            if r["action"] == "call_reinforcements"
            and r["attribution"] not in ("", "nan") and num(r["attribution"]) < 2.0],
           "called reinforcements on a lead barely better than its own diffusion")

    # F5. Stood down for good while the player was still in reach.
    record("F5 premature-stand-down",
           [r for r in belief_rows if r["action"] == "resume_post" and r["findable"] == "1"],
           "took the terminal action with the player still findable")

    # F6. Confident and wrong: high stated belief, player elsewhere.
    record("F6 confident-and-wrong",
           [r for r in belief_rows
            if r["peak_belief"] not in ("", "nan") and num(r["peak_belief"]) > 0.15
            and cell(r, "peak") and cell(r, "true")
            and world.distances_from(cell(r, "peak"))[cell(r, "true")] > 8],
           "stated better than 15% on a cell more than 8 steps from the player")

    return found


def error_cost(rows: list[dict], episodes: list[dict]) -> None:
    """Which error costs most, and why. §9 asks for this explicitly.

    Costed two ways, because the answer differs depending on which one the designer cares
    about, and saying so is more useful than picking one.
    """
    rule("Which error costs most")
    belief_rows = [r for r in rows
                   if r["agent"] == "belief" and r["policy"].startswith("A")]
    eps = [e for e in episodes if e["agent"] == "belief" and e["policy"].startswith("A")]

    fp = [r for r in belief_rows if r["committed"] == "1" and r["findable"] == "0"]
    fn = [r for r in belief_rows if r["committed"] == "0" and r["findable"] == "1"]

    # In outcome terms: did episodes containing each error end in a capture?
    by_case = defaultdict(lambda: {"fp": 0, "fn": 0})
    for r in fp:
        by_case[r["case"]]["fp"] += 1
    for r in fn:
        by_case[r["case"]]["fn"] += 1

    cap = {e["case"]: e["captured"] == "1" for e in eps}
    fn_heavy = [c for c, v in by_case.items() if v["fn"] >= 3]
    fp_heavy = [c for c, v in by_case.items() if v["fp"] >= 20]
    print(f"episodes with 3+ false negatives: {len(fn_heavy):3d}, "
          f"capture rate {np.mean([cap.get(c, False) for c in fn_heavy]):.0%}"
          if fn_heavy else "episodes with 3+ false negatives: 0")
    print(f"episodes with 20+ false positives: {len(fp_heavy):3d}, "
          f"capture rate {np.mean([cap.get(c, False) for c in fp_heavy]):.0%}"
          if fp_heavy else "episodes with 20+ false positives: 0")

    # In believability terms, using the project's own proxy.
    fp_attr = [num(r["attribution"]) for r in fp if r["attribution"] not in ("", "nan")]
    print(f"\nmean attribution when committing wrongly (F2): "
          f"{np.mean(fp_attr):.2f}" if fp_attr else "")
    print(f"false positives per episode:  {len(fp) / len(eps):5.1f}")
    print(f"false negatives per episode:  {len(fn) / len(eps):5.1f}")


def main() -> None:
    rows = load("decisions.csv")
    episodes = load("episodes.csv")
    print(f"{len(rows)} decisions, {len(episodes)} episodes")

    confusion_by_arm(rows)
    calibration(rows)
    escalation(episodes)
    found = failure_conditions(rows)
    error_cost(rows, episodes)

    if found:
        with open(RESULTS / "failures.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(found[0].keys()))
            writer.writeheader()
            writer.writerows(found)
        print(f"\nwrote {RESULTS / 'failures.csv'} ({len(found)} rows)")
        print("Counts by condition:", dict(Counter(f["failure"] for f in found)))


if __name__ == "__main__":
    main()
