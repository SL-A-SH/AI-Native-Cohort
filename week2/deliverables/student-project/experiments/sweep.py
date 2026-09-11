"""Two sweeps, because a single run is not a result.

    python experiments/sweep.py --mode seeds     # is the comparison stable?
    python experiments/sweep.py --mode dither    # what does the dither weight actually buy?

**Why the seed sweep exists.** On the previous agent a single run showed one design saving
money; three hundred runs showed it *costing* 6.13 units on average and winning only 49% of the
time. The first number was not a lie, it was one sample. Nothing from `run.py` goes in the paper
until it has been reproduced across case-generation seeds, and the spread is reported next to
the mean rather than hidden behind it.

**Why the dither sweep exists.** The forty-case run showed the belief agent switching targets
roughly twice as often as the baselines, which is the exact failure the dither term was added to
prevent. The weight is `[ASSUMED] 4.0`, and the tempting move is to raise it until the number
looks respectable. That is precisely the thing the day 3 post is about, so instead the whole
range is run and reported, including the values that make the agent look worse. If a higher
weight genuinely helps, the sweep says so and the change is evidence-led; if it only trades one
failure for another, the sweep says that too.

Writes `results/sweep-seeds.csv` and `results/sweep-dither.csv`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from agent import COSTS                        # noqa: E402
from run import RESULTS, build_cases, confusion, run_episode   # noqa: E402

ARMS = [
    ("belief", False),
    ("belief", True),
    ("last_known_position", False),
    ("scripted_sweep", False),
    ("degraded_omniscience", False),
]

GEN_SEEDS = [20260910, 11, 2027, 44497, 86243, 110503, 132049, 216091]
DITHER_WEIGHTS = [0.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def one_run(cases, kind: str, poi: bool, costs: dict | None = None) -> dict:
    """One arm over one set of cases. Returns the summary numbers, nothing written to disk."""
    rows, summaries = [], []
    for case in cases:
        r, s = run_episode(case, kind, poi, costs)
        rows.extend(r)
        summaries.append(s)
    cm = confusion(rows)
    attrs = [s["mean_attribution"] for s in summaries if s["mean_attribution"] != ""]
    return {
        "capture": float(np.mean([s["captured"] for s in summaries])),
        "ticks": float(np.mean([s["ticks"] for s in summaries])),
        "attribution": float(np.mean(attrs)) if attrs else float("nan"),
        "redundant": sum(s["redundant_searches"] for s in summaries),
        "switches": sum(s["target_switches"] for s in summaries),
        "low_attribution": sum(s["low_attribution_actions"] for s in summaries),
        "precision": cm["precision"],
        "recall": cm["recall"],
    }


def sweep_seeds(n_cases: int) -> None:
    """The same comparison, rebuilt from scratch under eight different case-generation seeds."""
    per_arm: dict[str, list[dict]] = {}
    rows_out = []

    for gen_seed in GEN_SEEDS:
        cases = build_cases(n_cases, gen_seed=gen_seed)
        for kind, poi in ARMS:
            label = f"{kind}{'_B' if poi else ''}"
            result = one_run(cases, kind, poi)
            per_arm.setdefault(label, []).append(result)
            rows_out.append({"gen_seed": gen_seed, "arm": label,
                             **{k: round(v, 4) if isinstance(v, float) else v
                                for k, v in result.items()}})

    with open(RESULTS / "sweep-seeds.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"{len(GEN_SEEDS)} generation seeds x {n_cases} cases x {len(ARMS)} arms")
    print(f"wrote {RESULTS / 'sweep-seeds.csv'}\n")

    head = (f"{'arm':24s} {'capture':>16s} {'redundant':>14s} "
            f"{'switches':>14s} {'attribution':>14s}")
    print(head)
    print("-" * len(head))
    for label, runs in per_arm.items():
        def ms(key):
            vals = [r[key] for r in runs]
            return float(np.mean(vals)), float(np.std(vals))
        cap, cap_sd = ms("capture")
        red, red_sd = ms("redundant")
        swi, swi_sd = ms("switches")
        att, att_sd = ms("attribution")
        print(f"{label:24s} {cap:7.1%} +/-{cap_sd:5.1%} {red:8.1f} +/-{red_sd:4.1f} "
              f"{swi:8.1f} +/-{swi_sd:4.1f} {att:8.2f} +/-{att_sd:4.2f}")

    print("\nBelief agent vs each baseline, per seed, on capture rate:")
    base_labels = [f"{k}{'_B' if p else ''}" for k, p in ARMS if k != "belief"]
    for other in base_labels:
        wins = sum(1 for a, b in zip(per_arm["belief"], per_arm[other])
                   if a["capture"] > b["capture"])
        deltas = [a["capture"] - b["capture"]
                  for a, b in zip(per_arm["belief"], per_arm[other])]
        print(f"  vs {other:26s} belief ahead in {wins}/{len(GEN_SEEDS)} seeds, "
              f"mean gap {np.mean(deltas):+.1%}")


def sweep_dither(n_cases: int) -> None:
    """The believability weight the day 3 post refused to tune, run across its whole range.

    **Across every generation seed, not one.** The first version of this called `build_cases`
    without a seed, so the whole sweep was a single set of 40 cases: every capture figure came
    out a multiple of 2.5%, which is 1 in 40, and a practitioner review spotted exactly that.
    Reporting a parameter's effect from one seed is the failure lesson 6 exists to prevent, and
    it had been avoided in the seed sweep two functions above while being committed here.
    """
    rows_out = []

    print(f"dither weight sweep, {len(GEN_SEEDS)} seeds x {n_cases} cases, "
          f"belief agent (policy A)")
    head = (f"{'dither':>7s} {'capture':>16s} {'switches':>15s} {'redundant':>14s} "
            f"{'attribution':>12s}")
    print(head)
    print("-" * len(head))

    for weight in DITHER_WEIGHTS:
        costs = dict(COSTS)
        costs["dither"] = weight
        runs = []
        for gen_seed in GEN_SEEDS:
            cases = build_cases(n_cases, gen_seed=gen_seed)
            runs.append(one_run(cases, "belief", False, costs))

        summary = {"dither": weight}
        for key in ("capture", "switches", "redundant", "attribution", "ticks"):
            vals = [r[key] for r in runs]
            summary[key] = round(float(np.mean(vals)), 4)
            summary[f"{key}_sd"] = round(float(np.std(vals)), 4)
        rows_out.append(summary)

        print(f"{weight:7.1f} {summary['capture']:9.1%} +/-{summary['capture_sd']:5.1%} "
              f"{summary['switches']:8.1f} +/-{summary['switches_sd']:5.1f} "
              f"{summary['redundant']:7.1f} +/-{summary['redundant_sd']:5.1f} "
              f"{summary['attribution']:12.2f}")

    with open(RESULTS / "sweep-dither.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nwrote {RESULTS / 'sweep-dither.csv'}")

    registered = [r for r in rows_out if r["dither"] == COSTS["dither"]][0]
    best_switch = min(rows_out, key=lambda r: r["switches"])
    print(f"\nregistered weight {COSTS['dither']}: "
          f"{registered['switches']} switches, {registered['capture']:.1%} capture")
    print(f"fewest switches at {best_switch['dither']}: "
          f"{best_switch['switches']} switches, {best_switch['capture']:.1%} capture")
    print("Report both. Moving the weight to whichever row flatters the agent is the exact\n"
          "failure the day 3 post is about.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["seeds", "dither"], required=True)
    parser.add_argument("--cases", type=int, default=40)
    args = parser.parse_args()

    if args.mode == "seeds":
        sweep_seeds(args.cases)
    else:
        sweep_dither(args.cases)


if __name__ == "__main__":
    main()
