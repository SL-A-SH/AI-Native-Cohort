"""The 2x2 ablation the probability review asked for.

    python experiments/ablation.py

The belief agent's clearest result is that it re-searches cleared ground far less than any
baseline: 0.012 redundant searches per committed action against 0.160, 0.184 and 0.252. The
project has been reading that as evidence that cell-scope negative information works.

**The probability review pointed out that this is not established at all**, and the alternative
explanation is embarrassingly direct: the agent also carries a hand-coded `last_cleared` memory
that charges an explicit +8 penalty for returning to recently swept ground, entirely
independently of the belief. That penalty alone could produce the whole effect.

An earlier check in `review-record.md` claimed to have tested and rejected this confound. It
did not. That check normalised redundant searches per committed action, which answers "is the
agent simply moving less?" It does not separate the two mechanisms, because both are still
switched on in every arm of it. This script is the test that does.

Four arms, crossing the two mechanisms:

    negative info | redundancy penalty | what it isolates
    off           | off                | the agent with neither
    on            | off                | probabilistic negative information alone
    off           | on                 | hand-coded anti-redundancy memory alone
    on            | on                 | the full agent

**Neither switch needs a code change, which is the point.** Negative information is turned off
by setting `search_miss` to 1.0, so a failed search multiplies every cell by one and the belief
is untouched. The redundancy penalty is turned off by setting its weight to 0.0. Nothing else
differs between arms, so any difference is attributable to the switches rather than to a second
edit made at the same time.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import world                                    # noqa: E402
from agent import COSTS, TacticalAgent          # noqa: E402
from belief import PARAMS, Observation          # noqa: E402
import run as R                                 # noqa: E402
from sweep import GEN_SEEDS                     # noqa: E402

RESULTS = ROOT / "results"

ARMS = [
    ("neither",            False, False),
    ("negative info only", True,  False),
    ("memory only",        False, True),
    ("full agent",         True,  True),
]


def run_episode(case, negative_info: bool, memory: bool) -> dict:
    """One episode with the two mechanisms independently switched.

    Mirrors `run.run_episode` rather than calling it, because that function has no way to pass
    belief parameters through. The tick order, the emission rules and the redundancy counting
    are copied from it exactly so the numbers stay comparable with the main results.
    """
    costs = dict(COSTS)
    if not memory:
        costs["redundancy"] = 0.0

    params = dict(PARAMS)
    if not negative_info:
        params["search_miss"] = 1.0     # a failed search multiplies by one: no update at all

    player_rng = np.random.default_rng(case.seed)
    world_rng = np.random.default_rng(case.seed + 104729)
    player = world.Player(position=case.player_start, rng=player_rng)
    agent = TacticalAgent(start=case.post, home_post=case.post, costs=costs,
                          sight_radius=R.SIGHT_RADIUS, belief_params=params)

    agent.observe(Observation("sound", case.player_start, tick=0))

    captured = False
    commits = 0
    redundant = 0
    visited: dict[tuple[int, int], int] = {}

    for tick in range(R.MAX_TICKS):
        true_cell = player.step()
        dist = world.distances_from(agent.cell)
        if (world.has_line_of_sight(agent.cell, true_cell)
                and dist[true_cell] <= R.SIGHT_RADIUS
                and world_rng.random() > R.SEARCH_MISS):
            agent.observe(Observation("sighting", true_cell, tick=tick))
        elif (dist[true_cell] <= R.HEARING_RADIUS
              and world_rng.random() < R.STEP_NOISE_CHANCE):
            agent.observe(Observation("sound", true_cell, tick=tick))

        decision = agent.decide()
        target = decision.target
        if decision.action in ("search", "flank", "call_reinforcements") and target is not None:
            commits += 1
            if target in visited and tick - visited[target] < COSTS["clear_memory"]:
                redundant += 1

        agent.execute(decision)
        visited[agent.cell] = tick

        if (world.has_line_of_sight(agent.cell, true_cell)
                and world.distances_from(agent.cell)[true_cell] <= 1.5
                and world_rng.random() > R.SEARCH_MISS):
            captured = True
            break
        if agent.finished:
            break

    return {"captured": captured, "commits": commits, "redundant": redundant,
            "ticks": tick + 1}


def main() -> None:
    n_cases = 40
    print(f"2x2 ablation, {len(GEN_SEEDS)} seeds x {n_cases} cases per arm\n")

    head = (f"{'arm':22s} {'neg':>4s} {'mem':>4s} {'redundant':>16s} "
            f"{'per commit':>14s} {'capture':>14s}")
    print(head)
    print("-" * len(head))

    table = {}
    rows_out = []
    for label, neg, mem in ARMS:
        per_seed_red, per_seed_rate, per_seed_cap = [], [], []
        for gen_seed in GEN_SEEDS:
            cases = R.build_cases(n_cases, gen_seed=gen_seed)
            results = [run_episode(c, neg, mem) for c in cases]
            red = sum(r["redundant"] for r in results)
            com = sum(r["commits"] for r in results)
            per_seed_red.append(red)
            per_seed_rate.append(red / max(com, 1))
            per_seed_cap.append(np.mean([r["captured"] for r in results]))

        table[label] = {
            "redundant": float(np.mean(per_seed_red)),
            "redundant_sd": float(np.std(per_seed_red)),
            "rate": float(np.mean(per_seed_rate)),
            "rate_sd": float(np.std(per_seed_rate)),
            "capture": float(np.mean(per_seed_cap)),
            "capture_sd": float(np.std(per_seed_cap)),
        }
        t = table[label]
        rows_out.append({"arm": label, "negative_info": neg, "memory": mem,
                         **{k: round(v, 5) for k, v in t.items()}})
        print(f"{label:22s} {'on' if neg else 'off':>4s} {'on' if mem else 'off':>4s} "
              f"{t['redundant']:8.1f} +/-{t['redundant_sd']:5.1f} "
              f"{t['rate']:8.3f} +/-{t['rate_sd']:5.3f} "
              f"{t['capture']:8.1%} +/-{t['capture_sd']:5.1%}")

    with open(RESULTS / "ablation.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nwrote {RESULTS / 'ablation.csv'}")

    # The two effects the review asked for, each measured against the full agent.
    full = table["full agent"]["rate"]
    d_negative = table["memory only"]["rate"] - full
    d_memory = table["negative info only"]["rate"] - full
    print("\nEffect of each mechanism on redundant searches per commit,")
    print("measured by removing it from the full agent:")
    print(f"  removing negative information: {d_negative:+.3f}  "
          f"({table['memory only']['rate']:.3f} vs {full:.3f})")
    print(f"  removing the memory penalty:   {d_memory:+.3f}  "
          f"({table['negative info only']['rate']:.3f} vs {full:.3f})")
    print(f"  removing both:                 "
          f"{table['neither']['rate'] - full:+.3f}  "
          f"({table['neither']['rate']:.3f} vs {full:.3f})")
    print("\nThe larger of the first two is the mechanism actually doing the work.")


if __name__ == "__main__":
    main()
