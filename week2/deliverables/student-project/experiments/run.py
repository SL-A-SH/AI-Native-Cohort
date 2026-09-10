"""The experiment: forty cases, four agents, every decision written down.

    python experiments/run.py                # the full sweep
    python experiments/run.py --cases 5      # a quick look

Writes `results/decisions.csv` (one row per decision) and `results/episodes.csv` (one row per
episode). Both are self-contained: every column needed to recompute a score is in them, so a
reader can audit the result without running any of this code (handover lesson 8).

**The label is hidden when the agent decides.** The loop below is ordered so that the player's
true position is used for exactly three things: emitting observations, letting the one agent
that is allowed to see it see it, and scoring afterwards. `decide()` is called with nothing but
the agent's own state. That ordering is the experiment; if it were wrong the whole thing would
be meaningless, so it is written out step by step rather than tucked into a helper.

**The shadow filter is the yardstick, and it is not any agent's belief.** It receives only the
observations the player actually caused, and it is used to score how attributable each agent's
actions were. That is deliberately the *player's* view of what the NPC could reasonably know:
an agent acting on information the shadow does not have is, from outside, indistinguishable
from an agent that is cheating. It is the same measure for all four agents, including the three
that hold no distribution and could not compute it themselves.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import world                                    # noqa: E402
from agent import COSTS, TacticalAgent          # noqa: E402
from baselines import (                         # noqa: E402
    DegradedOmniscience,
    LastKnownPosition,
    ScriptedSweep,
)
from belief import BeliefGrid, Observation      # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MAX_TICKS = 80          # [ASSUMED] twice the agent's own horizon, so quitting is its choice
SIGHT_RADIUS = 7.0      # [ASSUMED] shared by every agent, so nobody wins on eyesight
HEARING_RADIUS = 9.0    # [ASSUMED] how far a footstep carries
STEP_NOISE_CHANCE = 0.12   # [ASSUMED] chance a given player step is audible
SEARCH_MISS = 0.15         # matches belief.PARAMS, so the world is as forgiving as the model

# Guard posts, spread across the map so cases are not all one geometry.
POSTS = [(11, 3), (2, 3), (12, 20), (2, 20), (8, 12)]


@dataclass
class Case:
    """One scenario. The same case is replayed identically for every agent.

    The player does not react to the agent, so a fixed seed gives a fixed trajectory, and every
    agent faces exactly the same run. That makes the comparison paired rather than independent,
    which is worth far more per case than forty unpaired ones would be.
    """

    case_id: int
    seed: int
    post: tuple[int, int]
    player_start: tuple[int, int]

    @property
    def start_distance(self) -> float:
        return float(world.distances_from(self.post)[self.player_start])


BANDS = [(4, 12), (12, 22), (22, 34), (34, 60)]


def build_cases(count: int, gen_seed: int = 20260910) -> list[Case]:
    """Cases spread deliberately across posts and start distances, not sampled at random.

    Start distance is the variable that matters most here: a player four steps away is a
    different problem from one thirty steps away, and a sample that happened to cluster would
    hide that. So every (post, distance band) pair is walked in turn.

    **Not every pair exists.** The map is 24 by 14, so the furthest cell from the central post
    is 27 steps away and from (12, 20) it is 32: neither has anything in the 34 to 60 band. The
    first version of this retried the same empty pair with `continue`, which spun forever
    because the loop index never advanced, and the whole experiment hung in case generation
    before a single episode ran. Empty pairs are skipped by advancing instead, and the
    generator stops when it runs out of combinations rather than looping.
    """
    rng = np.random.default_rng(gen_seed)
    cases: list[Case] = []

    combos = [(post, band) for band in BANDS for post in POSTS]
    attempts = 0
    while len(cases) < count and attempts < count * len(combos):
        post, (low, high) = combos[attempts % len(combos)]
        attempts += 1
        dist = world.distances_from(post)
        candidates = [c for c in world.FREE_CELLS if low <= dist[c] < high]
        if not candidates:
            continue        # this pair does not exist on this map; the index has moved on
        start = candidates[int(rng.integers(len(candidates)))]
        cases.append(Case(case_id=len(cases), seed=int(rng.integers(1 << 30)),
                          post=post, player_start=start))

    if len(cases) < count:
        raise RuntimeError(
            f"only {len(cases)} of {count} cases could be built from "
            f"{len(POSTS)} posts x {len(BANDS)} bands on this map"
        )
    return cases


def make_agent(kind: str, case: Case, rng: np.random.Generator, poi_weighted: bool,
               costs: dict | None = None):
    if kind == "belief":
        return TacticalAgent(start=case.post, home_post=case.post,
                             poi_weighted=poi_weighted, costs=costs,
                             sight_radius=SIGHT_RADIUS)
    if kind == "last_known_position":
        return LastKnownPosition(case.post, case.post, sight_radius=SIGHT_RADIUS)
    if kind == "scripted_sweep":
        return ScriptedSweep(case.post, case.post, sight_radius=SIGHT_RADIUS)
    if kind == "degraded_omniscience":
        return DegradedOmniscience(case.post, case.post, rng=rng, sight_radius=SIGHT_RADIUS)
    raise ValueError(kind)


def run_episode(case: Case, kind: str, poi_weighted: bool,
                costs: dict | None = None) -> tuple[list[dict], dict]:
    """One agent, one case. Returns its decision rows and one episode summary."""
    player_rng = np.random.default_rng(case.seed)          # identical for every agent
    agent_rng = np.random.default_rng(case.seed + 7919)    # only the noisy baseline uses it
    world_rng = np.random.default_rng(case.seed + 104729)  # emissions and missed sightings

    player = world.Player(position=case.player_start, rng=player_rng)
    agent = make_agent(kind, case, agent_rng, poi_weighted, costs)
    shadow = BeliefGrid(poi_weighted=poi_weighted)

    policy = "B_poi_weighted" if poi_weighted else "A_uniform"
    rows: list[dict] = []
    captured_at: int | None = None
    redundant = 0
    switches = 0
    previous_target = None
    visited: dict[tuple[int, int], int] = {}

    # The inciting incident. Every episode opens with one noise from the player's starting
    # cell, which is what puts the guard into the situation the project is actually about.
    #
    # The first version had no such event, and the results were meaningless because of it: the
    # guard sat at its post until the player happened to wander within earshot, so 1666 of 1906
    # baseline decisions carried the reason "nothing remembered" and `ScriptedSweep` never
    # diverged from `LastKnownPosition` in a single decision out of 1906, because there was
    # never anything to sweep from. Worse, it inverted the comparison: the baselines mostly
    # stood still and were found by a wandering player, while the belief agent went looking and
    # was therefore somewhere else. That measured "does this agent stay put", not "does this
    # agent search well".
    #
    # An opening noise is also the scenario every figure and both videos already show, so the
    # experiment now tests the thing the paper describes.
    opening = Observation("sound", case.player_start, tick=0)
    agent.observe(opening)
    shadow.update(opening, agent_cell=agent.cell)

    for tick in range(MAX_TICKS):
        # 1. Ground truth moves. Nothing has looked at it yet.
        true_cell = player.step()

        # 2. What the world emits. This is the only channel the player's own actions travel
        #    down, and it is what the shadow filter is allowed to see.
        emitted: list[Observation] = []
        agent_dist = world.distances_from(agent.cell)
        if (world.has_line_of_sight(agent.cell, true_cell)
                and agent_dist[true_cell] <= SIGHT_RADIUS
                and world_rng.random() > SEARCH_MISS):
            emitted.append(Observation("sighting", true_cell, tick=tick))
        elif (agent_dist[true_cell] <= HEARING_RADIUS
              and world_rng.random() < STEP_NOISE_CHANCE):
            emitted.append(Observation("sound", true_cell, tick=tick))

        for obs in emitted:
            agent.observe(obs)
            shadow.update(obs, agent_cell=agent.cell)

        # 3. The one agent permitted ground truth gets it here, gated inside its own class.
        if isinstance(agent, DegradedOmniscience):
            agent.reveal(true_cell)

        # 4. The decision. No argument to this call carries the player's position, and the
        #    agent objects hold no reference to `player`. This is the line the experiment
        #    depends on.
        decision = agent.decide()

        # 5. Score it against the truth the agent could not see.
        target = decision.target
        committed = decision.action in ("search", "flank", "call_reinforcements")
        if target is not None:
            findable = (world.has_line_of_sight(target, true_cell)
                        and world.distances_from(target)[true_cell] <= SIGHT_RADIUS)
            attribution = shadow.evidence_ratio(target)
        else:
            findable = (world.has_line_of_sight(agent.cell, true_cell)
                        and agent_dist[true_cell] <= SIGHT_RADIUS)
            attribution = float("nan")

        if committed and target is not None:
            if target in visited and tick - visited[target] < (costs or COSTS)["clear_memory"]:
                redundant += 1
            if previous_target is not None and target != previous_target \
                    and agent.cell != previous_target:
                switches += 1
            previous_target = target

        row = decision.as_row()
        row.update({
            "case": case.case_id,
            "agent": kind,
            "policy": policy,
            "seed": case.seed,
            "committed": int(committed),
            "findable": int(findable),
            "attribution": "" if np.isnan(attribution) else round(attribution, 4),
            "true_r": true_cell[0],
            "true_c": true_cell[1],
            "shadow_peak_r": shadow.most_likely()[0],
            "shadow_peak_c": shadow.most_likely()[1],
        })
        rows.append(row)

        # 6. Act, then let the world move on.
        agent.execute(decision)
        shadow.predict(steps=1)
        visited[agent.cell] = tick

        # 7. Capture check, after the move, with the same miss chance the model assumes.
        if (world.has_line_of_sight(agent.cell, true_cell)
                and world.distances_from(agent.cell)[true_cell] <= 1.5
                and world_rng.random() > SEARCH_MISS):
            captured_at = tick
            break

        if getattr(agent, "finished", False):
            break

    committed_rows = [r for r in rows if r["committed"]]
    attributions = [float(r["attribution"]) for r in committed_rows if r["attribution"] != ""]

    summary = {
        "case": case.case_id,
        "agent": kind,
        "policy": policy,
        "seed": case.seed,
        "start_distance": round(case.start_distance, 1),
        "ticks": len(rows),
        "captured": int(captured_at is not None),
        "capture_tick": captured_at if captured_at is not None else "",
        "gave_up": int(getattr(agent, "finished", False)),
        "redundant_searches": redundant,
        "target_switches": switches,
        "called_reinforcements": int(any(r["action"] == "call_reinforcements" for r in rows)),
        "mean_attribution": round(float(np.mean(attributions)), 4) if attributions else "",
        "low_attribution_actions": sum(1 for a in attributions if a < 1.0),
        "committed_actions": len(committed_rows),
        "decision_cost": round(float(np.nansum(
            [r["chosen_cost"] for r in rows if not np.isnan(r["chosen_cost"])])), 3),
    }
    return rows, summary


def confusion(rows: list[dict]) -> dict:
    """The confusion matrix, per §9, for a problem that is not a classifier.

    Each decision is read as an implicit claim: **is the player findable at the place I am
    committing to right now?** Committing is the positive call, disengaging is the negative
    one, and the truth is whether going there would in fact have found them.

    The two error types are the two complaints the whole project is about. A false positive is
    an agent pressing on when the player is long gone, which reads as flailing. A false negative
    is an agent standing down while the player is right there, which reads as stupid.
    """
    tp = sum(1 for r in rows if r["committed"] and r["findable"])
    fp = sum(1 for r in rows if r["committed"] and not r["findable"])
    fn = sum(1 for r in rows if not r["committed"] and r["findable"])
    tn = sum(1 for r in rows if not r["committed"] and not r["findable"])
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=40)
    args = parser.parse_args()

    cases = build_cases(args.cases)
    arms = [
        ("belief", False),
        ("belief", True),
        ("last_known_position", False),
        ("scripted_sweep", False),
        ("degraded_omniscience", False),
    ]

    all_rows: list[dict] = []
    all_summaries: list[dict] = []

    for kind, poi in arms:
        for case in cases:
            rows, summary = run_episode(case, kind, poi)
            all_rows.extend(rows)
            all_summaries.append(summary)

    with open(RESULTS / "decisions.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    with open(RESULTS / "episodes.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(all_summaries)

    print(f"{len(cases)} cases, {len(arms)} arms, "
          f"{len(all_summaries)} episodes, {len(all_rows)} decisions")
    print(f"wrote {RESULTS / 'decisions.csv'}")
    print(f"wrote {RESULTS / 'episodes.csv'}\n")

    header = (f"{'arm':24s} {'capture':>8s} {'ticks':>7s} {'attrib':>7s} "
              f"{'low':>5s} {'redund':>7s} {'switch':>7s} {'prec':>6s} {'rec':>6s}")
    print(header)
    print("-" * len(header))
    for kind, poi in arms:
        label = f"{kind}{' (B)' if poi else ''}"
        eps = [s for s in all_summaries
               if s["agent"] == kind and s["policy"].startswith("B" if poi else "A")]
        rows = [r for r in all_rows
                if r["agent"] == kind and r["policy"].startswith("B" if poi else "A")]
        cm = confusion(rows)
        attrs = [s["mean_attribution"] for s in eps if s["mean_attribution"] != ""]
        print(f"{label:24s} "
              f"{np.mean([s['captured'] for s in eps]):8.1%} "
              f"{np.mean([s['ticks'] for s in eps]):7.1f} "
              f"{(np.mean(attrs) if attrs else float('nan')):7.2f} "
              f"{sum(s['low_attribution_actions'] for s in eps):5d} "
              f"{sum(s['redundant_searches'] for s in eps):7d} "
              f"{sum(s['target_switches'] for s in eps):7d} "
              f"{cm['precision']:6.2f} {cm['recall']:6.2f}")


if __name__ == "__main__":
    main()
