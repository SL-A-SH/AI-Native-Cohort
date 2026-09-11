"""Generate every number in `decisions/probability-decision-record.md`.

    python experiments/decision_case.py

§10 wants one decision recorded in full: the evidence, the hidden states with beliefs summing
to 100%, the actions and their costs, the decision and its reason, then one new piece of
evidence walked all the way through to a new action.

A belief over 201 cells is not a legible table, so the cells are grouped into eight named
regions **for reporting only**. The agent never sees them and never reasons over them; they are
a lens on its belief, not part of it. The script asserts the regions partition the free cells
exactly, so nothing is double counted or quietly dropped.

The case is case 10 tick 29, chosen by scanning every tick of the first twelve cases for one
where the agent is genuinely torn: two regions holding 40% and 33% of the belief with an
entropy of 0.82 of maximum. Case 0 tick 3 was the first candidate, being the clearest example
of failure F1, but it turned out to be a poor record: the agent had already sighted the player
twice and held 99.6% of its belief in one region, so the prior/posterior table was trivial and
the new evidence moved nothing. A record of a decision the agent was already certain about
would not demonstrate anything §10 asks for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import world                                  # noqa: E402
from agent import TacticalAgent               # noqa: E402
from belief import PARAMS, Observation        # noqa: E402
import run as R                               # noqa: E402

# The case and the tick this record walks through.
CASE_ID = 10
TICK = 29
# Chosen to contradict the leading hypothesis. The Mid corridor is the second-placed region, so
# a noise there tests whether the agent updates against its own lead rather than confirming it.
NEW_EVIDENCE_CELL = (6, 20)

# Eight regions, named by where they are on the map. Reporting only.
REGIONS = {
    "NW room":       [(1, 4, 1, 6)],
    "N corridor":    [(1, 2, 8, 15), (3, 3, 8, 10), (4, 4, 7, 10)],
    "NE room":       [(1, 4, 16, 22)],
    "W rooms":       [(5, 9, 1, 6)],
    "Mid corridor":  [(5, 7, 7, 11), (6, 6, 12, 22), (4, 5, 12, 15)],
    "Centre":        [(8, 10, 7, 14), (7, 7, 16, 16)],
    "SE room":       [(8, 11, 15, 22), (10, 10, 15, 18)],
    "S corridor":    [(10, 12, 1, 6), (12, 12, 7, 22)],
}


def region_masks() -> dict[str, np.ndarray]:
    masks, seen = {}, np.zeros((world.H, world.W), dtype=bool)
    for name, boxes in REGIONS.items():
        m = np.zeros((world.H, world.W), dtype=bool)
        for r0, r1, c0, c1 in boxes:
            m[r0:r1 + 1, c0:c1 + 1] = True
        m &= world.FREE
        overlap = m & seen
        if overlap.any():
            raise AssertionError(f"{name} overlaps an earlier region at "
                                 f"{[tuple(int(x) for x in c) for c in zip(*np.where(overlap))]}")
        seen |= m
        masks[name] = m

    missing = world.FREE & ~seen
    if missing.any():
        raise AssertionError(
            f"{int(missing.sum())} free cells belong to no region: "
            f"{[tuple(int(x) for x in c) for c in zip(*np.where(missing))][:12]}"
        )
    return masks


MASKS = region_masks()


def table(belief: np.ndarray, label: str) -> None:
    print(f"\n{label}")
    print(f"  {'region':14s} {'cells':>6s} {'P(player here)':>15s}")
    total = 0.0
    for name, mask in MASKS.items():
        p = float(belief[mask].sum())
        total += p
        print(f"  {name:14s} {int(mask.sum()):6d} {p:15.1%}")
    print(f"  {'TOTAL':14s} {len(world.FREE_CELLS):6d} {total:15.1%}")


def which_region(cell) -> str:
    for name, mask in MASKS.items():
        if mask[cell]:
            return name
    return "?"


def replay_to_tick(target_tick: int):
    """Re-run the case up to a tick, returning the agent and the player's true position."""
    case = R.build_cases(40)[CASE_ID]
    prng = np.random.default_rng(case.seed)
    wrng = np.random.default_rng(case.seed + 104729)
    player = world.Player(position=case.player_start, rng=prng)
    agent = TacticalAgent(start=case.post, home_post=case.post,
                          sight_radius=R.SIGHT_RADIUS)
    agent.observe(Observation("sound", case.player_start, tick=0))

    true_cell = case.player_start
    log = []
    for tick in range(target_tick + 1):
        true_cell = player.step()
        dist = world.distances_from(agent.cell)
        emitted = None
        if (world.has_line_of_sight(agent.cell, true_cell)
                and dist[true_cell] <= R.SIGHT_RADIUS and wrng.random() > R.SEARCH_MISS):
            emitted = Observation("sighting", true_cell, tick=tick)
        elif dist[true_cell] <= R.HEARING_RADIUS and wrng.random() < R.STEP_NOISE_CHANCE:
            emitted = Observation("sound", true_cell, tick=tick)
        if emitted:
            agent.observe(emitted)
            log.append((tick, emitted.kind, emitted.cell))

        decision = agent.decide()
        if tick == target_tick:
            return case, agent, true_cell, decision, log
        agent.execute(decision)
    raise RuntimeError("unreachable")


def main() -> None:
    case, agent, true_cell, decision, log = replay_to_tick(TICK)

    print("=" * 78)
    print(f"CASE {case.case_id}, TICK {TICK}")
    print("=" * 78)
    print(f"guard post          {case.post}")
    print(f"guard now           {agent.cell}  ({which_region(agent.cell)})")
    print(f"player actually at  {true_cell}  ({which_region(true_cell)})   "
          f"<-- hidden from the agent")
    print(f"walking distance    {world.distances_from(agent.cell)[true_cell]:.0f} steps")
    print(f"\nobservations received so far:")
    print(f"  tick 0  sound at {case.player_start}  (the opening noise)")
    for tick, kind, cell in log:
        print(f"  tick {tick}  {kind} at {cell}")

    table(agent.belief.belief, "BELIEF BEFORE THE NEW EVIDENCE (the prior)")

    print(f"\nnear/gone split, the two states the decision turns on:")
    reach = world.visible_from(agent.cell, radius=R.SIGHT_RADIUS)
    p_near = float(agent.belief.belief[reach].sum())
    print(f"  H_near  player within reach of where the guard stands   {p_near:6.1%}")
    print(f"  H_gone  player anywhere else                            {1 - p_near:6.1%}")

    print(f"\nentropy {agent.belief.normalised_entropy():.3f} of max   "
          f"peak {agent.belief.most_likely()} at "
          f"{agent.belief.belief[agent.belief.most_likely()]:.1%}   "
          f"peak traceability {agent.belief.evidence_ratio(agent.belief.most_likely()):.1f}")

    print("\nOPTIONS AND COSTS")
    for opt in decision.options:
        print("  ", opt.explain())
    print(f"\n  chosen: {decision.action}")
    print(f"  reason: {decision.reason}")

    # ---------------------------------------------------------------- the new evidence
    print("\n" + "=" * 78)
    print(f"NEW EVIDENCE: a sound at {NEW_EVIDENCE_CELL}")
    print("=" * 78)

    prior = agent.belief.belief.copy()
    new_obs = Observation("sound", NEW_EVIDENCE_CELL, tick=TICK + 1)

    # The likelihood comes from the agent's own method rather than being recomputed here, so
    # the table cannot drift from what the filter actually applies to its belief.
    print("\nlikelihood P(hear this | player in region), from belief.PARAMS:")
    print(f"  sound_sigma {PARAMS['sound_sigma']:.2f}, sound_floor {PARAMS['sound_floor']:.2f}, "
          f"distance measured through the map rather than across it")
    print(f"\n  {'region':14s} {'prior':>8s} {'mean L':>8s} {'prior x L':>11s} {'posterior':>11s}")

    like = agent.belief._positive_likelihood(new_obs, agent.cell)
    unnorm = prior * like
    evidence = float(unnorm.sum())

    for name, mask in MASKS.items():
        pri = float(prior[mask].sum())
        mean_l = float(like[mask].mean())
        joint = float(unnorm[mask].sum())
        print(f"  {name:14s} {pri:8.1%} {mean_l:8.2f} {joint:11.4f} "
              f"{joint / evidence:11.1%}")
    print(f"  {'TOTAL':14s} {1.0:8.1%} {'':8s} {evidence:11.4f} {1.0:11.1%}")
    print(f"\n  P(evidence) = {evidence:.4f}")

    agent.observe(new_obs)
    table(agent.belief.belief, "BELIEF AFTER THE NEW EVIDENCE (the posterior)")

    reach2 = world.visible_from(agent.cell, radius=R.SIGHT_RADIUS)
    p_near2 = float(agent.belief.belief[reach2].sum())
    print(f"\nnear/gone split after the update:")
    print(f"  H_near  {p_near2:6.1%}   (was {p_near:.1%})")
    print(f"  H_gone  {1 - p_near2:6.1%}   (was {1 - p_near:.1%})")
    print(f"\nentropy {agent.belief.normalised_entropy():.3f}   "
          f"peak {agent.belief.most_likely()} at "
          f"{agent.belief.belief[agent.belief.most_likely()]:.1%}   "
          f"peak traceability {agent.belief.evidence_ratio(agent.belief.most_likely()):.1f}")

    new_decision = agent.decide()
    print("\nOPTIONS AND COSTS AFTER THE UPDATE")
    for opt in new_decision.options:
        print("  ", opt.explain())
    print(f"\n  chosen: {new_decision.action} -> {new_decision.target}")
    print(f"  reason: {new_decision.reason}")
    print(f"\n  decision changed: {decision.action} -> {new_decision.action}")


if __name__ == "__main__":
    main()
