"""The agent: what it can do, what each option costs, and how it picks one.

This is the part of the project that is actually being argued about. The belief model in
`belief.py` is standard machinery, well understood and borrowed rather than invented. The cost
function here is the contribution, and it is where review pressure should land.

**The objective is not to find the player.** An agent that minimises time-to-capture walks
straight at you every time, which is a defect rather than a result. What is being written down
instead is a cost function whose optimum is an opponent that *looks like it is reasoning* and
loses at a rate a designer chose. Three of the six cost terms exist only to price how the
behaviour reads, and they came out of talking to people rather than out of a textbook:

    illegibility   acting on belief the player cannot trace to anything they did  (R6)
    redundancy     re-searching ground already cleared                            (R5)
    dither         abandoning a target before reaching it                         (R7)

See `discussion-record.md` for where each came from.

**The human reasoning function (§8) is `identify uncertainty`.** The agent computes how little
it knows, prices every option under that uncertainty, and is allowed to conclude that no active
option is worth taking. It is not a tuned entropy threshold: holding and resuming post are
priced like everything else, so "I do not know enough to commit" falls out of the comparison
rather than being asserted by a magic number. That matters because a hand-set threshold was
beaten by cost-derived boundaries on the previous agent, and the discrepancy sat in the
decision record for days before anyone noticed.

**The agent never sees the player.** It reads its own belief and its own position, and nothing
in this module imports ground truth. The simulator decides what it is told; the boundary is
enforced by keeping `Player` out of every signature here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import world
from belief import BeliefGrid, Observation

# ---------------------------------------------------------------------------- the parameters
# Every number here is [ASSUMED] and collected in one dict so the whole set of invented values
# can be read at once, and so the sensitivity sweep has a single place to reach into. None of
# them has been tuned against a result. Where a value could be derived rather than chosen, it
# is derived instead and does not appear here at all: the attributability boundary is 1.0
# because that is where belief equals what the agent would think anyway, not because 1.0 looked
# right.
COSTS: dict[str, float] = {
    # The pressure to do anything at all. Every tick the player is still at large costs this,
    # which is what stops the agent holding position forever.
    "miss_per_tick": 1.0,          # [ASSUMED] the unit the rest are denominated in

    # Tactical costs.
    "travel_per_step": 0.35,       # [ASSUMED] a step is cheaper than a tick of failure
    "exposure_per_step": 0.25,     # [ASSUMED] extra, for moving through open ground
    "reinforcement_call": 12.0,    # [ASSUMED] a big, visible, once-per-episode commitment

    # The believability terms. These are the contribution and the part to defend.
    "illegibility": 6.0,           # [ASSUMED] weight on acting where nothing pointed
    "redundancy": 8.0,             # [ASSUMED] weight on re-searching cleared ground
    "dither": 4.0,                 # [ASSUMED] weight on abandoning a committed target

    # How long the agent treats an episode as having left to run. Used to price the two
    # terminal actions, which forfeit the rest of it.
    "horizon_ticks": 40.0,         # [ASSUMED]

    # How long a cleared cell stays cleared, in ticks, for the redundancy term. Not a belief
    # parameter: the filter already handles the probability. This is only about how recently
    # the agent stood there and looked, which is what a player would notice.
    "clear_memory": 25.0,          # [ASSUMED]
}

# The six actions, and the §8 category each one answers to.
ACTIONS = (
    "search",               # get more evidence
    "hold_position",        # wait
    "flank",                # act
    "call_reinforcements",  # escalate
    "retreat",              # refuse, recoverable
    "resume_post",          # refuse, terminal
)

TERMINAL = ("resume_post",)

# Ceiling on the unattributable fraction. A ratio approaching zero means the evidence pushed
# mass hard away from a cell; without a cap that one term would dominate every comparison and
# the other five would stop mattering. 3.0 says "three times worse than acting on nothing",
# which is enough to rule the move out without making it the only thing the agent can see.
_UNATTRIBUTABLE_CAP = 3.0  # [ASSUMED]


@dataclass
class Option:
    """One thing the agent could do this tick, with its cost fully broken out.

    The breakdown is kept rather than summed away because §9 wants every decision auditable
    from the results file alone, and because a total tells you what the agent did while the
    terms tell you why.
    """

    action: str
    target: tuple[int, int] | None
    terms: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return float(sum(self.terms.values()))

    def explain(self) -> str:
        parts = [f"{k} {v:+.2f}" for k, v in self.terms.items() if abs(v) > 1e-9]
        where = f" -> {self.target}" if self.target else ""
        return f"{self.action}{where}: {self.total:+.2f}  ({', '.join(parts)})"


@dataclass
class Decision:
    """What the agent did and everything needed to recompute why.

    Written straight out to the results CSV. A reader with the CSV and no code should be able
    to reconstruct the comparison, per handover lesson 8.
    """

    tick: int
    action: str
    target: tuple[int, int] | None
    agent_cell: tuple[int, int]
    reason: str
    chosen_cost: float
    runner_up: str
    margin: float
    entropy: float
    peak_cell: tuple[int, int]
    peak_belief: float
    peak_evidence_ratio: float
    options: list[Option] = field(default_factory=list)

    def as_row(self) -> dict:
        """One flat dict per decision, for the results file."""
        return {
            "tick": self.tick,
            "action": self.action,
            "target_r": self.target[0] if self.target else "",
            "target_c": self.target[1] if self.target else "",
            "agent_r": self.agent_cell[0],
            "agent_c": self.agent_cell[1],
            "reason": self.reason,
            "chosen_cost": round(self.chosen_cost, 4),
            "runner_up": self.runner_up,
            "margin": round(self.margin, 4),
            "entropy": round(self.entropy, 4),
            "peak_r": self.peak_cell[0],
            "peak_c": self.peak_cell[1],
            "peak_belief": round(self.peak_belief, 6),
            "peak_evidence_ratio": round(self.peak_evidence_ratio, 4),
        }


class TacticalAgent:
    """One NPC. Holds a belief, prices six options against it, and takes the cheapest.

    The belief object has a single owner but does not assume a single reader, which is what
    keeps the squad expansion open without paying for it now.
    """

    def __init__(
        self,
        start: tuple[int, int],
        home_post: tuple[int, int],
        poi_weighted: bool = False,
        costs: dict[str, float] | None = None,
        sight_radius: float = 7.0,
    ):
        self.cell = start
        self.home_post = home_post
        self.sight_radius = sight_radius
        self.costs = dict(COSTS if costs is None else costs)
        self.belief = BeliefGrid(poi_weighted=poi_weighted)

        self.tick = 0
        # Ticks since the last *positive* observation. Drives the staleness term on holding:
        # the agent knows how long it has been staring at nothing, which is information it
        # genuinely has and a player can see it acting on.
        self.ticks_since_evidence = 0
        # Whether anything has ever actually been observed. Gates the terminal action: an agent
        # that has seen nothing is on duty, not pursuing, and has nothing to give up on.
        self.has_had_evidence = False
        self.committed_target: tuple[int, int] | None = None
        self.reinforcements_called = False
        self.finished = False

        # When each cell was last stood in and looked at. Drives the redundancy term, and is
        # deliberately separate from the belief: the filter already knows the probability, this
        # is about what a watching player would recognise as "it just checked there".
        self.last_cleared: dict[tuple[int, int], int] = {}

    # ------------------------------------------------------------------------- perception
    def observe(self, obs: Observation) -> None:
        self.belief.update(obs, agent_cell=self.cell)
        if obs.kind != "search":
            self.ticks_since_evidence = 0
            self.has_had_evidence = True

    def look(self) -> Observation:
        """Look at everything visible from where it stands, and find nothing.

        Returns the observation so the simulator can decide whether it was actually a miss.
        The agent applies it either way: from the inside, looking and seeing nothing is the
        same event regardless of whether the player was there and got lucky.
        """
        visible = world.visible_from(self.cell, radius=self.sight_radius)
        cells = tuple((int(r), int(c)) for r, c in zip(*np.where(visible)))
        for cell in cells:
            self.last_cleared[cell] = self.tick
        return Observation("search", self.cell, tick=self.tick, cells=cells)

    # ------------------------------------------------------------------------ the costing
    def _travel(self, target: tuple[int, int]) -> float:
        steps = float(world.distances_from(self.cell)[target])
        if not np.isfinite(steps):
            return float("inf")
        return steps * (self.costs["travel_per_step"] + self.costs["exposure_per_step"])

    def unattributable_fraction(self, target: tuple[int, int]) -> float:
        """How much of the belief at `target` is *not* owed to anything the player did.

        The evidence ratio is belief over what the agent would have believed with no positive
        observation at all, so the attributable share of that belief is `1 - 1/ratio` and the
        unattributable share is `1/ratio`. At ratio 11 only 9% of the confidence is unaccounted
        for; at ratio 1, all of it is; below 1 it passes 100%, because the evidence actually
        pushed mass *away* from here and the agent is going anyway.

        Every quantity in that sentence falls out of the definition of the ratio. Nothing is
        chosen, which is the property the illegibility term needs in order to be defensible.
        """
        ratio = self.belief.evidence_ratio(target)
        if ratio <= 0.0:
            return _UNATTRIBUTABLE_CAP
        return min(1.0 / ratio, _UNATTRIBUTABLE_CAP)

    def _illegibility(self, target: tuple[int, int]) -> float:
        """The price of moving somewhere nothing pointed to.

        The first version of this went to zero whenever the ratio reached 1.0, and 1.0 is
        exactly the unattributable case: belief equal to what the agent would think anyway.
        The result was an agent that flanked confidently having observed nothing at all, which
        is the precise behaviour this term exists to prevent. It is priced off the
        unattributable fraction instead, so a move nothing supports is never free.
        """
        return self.costs["illegibility"] * self.unattributable_fraction(target)

    def _redundancy(self, target: tuple[int, int]) -> float:
        """The price of walking back into a room it just cleared.

        This is the failure every player has watched happen. It decays: coming back an hour
        later is reasonable, coming back in ten seconds is not.
        """
        seen = self.last_cleared.get(target)
        if seen is None:
            return 0.0
        age = self.tick - seen
        memory = self.costs["clear_memory"]
        if age >= memory:
            return 0.0
        return self.costs["redundancy"] * (1.0 - age / memory)

    def _dither(self, target: tuple[int, int] | None) -> float:
        """The price of changing its mind before arriving.

        Without this the agent flips between two near-equal cells forever, because each tick it
        re-picks whichever is fractionally ahead. A developer on r/roguelikedev animates that
        same flipping deliberately to show a monster is confused (R7), so the flipping is not
        itself the failure: unbounded, unexplained flipping is. This term bounds it, and
        `resume_post` ends it.
        """
        if self.committed_target is None or target is None:
            return 0.0
        if target == self.committed_target:
            return 0.0
        if self.cell == self.committed_target:
            return 0.0  # arrived, so switching is not abandoning
        return self.costs["dither"]

    def _expected_gain(self, target: tuple[int, int]) -> float:
        """What going there would be worth, as belief mass it has not already accounted for.

        Negative, because it offsets cost. Priced in the same units as everything else: mass
        resolved is failure-ticks not paid later.

        **Only mass in cells it has not recently cleared counts.** The first version credited
        every cell visible from the target, every tick, which meant an agent standing where it
        could see most of its own belief was paid the entire episode's value for staying still,
        and paid it again on the next tick, and the next. The forty-case run showed the
        consequence exactly: 87% of all decisions were `hold_position`, including cases where
        the agent could see the player three steps away and simply watched. Holding scored
        -36.18 against -22.27 for walking over to look.

        The error was conflating *observing* mass with *resolving* it. Seeing a cell tells the
        agent whether the player is in it; it does not end anything, and looking at the same
        cell a second time tells it almost nothing it did not already know. Counting only
        uncleared cells is what makes information gain actually diminish, which is the property
        the word "gain" was claiming all along.
        """
        visible = world.visible_from(target, radius=self.sight_radius)
        fresh = visible.copy()
        for cell, seen in self.last_cleared.items():
            if self.tick - seen < self.costs["clear_memory"]:
                fresh[cell] = False
        mass = float(self.belief.belief[fresh].sum())
        return -mass * self.costs["miss_per_tick"] * self.costs["horizon_ticks"]

    # -------------------------------------------------------------------------- the options
    def _option_search(self) -> list[Option]:
        """Search the most promising cells, not every cell.

        Candidates are the belief peak plus the strongest chokepoints, which keeps the
        comparison small and legible. Considering all 201 cells would cost nothing at this size
        but would make the audit trail unreadable, and the argmin over a shortlist is the same
        decision whenever the shortlist contains the winner.
        """
        peak = self.belief.most_likely()
        candidates = {peak}
        chokepoints = sorted(
            world.POINTS_OF_INTEREST,
            key=lambda c: self.belief.belief[c],
            reverse=True,
        )
        candidates.update(chokepoints[:4])

        options = []
        for target in candidates:
            travel = self._travel(target)
            if not np.isfinite(travel):
                continue
            options.append(Option("search", target, {
                "travel": travel,
                "gain": self._expected_gain(target),
                "illegibility": self._illegibility(target),
                "redundancy": self._redundancy(target),
                "dither": self._dither(target),
            }))
        return options

    def _option_hold(self) -> Option:
        """Stay put and watch. Cheap, safe, and it learns almost nothing.

        The staleness term is what stops this being free forever. Holding is a bet that new
        evidence will arrive; every tick it does not, the bet has been losing for longer, and
        the agent has direct evidence that standing here is not producing anything. So the
        price of holding is scaled by how long it has been since anything was observed.

        Without this the agent stands still indefinitely, which is exactly the unbounded failure
        the dither term was added to prevent, wearing a different costume. A single tick of
        holding will always beat forfeiting a whole episode, so a myopic comparison never ends.
        """
        staleness = 1.0 + self.ticks_since_evidence / self.costs["horizon_ticks"]
        return Option("hold_position", None, {
            "miss": self.costs["miss_per_tick"] * staleness,
            "gain": self._expected_gain(self.cell),
        })

    def _option_flank(self) -> list[Option]:
        """Move to a chokepoint the player would have to pass through.

        Positional rather than investigative: it does not go to where the belief is, it goes to
        where the belief would have to come out. Only chokepoints away from the peak qualify,
        because standing on the peak is a search by another name.
        """
        peak = self.belief.most_likely()
        peak_dist = world.distances_from(peak)

        options = []
        for choke in world.POINTS_OF_INTEREST:
            gap = peak_dist[choke]
            if not np.isfinite(gap) or gap < 2 or gap > 8:
                continue
            travel = self._travel(choke)
            if not np.isfinite(travel):
                continue
            # Worth the trip in proportion to how much *more* belief sits beyond it than would
            # have sat there anyway. Measured against the null belief for the same reason the
            # illegibility term is: under a uniform belief half the map lies beyond any
            # chokepoint, so an absolute count makes cutting off an escape route look valuable
            # when the agent has observed nothing at all, and the first version of this did
            # exactly that on tick zero.
            #
            # Search gain is deliberately *not* null-adjusted, and the asymmetry is the point.
            # Looking somewhere genuinely resolves whatever mass is there, whatever put it
            # there. Standing at a chokepoint only pays off if evidence actually concentrated
            # something behind it.
            outside = peak_dist > gap
            beyond = float(self.belief.belief[outside].sum())
            beyond_null = float(self.belief.null_belief[outside].sum())
            beyond = max(0.0, beyond - beyond_null)
            options.append(Option("flank", choke, {
                "travel": travel,
                "gain": -beyond * self.costs["miss_per_tick"] * self.costs["horizon_ticks"] * 0.5,
                "illegibility": self._illegibility(choke),
                "dither": self._dither(choke),
            }))
        options.sort(key=lambda o: o.total)
        return options[:3]

    def _option_reinforcements(self) -> Option | None:
        """Escalate. Expensive, once per episode, and only sensible when it is fairly sure.

        Calling a squad because it heard a mouse is exactly the behaviour that reads as broken,
        so the illegibility term is applied to the peak: the less traceable the belief, the
        worse an escalation looks.
        """
        if self.reinforcements_called:
            return None
        peak = self.belief.most_likely()
        confidence = float(self.belief.belief[peak])
        return Option("call_reinforcements", peak, {
            "call": self.costs["reinforcement_call"],
            "gain": -confidence * self.costs["miss_per_tick"] * self.costs["horizon_ticks"] * 2.0,
            "illegibility": self._illegibility(peak) * 2.0,
        })

    def _option_retreat(self) -> Option:
        """Break off toward the post without ending the episode. Recoverable.

        Retreating is a *form of not pursuing*, so it carries the same staleness the holding
        option does. Without that it was a flat 1.35 forever while holding grew with time, and
        the agent discovered it could dodge the staleness penalty entirely by shuffling toward
        its post. The forty-case run showed it retreating on 54% of all decisions, which
        wrecked both precision and recall and had nothing to do with the belief model.

        With staleness applied, retreat is holding plus the price of walking, so on this map it
        should essentially never win. That is the correct outcome and it exposes something the
        paper has to say plainly: **nothing in this simulation makes an agent safer by
        withdrawing.** There is no threat model, the player cannot hurt the guard, so the one
        thing that would justify retreating is not modelled. Pricing it as though it had a
        benefit would be modelling an action the system cannot actually perform, which is
        precisely the mistake that cost the previous agent (handover lesson 7).

        It is kept in the action set, priced honestly, and reported in Limitations as an option
        that cannot earn its place until a threat model exists.
        """
        staleness = 1.0 + self.ticks_since_evidence / self.costs["horizon_ticks"]
        step = world.step_towards(self.cell, self.home_post)
        return Option("retreat", step, {
            "miss": self.costs["miss_per_tick"] * staleness,
            "travel": self.costs["travel_per_step"],
        })

    def _option_resume_post(self) -> Option | None:
        """Give up and go back to work. The only terminal action, and the bound on everything.

        What quitting costs is the rest of the episode, but only to the extent there was still
        something to give up. That is priced off the attributable share of the peak: with a
        strong lead, most of the belief is owed to something the player did, and walking away
        throws a real chance away. With a cold trail nothing is owed to anything, the agent is
        only staring at its own diffusion, and there is no chance to throw away.

        So this gets cheaper as the trail goes cold, while holding gets dearer the longer it
        produces nothing, and the two cross on their own. That crossing is the bound R7 asked
        for, and it is derived from observable quantities rather than set as a timeout, which
        matters because a hand-set timer was the exact thing lesson 5 warns about.

        It exists at all because four separate developers independently described this
        behaviour, and because without a way to stop, the agent cannot end an episode.
        """
        # You cannot give up on a chase you never started. Until something has actually been
        # observed the agent is not pursuing anything, it is simply on duty, and the correct
        # behaviour is to stand there rather than to declare the episode over.
        #
        # Leaving this out was a real regression and the experiment caught it: forfeiting is
        # priced by how much of the belief is owed to evidence, so with no evidence at all it
        # cost exactly nothing, and the agent resigned on tick three of every episode before
        # the player had made a sound. Capture rate fell to 25% against 75% for every baseline.
        if not self.has_had_evidence:
            return None

        attributable = max(0.0, 1.0 - self.unattributable_fraction(self.belief.most_likely()))
        return Option("resume_post", self.home_post, {
            "forfeit": self.costs["miss_per_tick"] * self.costs["horizon_ticks"] * attributable,
        })

    # -------------------------------------------------------------------------- the decision
    def decide(self) -> Decision:
        """Price everything, take the cheapest, and record why.

        Nothing about the player's true position enters here. The only inputs are the belief,
        the agent's own position, and what it remembers having already looked at.
        """
        options: list[Option] = []
        options.extend(self._option_search())
        options.append(self._option_hold())
        options.extend(self._option_flank())
        reinforcements = self._option_reinforcements()
        if reinforcements is not None:
            options.append(reinforcements)
        options.append(self._option_retreat())
        give_up = self._option_resume_post()
        if give_up is not None:
            options.append(give_up)

        options.sort(key=lambda o: o.total)
        best = options[0]
        second = options[1] if len(options) > 1 else best

        peak = self.belief.most_likely()
        entropy = self.belief.normalised_entropy()
        ratio = self.belief.evidence_ratio(peak)

        # The human reasoning function, stated in the agent's own terms. This is a description
        # of a comparison that already happened, not a second rule applied on top of it.
        if best.action in TERMINAL:
            reason = (f"nothing worth doing: best active option cost {second.total:+.2f}, "
                      f"uncertainty {entropy:.2f}")
        elif best.action == "hold_position":
            reason = (f"too little to go on to commit: uncertainty {entropy:.2f}, "
                      f"peak traceability {ratio:.2f}")
        elif best.terms.get("illegibility", 0.0) >= max(
                (v for k, v in best.terms.items() if v > 0 and k != "illegibility"), default=0.0):
            reason = (f"acting on weakly supported belief: traceability "
                      f"{self.belief.evidence_ratio(best.target):.2f}, "
                      f"{self.unattributable_fraction(best.target):.0%} unaccounted for")
        else:
            reason = (f"following the evidence: peak {peak} at {self.belief.belief[peak]:.3f}, "
                      f"traceability {ratio:.2f}")

        return Decision(
            tick=self.tick,
            action=best.action,
            target=best.target,
            agent_cell=self.cell,
            reason=reason,
            chosen_cost=best.total,
            runner_up=second.action,
            margin=second.total - best.total,
            entropy=entropy,
            peak_cell=peak,
            peak_belief=float(self.belief.belief[peak]),
            peak_evidence_ratio=ratio,
            options=options,
        )

    # ------------------------------------------------------------------------------ acting
    def execute(self, decision: Decision) -> None:
        """Carry out one tick of the chosen action, and let time pass.

        Movement is one step per tick, so a decision to search somewhere eight steps away is
        re-examined eight times on the way. That is what makes the dither term necessary and
        what makes changing its mind a real, priced event rather than a free one.
        """
        action, target = decision.action, decision.target

        if action in TERMINAL:
            self.finished = True
            self.committed_target = None

        elif action == "hold_position":
            self.committed_target = None

        elif action in ("search", "flank"):
            self.committed_target = target
            if self.cell != target:
                self.cell = world.step_towards(self.cell, target)

        elif action == "retreat":
            self.committed_target = None
            self.cell = target if target else self.cell

        elif action == "call_reinforcements":
            self.reinforcements_called = True
            # Calling takes the tick; it does not also move.

        # Whatever it did, it looked at where it now stands, and time moved on.
        self.observe(self.look())
        self.belief.predict(steps=1)
        self.tick += 1
        self.ticks_since_evidence += 1


if __name__ == "__main__":
    agent = TacticalAgent(start=(11, 3), home_post=(11, 3))
    print("no evidence at all, first three decisions:")
    for _ in range(3):
        d = agent.decide()
        print("  ", d.action, d.target, "|", d.reason)
        agent.execute(d)

    print("\nafter hearing something at (4, 17):")
    agent.observe(Observation("sound", (4, 17), tick=agent.tick))
    for _ in range(6):
        d = agent.decide()
        print(f"   t{d.tick:<3d} {d.action:<20s} {str(d.target):<9s} "
              f"cost {d.chosen_cost:+7.2f}  next best {d.runner_up} (+{d.margin:.2f})")
        agent.execute(d)

    print("\nfull costing at this tick:")
    for opt in agent.decide().options:
        print("   ", opt.explain())
