"""Three agents to measure the belief agent against.

The point of a baseline is to be beatable only if the thing being tested is actually worth
something, so each of these is written to be the strongest fair version of itself rather than
a strawman. Two of the three were described to me by practitioners; the third is the one that
every published source says games actually ship.

    LastKnownPosition     remember the last place you saw them, go there, give up      (R5)
    ScriptedSweep         the same, plus a timed sweep of what is nearby               (R3)
    DegradedOmniscience   the true position, behind a vision cone and a delay          (R2)

**`DegradedOmniscience` is the one that matters.** It receives ground truth, gated. It is
therefore *better informed* than the belief agent, not worse, and it is what the industry
actually builds (see [A-07] in `research-file.md`). Beating last-known-position proves very
little. Failing to beat degraded omniscience would mean the filter is not earning its place,
and that is a result worth publishing either way.

None of these three holds a probability distribution, so none of them can report an attribution
score for its own actions. The harness measures that externally with a shadow filter, so the
same yardstick is applied to every agent including the ones that cannot compute it themselves.
"""

from __future__ import annotations

import numpy as np

import world
from agent import Decision
from belief import Observation


def _decision(tick, action, target, cell, reason) -> Decision:
    """A Decision with the fields a belief agent fills in left empty.

    Same shape for every agent so one results file covers all four, and so nothing has to
    branch on agent type when reading it back.
    """
    return Decision(
        tick=tick,
        action=action,
        target=target,
        agent_cell=cell,
        reason=reason,
        chosen_cost=float("nan"),
        runner_up="",
        margin=float("nan"),
        entropy=float("nan"),
        peak_cell=target if target else cell,
        peak_belief=float("nan"),
        peak_evidence_ratio=float("nan"),
        options=[],
    )


class LastKnownPosition:
    """Baseline A. A remembered point and a timestamp, which is the shipped discrete model.

    This is Isla's Halo 2 belief in its simplest honest form and matches what a developer on
    r/roguelikedev described (R5): mark the last tile seen, path to it, and if the trail goes
    cold, go back to your post. It has no motion model, so it cannot guess forward, and it has
    no notion of negative information beyond giving up once it has looked.

    `patience` is the give-up timer. Configurable per NPC type in his game, which is where the
    default came from.
    """

    name = "last_known_position"

    def __init__(self, start, home_post, patience: int = 12, sight_radius: float = 7.0):
        self.cell = start
        self.home_post = home_post
        self.patience = patience          # [ASSUMED] ticks of fruitless looking before quitting
        self.sight_radius = sight_radius
        self.tick = 0
        self.finished = False
        self.last_known: tuple[int, int] | None = None
        self.waited = 0

    def observe(self, obs: Observation) -> None:
        # Any positive event overwrites the memory. There is nothing to combine it with, which
        # is exactly the limitation being tested.
        if obs.kind in ("sighting", "sound", "door", "shot"):
            self.last_known = obs.cell
            self.waited = 0

    def decide(self) -> Decision:
        if self.last_known is None:
            return _decision(self.tick, "hold_position", None, self.cell, "nothing remembered")
        if self.cell != self.last_known:
            return _decision(self.tick, "search", self.last_known, self.cell,
                             "moving to last known position")
        if self.waited < self.patience:
            return _decision(self.tick, "search", self.last_known, self.cell,
                             f"looking, {self.waited}/{self.patience}")
        return _decision(self.tick, "resume_post", self.home_post, self.cell,
                         "trail went cold, returning to post")

    def execute(self, decision: Decision) -> None:
        if decision.action == "resume_post":
            self.finished = True
        elif decision.target and self.cell != decision.target:
            self.cell = world.step_towards(self.cell, decision.target)
        else:
            self.waited += 1
        self.tick += 1


class ScriptedSweep:
    """Baseline C. Go to the last known point, then sweep what is nearby on a timer.

    This is the "cheap approximation" a commenter on r/GameAI argued would look identical on
    screen for a tenth of the work (R3). If he is right, the belief machinery is not earning
    its place, and that is a finding rather than an embarrassment.

    It is deliberately not stupid: it sweeps a genuine ring of chokepoints outward from the last
    known point, in nearest-first order, which is a reasonable approximation of what a person
    would do. What it cannot do is weigh one candidate against another, because it has no
    quantity to weigh with.
    """

    name = "scripted_sweep"

    def __init__(self, start, home_post, sweep_limit: int = 3, dwell: int = 4,
                 sight_radius: float = 7.0):
        self.cell = start
        self.home_post = home_post
        self.sweep_limit = sweep_limit    # [ASSUMED] rooms checked before giving up
        self.dwell = dwell                # [ASSUMED] ticks spent looking at each
        self.sight_radius = sight_radius
        self.tick = 0
        self.finished = False
        self.last_known: tuple[int, int] | None = None
        self.queue: list[tuple[int, int]] = []
        self.swept = 0
        self.waited = 0

    def observe(self, obs: Observation) -> None:
        if obs.kind in ("sighting", "sound", "door", "shot"):
            self.last_known = obs.cell
            self.queue = []
            self.swept = 0
            self.waited = 0

    def _build_queue(self) -> None:
        """Nearest chokepoints to the last known point, nearest first."""
        if self.last_known is None:
            return
        dist = world.distances_from(self.last_known)
        reachable = [c for c in world.POINTS_OF_INTEREST
                     if np.isfinite(dist[c]) and 0 < dist[c] <= 10]
        reachable.sort(key=lambda c: dist[c])
        self.queue = reachable[:self.sweep_limit]

    def decide(self) -> Decision:
        if self.last_known is None:
            return _decision(self.tick, "hold_position", None, self.cell, "nothing remembered")

        if self.cell != self.last_known and self.swept == 0 and not self.queue:
            return _decision(self.tick, "search", self.last_known, self.cell,
                             "moving to last known position")

        if self.waited < self.dwell and not self.queue:
            return _decision(self.tick, "search", self.cell, self.cell,
                             f"looking here, {self.waited}/{self.dwell}")

        if not self.queue and self.swept < self.sweep_limit:
            self._build_queue()

        if self.queue:
            target = self.queue[0]
            return _decision(self.tick, "search", target, self.cell,
                             f"sweeping room {self.swept + 1}/{self.sweep_limit}")

        return _decision(self.tick, "resume_post", self.home_post, self.cell,
                         f"swept {self.swept} rooms, giving up")

    def execute(self, decision: Decision) -> None:
        if decision.action == "resume_post":
            self.finished = True
        elif decision.action == "hold_position":
            pass
        elif decision.target and self.cell != decision.target:
            self.cell = world.step_towards(self.cell, decision.target)
        else:
            self.waited += 1
            if self.queue and self.cell == self.queue[0]:
                self.queue.pop(0)
                self.swept += 1
                self.waited = 0
        self.tick += 1


class DegradedOmniscience:
    """Baseline B. The true position, gated by sight, a reaction delay and an error term.

    The strongest baseline and the one the paper turns on. A commenter on r/GameAI put it
    plainly: game AI has no sensor noise by default, because the engine knows the player's
    position to the pixel, so what actually gets implemented is an *artificial reduction of
    accuracy* (R2). The Alien: Isolation director is the published version of the same idea.

    So this is not a weaker agent. It knows more than the belief agent ever does. What it does
    not have is any way to reason about where the player might have gone: when the cone breaks,
    it has a stale point and nothing else.

    **This is the only class in the project permitted to receive ground truth**, and it arrives
    through `reveal`, which nothing else implements. Keeping that on one clearly named method
    is what stops the leak happening by accident somewhere it would be invisible.
    """

    name = "degraded_omniscience"

    def __init__(self, start, home_post, rng: np.random.Generator,
                 reaction_delay: int = 3, position_error: float = 1.5,
                 patience: int = 12, sight_radius: float = 7.0):
        self.cell = start
        self.home_post = home_post
        self.rng = rng
        self.reaction_delay = reaction_delay    # [ASSUMED] ticks between seeing and acting
        self.position_error = position_error    # [ASSUMED] cells of slop on the reading
        self.patience = patience                # [ASSUMED]
        self.sight_radius = sight_radius
        self.tick = 0
        self.finished = False
        self.pending: list[tuple[int, tuple[int, int]]] = []
        self.last_known: tuple[int, int] | None = None
        self.waited = 0

    def observe(self, obs: Observation) -> None:
        # It hears things like everyone else, but a heard event is much weaker than a look.
        if obs.kind in ("sound", "door", "shot") and self.last_known is None:
            self.last_known = obs.cell
            self.waited = 0

    def reveal(self, true_cell: tuple[int, int]) -> None:
        """Ground truth, gated. Called by the harness for this agent only.

        The gate is line of sight plus range, which is the vision cone in its most generous
        form. What gets through is queued rather than used, so the agent acts on where the
        player *was* `reaction_delay` ticks ago, then blurred by the error term. That is the
        subtractive design: start from perfect and take accuracy away.
        """
        if not world.has_line_of_sight(self.cell, true_cell):
            return
        if world.distances_from(self.cell)[true_cell] > self.sight_radius:
            return

        blurred = true_cell
        if self.position_error > 0:
            for _ in range(int(self.rng.poisson(self.position_error))):
                options = world.neighbours(blurred)
                if options:
                    blurred = options[int(self.rng.integers(len(options)))]
        self.pending.append((self.tick + self.reaction_delay, blurred))

    def decide(self) -> Decision:
        ready = [cell for due, cell in self.pending if due <= self.tick]
        if ready:
            self.last_known = ready[-1]
            self.pending = [(d, c) for d, c in self.pending if d > self.tick]
            self.waited = 0

        if self.last_known is None:
            return _decision(self.tick, "hold_position", None, self.cell, "no contact")
        if self.cell != self.last_known:
            return _decision(self.tick, "search", self.last_known, self.cell,
                             "closing on last reading")
        if self.waited < self.patience:
            return _decision(self.tick, "search", self.last_known, self.cell,
                             f"holding on last reading, {self.waited}/{self.patience}")
        return _decision(self.tick, "resume_post", self.home_post, self.cell,
                         "contact lost, returning to post")

    def execute(self, decision: Decision) -> None:
        if decision.action == "resume_post":
            self.finished = True
        elif decision.target and self.cell != decision.target:
            self.cell = world.step_towards(self.cell, decision.target)
        else:
            self.waited += 1
        self.tick += 1


BASELINES = {
    "last_known_position": LastKnownPosition,
    "scripted_sweep": ScriptedSweep,
    "degraded_omniscience": DegradedOmniscience,
}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for name, cls in BASELINES.items():
        kwargs = {"rng": rng} if name == "degraded_omniscience" else {}
        b = cls(start=(11, 3), home_post=(11, 3), **kwargs)
        b.observe(Observation("sound", (4, 17), tick=0))
        actions = []
        for _ in range(60):
            d = b.decide()
            actions.append(d.action)
            b.execute(d)
            if b.finished:
                break
        print(f"{name:22s} {len(actions):3d} ticks, ended {actions[-1]:14s} at {b.cell}")
