"""The belief: a probability distribution over where the player is, and the rules that move it.

Two operations, alternating, which is what makes this a Bayes filter rather than a one-shot
posterior:

    predict   time passed, so the player may have moved. Mass spreads along walkable cells.
    update    something was observed, so some cells explain it better than others.

Every number in PARAMS is invented. All of them are labelled [ASSUMED] and none of them has
been tuned against a result, because tuning a prior until the answer looks good is the
failure this project is supposed to avoid. Thresholds are derived from the cost matrix in
agent.py instead.

Two things here came directly out of the Reddit threads and are marked where they appear:
the POI-weighted transition model (discussion-record.md R4) and the treatment of a failed
search as evidence (R5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import world

# ---------------------------------------------------------------------------- the parameters
# Every value here is [ASSUMED]. They are collected in one dict rather than scattered through
# the code so that a reader can see the whole set of invented numbers at once, and so the
# sensitivity sweep has one place to reach into.
PARAMS: dict[str, float] = {
    # transition model
    "stay": 0.55,            # [ASSUMED] chance the player did not leave the cell this tick
    "poi_weight": 2.0,       # [ASSUMED] pull toward chokepoints, policy B only

    # observation model, per event type
    "sighting_hit": 0.95,    # [ASSUMED] P(report a sighting | player is in the sighted cell)
    "sighting_sigma": 1.0,   # [ASSUMED] cells of slop on a sighting. Tight: you saw them.
    "sighting_floor": 0.01,  # [ASSUMED] P(report a sighting | player is elsewhere)
    "sound_sigma": 2.2,      # [ASSUMED] how sharply a sound localises, in walking steps
    "sound_floor": 0.04,     # [ASSUMED] a noise that was not the player
    "door_floor": 0.15,      # [ASSUMED] a door can move for reasons that are not the player
    "shot_sigma": 3.5,       # [ASSUMED] a gunshot bearing is loud but vague
    "shot_floor": 0.02,      # [ASSUMED]

    # negative information (R5)
    "search_miss": 0.15,     # [ASSUMED] P(search a cell and miss | player is in it)
}


@dataclass(frozen=True)
class Observation:
    """One perception event. `kind` selects the likelihood, `cell` is where it came from.

    Note what this does not carry: the truth. An Observation is what the agent was told, and
    a false positive is an Observation like any other. The simulator decides what to emit;
    the agent cannot tell the difference, which is the point.
    """

    kind: str                      # sighting | sound | door | shot | search
    cell: tuple[int, int]
    tick: int
    cells: tuple[tuple[int, int], ...] = ()   # for `search`, everywhere that was looked at


# --------------------------------------------------------------------------- the belief grid
class BeliefGrid:
    """A distribution over free cells, plus the bookkeeping needed to ask where it came from.

    Alongside the belief it carries `null_belief`: the same filter run with every *positive*
    observation withheld, so it holds only diffusion and the results of failed searches. The
    two together answer a question a plain posterior cannot, which is whether the agent is
    moving because something happened or because the smear happens to be thickest there.

    That question came from a player on r/roguelikedev (discussion-record.md R6), who pointed
    out that an omniscient psionic feels fair while an omniscient guard does not, so the
    complaint was never about how much the NPC knows but about whether the player can
    attribute the knowledge to anything.
    """

    def __init__(self, poi_weighted: bool = False, params: dict[str, float] | None = None):
        self.params = dict(PARAMS if params is None else params)
        self.poi_weighted = poi_weighted
        self.belief = self._uniform()
        self.null_belief = self._uniform()
        self._transition = self._build_transition()

    # ------------------------------------------------------------------------- construction
    @staticmethod
    def _uniform() -> np.ndarray:
        b = world.FREE.astype(float)
        return b / b.sum()

    def _build_transition(self) -> np.ndarray:
        """Precompute the per-cell weights used by predict().

        Policy A leaves this flat. Policy B raises the weight on chokepoints, so mass flows
        toward the cells a player crossing the map has to pass through. Built once, because
        the map does not change.
        """
        weights = np.ones((world.H, world.W))
        if self.poi_weighted:
            for cell in world.POINTS_OF_INTEREST:
                weights[cell] = self.params["poi_weight"]
        return np.where(world.FREE, weights, 0.0)

    # ------------------------------------------------------------------------------ helpers
    @staticmethod
    def _normalise(b: np.ndarray) -> np.ndarray:
        b = np.where(world.FREE, b, 0.0)
        total = b.sum()
        if total <= 0:
            # Every hypothesis was ruled out, which means the model is wrong rather than the
            # player having vanished. Fall back to uniform and let the run record it, instead
            # of dividing by zero and producing NaNs that quietly poison the results file.
            return BeliefGrid._uniform()
        return b / total

    # ------------------------------------------------------------------------- predict step
    def predict(self, steps: int = 1) -> None:
        """Time passes. Mass spreads to walkable neighbours, weighted by the transition model.

        Mass flows around walls rather than through them, because the player had to walk
        somewhere real to get there. This is the step a last-known-position model does not
        have, and it is the whole difference between remembering and guessing forward.
        """
        for _ in range(steps):
            self.belief = self._diffuse(self.belief)
            self.null_belief = self._diffuse(self.null_belief)

    def _diffuse(self, b: np.ndarray) -> np.ndarray:
        stay = self.params["stay"]
        moving = b * (1.0 - stay)
        nxt = b * stay
        # Each cell hands its moving mass to its walkable neighbours, split in proportion to
        # the transition weights, so no mass is created or lost at a wall.
        weight_sum = np.zeros((world.H, world.W))
        contributions = []
        for dr, dc in world.ORTHOGONAL:
            shifted_free = np.roll(np.roll(world.FREE, -dr, axis=0), -dc, axis=1)
            shifted_weight = np.roll(np.roll(self._transition, -dr, axis=0), -dc, axis=1)
            usable = world.FREE & shifted_free
            w = np.where(usable, shifted_weight, 0.0)
            weight_sum += w
            contributions.append((dr, dc, w))

        # A cell with no walkable neighbours keeps everything rather than dividing by zero.
        safe = np.where(weight_sum > 0, weight_sum, 1.0)
        stuck = moving * (weight_sum == 0)
        for dr, dc, w in contributions:
            share = moving * w / safe
            nxt += np.roll(np.roll(share, dr, axis=0), dc, axis=1)
        return self._normalise(nxt + stuck)

    # -------------------------------------------------------------------------- update step
    def update(self, obs: Observation, agent_cell: tuple[int, int]) -> None:
        """Apply one observation. Positive events move both beliefs apart; a failed search
        moves them together, because it is evidence the agent has whether or not the player
        ever did anything."""
        if obs.kind == "search":
            like = self._search_likelihood(obs)
            self.belief = self._normalise(self.belief * like)
            self.null_belief = self._normalise(self.null_belief * like)
            return

        like = self._positive_likelihood(obs, agent_cell)
        self.belief = self._normalise(self.belief * like)
        # null_belief deliberately does not see this. It is the counterfactual agent that was
        # told nothing, and the gap between the two is what makes an action attributable.

    def _positive_likelihood(self, obs: Observation, agent_cell: tuple[int, int]) -> np.ndarray:
        p = self.params
        if obs.kind == "sighting":
            # A sighting localises. It is the sharpest evidence the agent ever gets, and
            # `sighting_sigma` is deliberately much tighter than the sound model: seeing
            # someone tells you roughly which cell, hearing them tells you roughly which room.
            #
            # The first version of this ignored `obs.cell` entirely and spread the likelihood
            # across every cell visible from the agent, which read as "the player is somewhere
            # in my field of view". That was consistent with its own docstring and consistent
            # inside the experiment, since a sighting is only ever emitted when line of sight
            # exists, so nothing looked broken. It was still throwing away the location of the
            # single most precise observation the agent receives. Found while building the
            # decision record, where injecting a sighting the guard could not possibly have
            # made produced a confident update about the wrong side of the map.
            #
            # Whether a sighting *can* happen is the simulator's business and it checks line of
            # sight before emitting one. The likelihood's only job is to say where the player
            # must have been for that report to arrive, so it no longer consults visibility.
            dist = world.distances_from(obs.cell)
            finite = np.where(np.isfinite(dist), dist, 1e6)
            sigma = p["sighting_sigma"]
            like = (p["sighting_hit"] * np.exp(-(finite ** 2) / (2 * sigma ** 2))
                    + p["sighting_floor"])
            return np.where(world.FREE, like, 0.0)

        if obs.kind == "door":
            # A door moved. The player is probably at it, but doors move for other reasons.
            like = np.full((world.H, world.W), p["door_floor"])
            for cell in [obs.cell, *world.neighbours(obs.cell)]:
                like[cell] = 1.0
            return np.where(world.FREE, like, 0.0)

        sigma = p["sound_sigma"] if obs.kind == "sound" else p["shot_sigma"]
        floor = p["sound_floor"] if obs.kind == "sound" else p["shot_floor"]
        # Distance is measured through the map, not across it, so a cell two steps away
        # through a wall is as unlikely as the twelve-step walk that actually reaches it.
        dist = world.distances_from(obs.cell)
        finite = np.where(np.isfinite(dist), dist, 1e6)
        like = np.exp(-(finite ** 2) / (2 * sigma ** 2)) + floor
        return np.where(world.FREE, like, 0.0)

    def _search_likelihood(self, obs: Observation) -> np.ndarray:
        """Looked, saw nothing. This is evidence, and it is the piece most often left out.

        A searched cell is not ruled out, it is multiplied down by the miss rate, because the
        agent can look straight at a cell and fail to see the player in it. The mass that
        leaves those cells lands on everywhere it did not look, which is what stops the agent
        walking back into a room it has already cleared.

        A developer on r/roguelikedev showed me this is already done in practice, but at room
        scope as a rule that ends the pursuit, rather than at cell scope as a likelihood
        (discussion-record.md R5). The difference is what happens to the belief afterwards,
        and that difference is the thing this project is arguing about.
        """
        like = np.ones((world.H, world.W))
        for cell in obs.cells or (obs.cell,):
            like[cell] = self.params["search_miss"]
        return np.where(world.FREE, like, 0.0)

    # --------------------------------------------------------------------------- readouts
    def most_likely(self) -> tuple[int, int]:
        return tuple(int(x) for x in np.unravel_index(np.argmax(self.belief), self.belief.shape))

    def entropy(self) -> float:
        """Shannon entropy in bits. The agent's own measure of how little it knows, and the
        input to the one human reasoning function in agent.py."""
        b = self.belief[world.FREE]
        b = b[b > 0]
        return float(-(b * np.log2(b)).sum())

    def max_entropy(self) -> float:
        return float(np.log2(len(world.FREE_CELLS)))

    def normalised_entropy(self) -> float:
        """Entropy as a fraction of the worst case, so a threshold on it means the same thing
        on any map. 1.0 is knowing nothing at all."""
        return self.entropy() / self.max_entropy()

    def evidence_ratio(self, cell: tuple[int, int]) -> float:
        """How much of this cell's probability is owed to something that actually happened.

        Above 1.0 means positive observations put mass here. At or below 1.0 means the agent
        would believe this much anyway, from diffusion and from what it failed to find, so an
        action taken on it is not attributable to anything the player did.

        This is the computable half of the legibility idea from R6. It is a proxy, it is not
        a measurement of believability, and it must be described that way everywhere.
        """
        null = float(self.null_belief[cell])
        if null <= 0:
            return 0.0
        return float(self.belief[cell]) / null

    def mass_in(self, cells) -> float:
        return float(sum(self.belief[cell] for cell in cells))

    def copy(self) -> "BeliefGrid":
        clone = BeliefGrid.__new__(BeliefGrid)
        clone.params = dict(self.params)
        clone.poi_weighted = self.poi_weighted
        clone.belief = self.belief.copy()
        clone.null_belief = self.null_belief.copy()
        clone._transition = self._transition
        return clone


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for poi in (False, True):
        g = BeliefGrid(poi_weighted=poi)
        label = "policy B (POI)" if poi else "policy A (uniform)"
        g.update(Observation("sound", (4, 17), tick=0), agent_cell=(11, 3))
        g.predict(steps=6)
        print(f"{label:16s} sums to {g.belief.sum():.6f}  "
              f"peak {g.most_likely()}  H {g.normalised_entropy():.3f}  "
              f"evidence_ratio at peak {g.evidence_ratio(g.most_likely()):.2f}")

    # A failed search must move mass off the searched cells and must not zero them.
    g = BeliefGrid()
    g.update(Observation("sound", (4, 17), tick=0), agent_cell=(11, 3))
    before = g.belief[(4, 17)]
    cells = tuple((int(r), int(c)) for r, c in zip(*np.where(world.visible_from((4, 17), 2.0))))
    g.update(Observation("search", (4, 17), tick=1, cells=cells), agent_cell=(4, 17))
    after = g.belief[(4, 17)]
    print(f"searched {len(cells)} cells: p(4,17) {before:.4f} -> {after:.4f}, "
          f"still nonzero: {after > 0}, sums to {g.belief.sum():.6f}")
