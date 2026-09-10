"""The world the agent lives in: a hand-made grid, its geometry, and the player in it.

Everything here is ground truth. The agent never imports from this module except through
the simulator, which decides what it is allowed to observe. Keeping that boundary strict is
the point: if the agent could read the player's position directly the whole experiment would
be meaningless, and it is the kind of leak that is easy to introduce by accident.

The map is the same one used for the day 1 figure. Hand-made on purpose: reproducible,
legible as a figure, and small enough that nobody can claim the result is an artifact of one
level's geometry.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------- the map
# '#' is a wall. 24 wide, 14 tall, so 336 cells, 201 of them walkable.
MAP_ROWS = [
    "########################",
    "#......#........#......#",
    "#......#........#......#",
    "#......#...######......#",
    "#..........#...........#",
    "####.###...#....########",
    "#......#...#...........#",
    "#......#...#####.#######",
    "#......#.......#.......#",
    "#..#####.......#.......#",
    "#......#...........#...#",
    "#......#############...#",
    "#......................#",
    "########################",
]

WALLS = np.array([[c == "#" for c in row] for row in MAP_ROWS])
FREE = ~WALLS
H, W = WALLS.shape
FREE_CELLS = [(int(r), int(c)) for r, c in zip(*np.where(FREE))]

ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def in_bounds(cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < H and 0 <= c < W


def is_free(cell: tuple[int, int]) -> bool:
    return in_bounds(cell) and bool(FREE[cell])


def neighbours(cell: tuple[int, int]) -> list[tuple[int, int]]:
    """Walkable orthogonal neighbours. Movement is four-way, not eight-way, so that
    'the player could have got there' means the same thing as 'mass can flow there'."""
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in ORTHOGONAL if is_free((r + dr, c + dc))]


# ------------------------------------------------------------------ chokepoints and doorways
def _find_chokepoints() -> list[tuple[int, int]]:
    """Free cells with exactly two walkable neighbours facing each other: a one-cell-wide
    passage. Doorways and corridors both match, and that is intended.

    First attempt at this called them doorways, which was wrong. The test run returned the
    whole of row 12 and a stretch of row 6, because every cell along a corridor has the same
    local shape as a door. The set is the right one for the purpose, the name was not: what
    the POI transition model wants is cells a player has to move *through*, and a corridor
    qualifies just as much as the door at the end of it.

    Derived from the map rather than hand-listed, so it cannot drift out of sync with the map
    and there is nothing here for me to quietly tune.
    """
    found = []
    for cell in FREE_CELLS:
        nbrs = neighbours(cell)
        if len(nbrs) != 2:
            continue
        (r1, c1), (r2, c2) = nbrs
        if r1 == r2 or c1 == c2:  # the two openings face each other
            found.append(cell)
    return found


CHOKEPOINTS = _find_chokepoints()

# A doorway is a chokepoint that opens directly into somewhere wider, so it is the mouth of a
# passage rather than the middle of one. Used only for labelling the figure. The transition
# model does not treat these differently from any other chokepoint.
DOORWAYS = [
    cell for cell in CHOKEPOINTS
    if any(nxt not in set(CHOKEPOINTS) for nxt in neighbours(cell))
]

# Points of interest for the POI-weighted transition model, which came from a design
# suggestion on r/roguelikedev (discussion-record.md R4). Chokepoints are derived above.
# Nothing is hand-added; if anything ever is, it must be listed here in the open rather than
# buried in a function, so a reader can see every place the prior is being nudged.
POINTS_OF_INTEREST = list(CHOKEPOINTS)


# ------------------------------------------------------------------------------ distances
def bfs_distances(source: tuple[int, int]) -> np.ndarray:
    """Walking distance in steps from `source` to every free cell, around walls.

    Returns an (H, W) array of floats with np.inf for walls and unreachable cells. Used for
    pathing, for the cost of moving somewhere, and for the sound likelihood, where distance
    through the map matters and straight-line distance does not.
    """
    dist = np.full((H, W), np.inf)
    if not is_free(source):
        return dist
    dist[source] = 0.0
    queue = deque([source])
    while queue:
        cell = queue.popleft()
        for nxt in neighbours(cell):
            if dist[nxt] == np.inf:
                dist[nxt] = dist[cell] + 1.0
                queue.append(nxt)
    return dist


_DISTANCE_CACHE: dict[tuple[int, int], np.ndarray] = {}


def distances_from(source: tuple[int, int]) -> np.ndarray:
    """Cached bfs_distances. The map never changes during a run, so every distance field is
    computed at most once. This is what keeps a 50 case sweep cheap."""
    if source not in _DISTANCE_CACHE:
        _DISTANCE_CACHE[source] = bfs_distances(source)
    return _DISTANCE_CACHE[source]


def step_towards(start: tuple[int, int], goal: tuple[int, int]) -> tuple[int, int]:
    """One step along a shortest path. Returns `start` if the goal is unreachable or already
    reached. Ties break by the neighbour order in ORTHOGONAL, which is deterministic, so two
    runs with the same seed produce identical paths."""
    if start == goal:
        return start
    field_ = distances_from(goal)
    if field_[start] == np.inf:
        return start
    best, best_d = start, field_[start]
    for nxt in neighbours(start):
        if field_[nxt] < best_d:
            best, best_d = nxt, field_[nxt]
    return best


# ---------------------------------------------------------------------------- line of sight
def line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """Cells on the straight line from a to b, endpoints included. Bresenham."""
    (r0, c0), (r1, c1) = a, b
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    cells = []
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if (r, c) == (r1, c1):
            return cells
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc


def has_line_of_sight(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True if nothing solid sits between a and b. Endpoints are not required to be free,
    so this can be asked about a wall cell without lying."""
    return all(is_free(cell) for cell in line_cells(a, b)[1:-1]) and is_free(a) and is_free(b)


_VISIBILITY_CACHE: dict[tuple[tuple[int, int], float], np.ndarray] = {}


def visible_from(cell: tuple[int, int], radius: float = 7.0) -> np.ndarray:
    """Cached wrapper. See `_visible_from` for what it computes and why it is shaped this way.

    The map never changes during a run, so a visibility mask is computed at most once per
    (cell, radius) pair. Without this the agent recomputes line of sight to all 201 cells
    several times per tick, which dominated the runtime of the sweep: caching is what makes a
    multi-seed run affordable, and lesson 6 says a single seed is not a result.

    The returned array is shared, not copied, so callers must treat it as read-only. Every
    caller currently uses it as a mask, which is why that is worth the speed.
    """
    key = (cell, radius)
    if key not in _VISIBILITY_CACHE:
        _VISIBILITY_CACHE[key] = _visible_from(cell, radius)
    return _VISIBILITY_CACHE[key]


def _visible_from(cell: tuple[int, int], radius: float = 7.0) -> np.ndarray:
    """Boolean mask of free cells the agent can see from `cell`.

    A plain radius plus line of sight, with no facing and no vision cone. [ASSUMED] and
    deliberately generous: a cone would add a heading to the agent's state and a second
    parameter to defend, and the scope decision was to spend complexity on the cost function
    instead. Recorded as a limitation, not hidden.
    """
    mask = np.zeros((H, W), dtype=bool)
    dist = distances_from(cell)
    for target in FREE_CELLS:
        if dist[target] <= radius and has_line_of_sight(cell, target):
            mask[target] = True
    return mask


# ------------------------------------------------------------------------------- the player
@dataclass
class Player:
    """The hidden state. The agent never sees this object.

    The player is not adversarial yet: they head for a goal and re-pick a new one when they
    reach it, with an occasional random step so the path is not perfectly predictable. A
    genuinely evasive player who reasons about the agent's belief is the obvious next step
    and is deferred, with the reason recorded in research-file.md.
    """

    position: tuple[int, int]
    rng: np.random.Generator
    goal: tuple[int, int] | None = None
    wander: float = 0.15          # [ASSUMED] chance of a random step instead of a goal step
    trail: list[tuple[int, int]] = field(default_factory=list)

    def _pick_goal(self) -> tuple[int, int]:
        """Head somewhere reachable and not adjacent. Points of interest are twice as likely
        as ordinary cells, because a player crossing a level does move through doorways more
        than through the middle of a room. [ASSUMED], and it is the same bias the POI
        transition model assumes, so the two must never be compared as if independent.
        """
        weights = np.ones(len(FREE_CELLS))
        poi = set(POINTS_OF_INTEREST)
        for i, cell in enumerate(FREE_CELLS):
            if cell in poi:
                weights[i] = 2.0
        weights /= weights.sum()
        while True:
            goal = FREE_CELLS[int(self.rng.choice(len(FREE_CELLS), p=weights))]
            if distances_from(goal)[self.position] > 3:
                return goal

    def step(self) -> tuple[int, int]:
        """Advance one tick and return the new position."""
        self.trail.append(self.position)
        if self.goal is None or self.position == self.goal:
            self.goal = self._pick_goal()
        if self.rng.random() < self.wander:
            options = neighbours(self.position)
            self.position = options[int(self.rng.integers(len(options)))]
        else:
            self.position = step_towards(self.position, self.goal)
        return self.position


def describe() -> str:
    return (
        f"grid {W} by {H}, {len(FREE_CELLS)} walkable cells, "
        f"{len(CHOKEPOINTS)} chokepoints, {len(DOORWAYS)} of them doorways"
    )


if __name__ == "__main__":
    print(describe())
    print("chokepoints:", len(CHOKEPOINTS), "of which doorways:", DOORWAYS)
    rng = np.random.default_rng(0)
    p = Player(position=(12, 2), rng=rng)
    path = [p.step() for _ in range(10)]
    print("player walked:", path)
    print("LOS (12,2) -> (12,20):", has_line_of_sight((12, 2), (12, 20)))
    print("visible cells from (11,3):", int(visible_from((11, 3)).sum()))
