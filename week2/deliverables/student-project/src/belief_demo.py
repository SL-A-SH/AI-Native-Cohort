"""First belief map for the NPC tactical belief agent.

Minimal and honest: a grid with walls, one noisy observation, and a diffusion step that
respects walls. No policy yet, no actions, no evaluation. This exists to render the thing
the whole project is about, which is what the agent believes rather than what it knows.

    python src/belief_demo.py        # writes figures/day1-belief-map.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------- the map
# 1 marks a wall. Hand-made so the geometry is legible and reproducible.
MAP = [
    "########################",
    "#......#........#......#",
    "#......#........#......#",
    "#......#...######......#",
    "#..........#.....……....#",
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
MAP = [row.replace("…", ".") for row in MAP]
walls = np.array([[c == "#" for c in row] for row in MAP])
H, W = walls.shape
free = ~walls

NPC = (11, 3)        # row, col. Where the guard is standing.
NOISE = (4, 17)      # row, col. Where a sound came from.

# --------------------------------------------------------------------- the belief model
def normalise(b):
    b = np.where(free, b, 0.0)
    return b / b.sum()

def observe(belief, source, sigma=2.2, floor=0.04):
    """A sound was heard from roughly `source`. Nearby cells explain it better.

    `floor` is the false-positive allowance: the noise might not have been the player, so
    no cell is ever driven to exactly zero by one observation.
    """
    rr, cc = np.mgrid[0:H, 0:W]
    d2 = (rr - source[0]) ** 2 + (cc - source[1]) ** 2
    likelihood = np.exp(-d2 / (2 * sigma ** 2)) + floor
    return normalise(belief * likelihood)

def diffuse(belief, steps=1, stay=0.55):
    """Time passes and the player may have moved. Mass spreads to free neighbours only,
    so probability flows around walls rather than through them."""
    for _ in range(steps):
        nxt = belief * stay
        share = belief * (1 - stay)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted = np.roll(np.roll(share, dr, axis=0), dc, axis=1)
            blocked = np.roll(np.roll(free, dr, axis=0), dc, axis=1)
            nxt += np.where(free & blocked, shifted / 4.0, 0.0)
        belief = normalise(nxt)
    return belief

belief = normalise(free.astype(float))   # knows nothing except where walls are
belief = observe(belief, NOISE)          # hears something
belief = diffuse(belief, steps=6)        # six ticks pass, the player could have moved

# ------------------------------------------------------------------------------ the plot
# Sequential single hue, light to dark, from a validated palette.
RAMP = ["#eef5fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
cmap = LinearSegmentedColormap.from_list("belief_blue", RAMP)
INK, MUTED, WALL, SURFACE = "#0b0b0b", "#6f6d68", "#33332f", "#fcfcfb"

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": SURFACE})
fig = plt.figure(figsize=(7.2, 9.0), dpi=150, facecolor=SURFACE)
ax = fig.add_axes([0.06, 0.325, 0.88, 0.415])

shown = np.where(free, belief, np.nan)
ax.imshow(shown, cmap=cmap, interpolation="nearest",
          vmin=0, vmax=np.nanmax(shown) * 1.02)
ax.imshow(np.where(walls, 1.0, np.nan), cmap=LinearSegmentedColormap.from_list(
    "w", [WALL, WALL]), interpolation="nearest")

ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=1.1)
ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
for s in ax.spines.values():
    s.set_visible(False)

ax.plot(NOISE[1], NOISE[0], marker="*", markersize=19, color="#eb6834",
        markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=5)
halo = [pe.withStroke(linewidth=3.5, foreground=SURFACE)]
ax.annotate("sound heard here", (NOISE[1], NOISE[0]), xytext=(0, 17),
            textcoords="offset points", ha="center", color="#c4501f",
            fontsize=10.5, fontweight="bold", path_effects=halo)
ax.plot(NPC[1], NPC[0], marker="o", markersize=13, color=INK,
        markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=5)
ax.annotate("the guard", (NPC[1], NPC[0]), xytext=(0, -23),
            textcoords="offset points", ha="center", color=INK,
            fontsize=10.5, fontweight="bold", path_effects=halo)

fig.text(0.06, 0.950, "Where the guard thinks you are", fontsize=25,
         fontweight="bold", color=INK, ha="left")
fig.text(0.06, 0.908, "It heard one noise. Six seconds passed. It has not seen you.",
         fontsize=13.5, color="#4a4945", ha="left")

fig.text(0.06, 0.830, "Darker cells are more probable. Belief flows around walls,\n"
                      "not through them, because you had to walk somewhere real.",
         fontsize=11.5, color=MUTED, ha="left", linespacing=1.5)

fig.text(0.06, 0.205,
         "The hard part is not making this accurate.\n"
         "An accurate guard walks straight to you, every time.\n"
         "That is a bug, not a feature.",
         fontsize=15, color=INK, ha="left", linespacing=1.55, fontweight="bold")
fig.text(0.06, 0.062,
         "Day 1 of 7. Building a tactical AI whose objective is to lose convincingly.\n"
         "Grid, diffusion and observation model are real output from the code, not a mockup.",
         fontsize=10.5, color=MUTED, ha="left", linespacing=1.6)

path = OUT / "day1-belief-map.png"
fig.savefig(path)
print("wrote", path)
print("belief sums to", round(float(belief.sum()), 6))
print("most probable cell", np.unravel_index(np.argmax(belief), belief.shape))
