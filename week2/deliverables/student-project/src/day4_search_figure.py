"""The day 4 figure: looking somewhere and finding nothing is evidence.

One scenario, two moments. A guard hears something across the level, walks to it, and sweeps
the area. It finds nothing. The picture is what that failed search does to what it believes.

The point is the second panel. The belief does not stop, and it does not stay put. The cell the
guard thought was most likely drops to a quarter of what it was, and the most likely cell in
the whole map moves out of the room it just searched.

Everything drawn here, including every number in the captions, is computed by `belief.py` at
render time. Nothing is typed in by hand.

    python src/day4_search_figure.py     # writes figures/day4-failed-search.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import belief as belief_mod
import world

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

NPC_START = (11, 3)    # where the guard is when it hears the sound
NOISE = (4, 17)        # where the sound came from
SEARCH_RADIUS = 5.0    # how far it can see once it gets there

# ------------------------------------------------------------------------------ the scenario
# The guard walks the whole way rather than teleporting: the number of ticks is the actual
# walking distance around the walls, so the belief has spread by exactly as much as the trip
# really costs. Inventing a smaller number here would quietly make the search look better.
WALK = int(world.distances_from(NOISE)[NPC_START])

grid = belief_mod.BeliefGrid()
grid.update(belief_mod.Observation("sound", NOISE, tick=0), agent_cell=NPC_START)
grid.predict(steps=WALK)

before = grid.belief.copy()
before_peak = grid.most_likely()
before_peak_p = float(before[before_peak])

searched_mask = world.visible_from(NOISE, radius=SEARCH_RADIUS)
searched_cells = tuple((int(r), int(c)) for r, c in zip(*np.where(searched_mask)))
before_mass = float(before[searched_mask].sum())

grid.update(
    belief_mod.Observation("search", NOISE, tick=WALK + 1, cells=searched_cells),
    agent_cell=NOISE,
)

after = grid.belief.copy()
after_peak = grid.most_likely()
after_mass = float(after[searched_mask].sum())
after_old_peak_p = float(after[before_peak])

# What one cell the guard did not look at gained, as a multiplier. Every unsearched cell gains
# the same factor, because renormalising scales them all equally.
unsearched_gain = (1.0 - after_mass) / (1.0 - before_mass)

# ------------------------------------------------------------------------------- the plot
# Sequential single hue, light to dark: the blue ramp from the reference palette, the same one
# the day 1 and day 3 figures use. One scale across both panels, because the whole figure is a
# before and after and per-panel scaling would hide the change it is about.
RAMP = ["#eef5fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
cmap = LinearSegmentedColormap.from_list("belief_blue", RAMP)
WALL_CMAP = LinearSegmentedColormap.from_list("w", ["#33332f", "#33332f"])
INK, MUTED, SURFACE = "#0b0b0b", "#6f6d68", "#fcfcfb"
ATTENTION = "#eb6834"   # where the guard went and looked

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": SURFACE})
FIG_W, FIG_H = 9.0, 8.5
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=150, facecolor=SURFACE)

vmax = max(float(np.nanmax(before)), float(np.nanmax(after))) * 1.02

panels = [
    {
        "belief": before,
        "peak": before_peak,
        "eyebrow": "Before",
        "caption": f"It walked {WALK} steps to the noise.\n"
                   f"The area it is about to sweep holds\n"
                   f"{before_mass:.0%} of everything it believes.",
    },
    {
        "belief": after,
        "peak": after_peak,
        "eyebrow": "After looking and seeing nothing",
        "caption": f"That same area now holds {after_mass:.0%}.\n"
                   f"Every cell it did not look at is\n"
                   f"{unsearched_gain:.1f} times more likely than it was.",
    },
]

LEFT, PANEL_W, COL_GAP = 0.055, 0.435, 0.040
PANEL_H = (PANEL_W * FIG_W) * (world.H / world.W) / FIG_H
PANEL_Y = 0.463

for i, panel in enumerate(panels):
    x = LEFT + i * (PANEL_W + COL_GAP)
    ax = fig.add_axes([x, PANEL_Y, PANEL_W, PANEL_H])

    shown = np.where(world.FREE, panel["belief"], np.nan)
    ax.imshow(shown, cmap=cmap, interpolation="nearest", vmin=0, vmax=vmax)
    ax.imshow(np.where(world.WALLS, 1.0, np.nan), cmap=WALL_CMAP, interpolation="nearest")

    ax.set_xticks(np.arange(-0.5, world.W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, world.H, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=0.7)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # The swept cells, outlined rather than tinted: a fill would compete with the belief
    # underneath it, which is the thing the reader is supposed to be comparing.
    for r, c in searched_cells:
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                               edgecolor=ATTENTION, linewidth=1.0, zorder=4))

    ax.plot(NOISE[1], NOISE[0], marker="*", markersize=12, color=ATTENTION,
            markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=6)
    ax.plot(panel["peak"][1], panel["peak"][0], marker="o", markersize=13,
            markerfacecolor="none", markeredgecolor=INK, markeredgewidth=2.0, zorder=6)

    fig.text(x, PANEL_Y + PANEL_H + 0.016, panel["eyebrow"], fontsize=13,
             fontweight="bold", color=INK, ha="left")
    fig.text(x, PANEL_Y - 0.030, panel["caption"], fontsize=11.5,
             color=INK, ha="left", va="top", linespacing=1.55)

fig.text(LEFT, 0.945, "Looking and finding nothing", fontsize=27,
         fontweight="bold", color=INK, ha="left")
fig.text(LEFT, 0.878,
         "Every player has watched a guard search a room, find nothing,\n"
         "and walk back into it ten seconds later. Here is the alternative.",
         fontsize=13.5, color="#4a4945", ha="left", va="top", linespacing=1.55)
fig.text(LEFT, 0.788,
         "★ the sound     ▢ swept and empty     ◯ where it now thinks you are",
         fontsize=11, color=MUTED, ha="left")

halo = [pe.withStroke(linewidth=3.0, foreground=SURFACE)]
fig.text(LEFT, 0.310,
         "The failed search did not end the chase. It aimed it.",
         fontsize=17.5, color=INK, ha="left", va="top", fontweight="bold",
         path_effects=halo)
fig.text(LEFT, 0.250,
         f"The cell it was walking towards fell from {before_peak_p:.1%} to {after_old_peak_p:.1%}. It is not zero,\n"
         "because the guard might have looked straight at you and missed. The probability that\n"
         "left those cells had to go somewhere, and it went to every corner it has not checked,\n"
         "which is why the black ring moves out of the room it just swept.",
         fontsize=11.5, color=MUTED, ha="left", va="top", linespacing=1.6)
fig.text(LEFT, 0.055,
         f"Day 4 of 7.  One run of the filter. Every cell and every number above is its output, "
         f"not drawn by hand.",
         fontsize=10, color=MUTED, ha="left")

path = OUT / "day4-failed-search.png"
fig.savefig(path)
print("wrote", path)
print(f"walk {WALK} ticks, searched {len(searched_cells)} cells")
print(f"searched area mass {before_mass:.3f} -> {after_mass:.3f}")
print(f"old peak {before_peak} {before_peak_p:.4f} -> {after_old_peak_p:.4f} "
      f"({after_old_peak_p / before_peak_p:.2f}x)")
print(f"peak moves {before_peak} -> {after_peak}")
print(f"unsearched cells gain x{unsearched_gain:.3f}")
