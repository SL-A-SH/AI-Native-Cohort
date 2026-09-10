"""The day 5 figure: catching an AI guessing, as a division you can see.

Three maps with a division sign and an equals sign between them, because the whole idea is one
piece of arithmetic:

    what it believes  ÷  what it would think anyway  =  the part you caused

The operators do the explaining. An earlier version of this figure labelled the same three
panels in prose and needed a paragraph underneath to say how they related; nobody reads a
paragraph under a picture. A reader who has seen a division sign before already knows what this
means before reading a word.

Everything drawn, including every number, is computed at render time.

    python src/day5_attribution_figure.py     # writes figures/day5-attribution.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import belief as belief_mod
import world

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

NPC_START = (11, 3)
NOISE = (4, 17)
SEARCH_RADIUS = 5.0

# ------------------------------------------------------------------------------ the scenario
# The same run as the day 4 figure and video, so the week is one continuous story.
grid = belief_mod.BeliefGrid()
grid.update(belief_mod.Observation("sound", NOISE, tick=0), agent_cell=NPC_START)
WALK = int(world.distances_from(NOISE)[NPC_START])
grid.predict(steps=WALK)

searched = world.visible_from(NOISE, radius=SEARCH_RADIUS)
searched_cells = tuple((int(r), int(c)) for r, c in zip(*np.where(searched)))
grid.update(
    belief_mod.Observation("search", NOISE, tick=WALK + 1, cells=searched_cells),
    agent_cell=NOISE,
)

belief = grid.belief.copy()
null_belief = grid.null_belief.copy()

ratio = np.full((world.H, world.W), np.nan)
for r, c in world.FREE_CELLS:
    ratio[r, c] = grid.evidence_ratio((r, c))

below = int(np.nansum(ratio < 1.0))
total_cells = len(world.FREE_CELLS)
hot_cell = tuple(int(x) for x in np.unravel_index(np.nanargmax(ratio), ratio.shape))
hot_value = float(ratio[hot_cell])

# A representative cold cell, picked as the median rather than the worst so the callout is
# typical of the map instead of the most flattering example available.
cold_values = ratio[np.isfinite(ratio) & (ratio < 1.0)]
cold_target = float(np.median(cold_values))
cold_cell = min(
    (c for c in world.FREE_CELLS if np.isfinite(ratio[c]) and ratio[c] < 1.0),
    key=lambda c: abs(ratio[c] - cold_target),
)

# ------------------------------------------------------------------------------------ plot
RAMP = ["#eef5fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
seq = LinearSegmentedColormap.from_list("belief_blue", RAMP)
div = LinearSegmentedColormap.from_list(
    "attribution", ["#d03b3b", "#e88c8c", "#f0efec", "#86b6ef", "#1c5cab"]
)
WALL_CMAP = LinearSegmentedColormap.from_list("w", ["#33332f", "#33332f"])
INK, MUTED, SURFACE = "#0b0b0b", "#6f6d68", "#fcfcfb"
BLUE, RED = "#1c5cab", "#c0392b"

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": SURFACE})
FIG_W, FIG_H = 10.0, 6.6
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=150, facecolor=SURFACE)

log_ratio = np.log2(ratio)
limit = float(np.nanmax(np.abs(log_ratio)))
vmax_seq = max(float(np.nanmax(belief)), float(np.nanmax(null_belief))) * 1.02

panels = [
    {"data": belief, "cmap": seq, "norm": None, "vmax": vmax_seq,
     "label": "What it believes"},
    {"data": null_belief, "cmap": seq, "norm": None, "vmax": vmax_seq,
     "label": "What it'd think anyway"},
    {"data": log_ratio, "cmap": div,
     "norm": TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit), "vmax": None,
     "label": "The part you caused"},
]

LEFT, PANEL_W, GAP = 0.045, 0.265, 0.050   # 0.045 + 3*0.265 + 2*0.05 = 0.94, fits
PANEL_H = (PANEL_W * FIG_W) * (world.H / world.W) / FIG_H
PANEL_Y = 0.455

axes = []
for i, panel in enumerate(panels):
    x = LEFT + i * (PANEL_W + GAP)
    ax = fig.add_axes([x, PANEL_Y, PANEL_W, PANEL_H])
    axes.append((ax, x))

    shown = np.where(world.FREE, panel["data"], np.nan)
    if panel["norm"] is not None:
        ax.imshow(shown, cmap=panel["cmap"], norm=panel["norm"], interpolation="nearest")
    else:
        ax.imshow(shown, cmap=panel["cmap"], interpolation="nearest", vmin=0,
                  vmax=panel["vmax"])
    ax.imshow(np.where(world.WALLS, 1.0, np.nan), cmap=WALL_CMAP, interpolation="nearest")

    ax.set_xticks(np.arange(-0.5, world.W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, world.H, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=0.6)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(x + PANEL_W / 2, PANEL_Y - 0.055, panel["label"], fontsize=12.5,
             color=INK, ha="center", fontweight="bold")

# The operators. These are the whole explanation, so they are set large and dark enough to
# read before any of the words do.
mid_y = PANEL_Y + PANEL_H / 2
for i, symbol in enumerate(("÷", "=")):
    x = LEFT + (i + 1) * PANEL_W + (i + 0.5) * GAP
    fig.text(x, mid_y, symbol, fontsize=34, color=INK, ha="center", va="center",
             fontweight="bold")

# Two callouts on the result panel, so the colours mean something specific rather than
# gesturing at a mood.
ax3, x3 = axes[2]
halo = [pe.withStroke(linewidth=3.0, foreground=SURFACE)]
ax3.annotate(f"{hot_value:.0f}x", xy=(hot_cell[1], hot_cell[0]),
             xytext=(hot_cell[1] - 6.5, hot_cell[0] - 2.2),
             fontsize=15, fontweight="bold", color=BLUE, path_effects=halo,
             arrowprops=dict(arrowstyle="->", color=BLUE, lw=2.0))
ax3.annotate(f"{ratio[cold_cell]:.1f}x", xy=(cold_cell[1], cold_cell[0]),
             xytext=(cold_cell[1] + 1.5, cold_cell[0] + 3.4),
             fontsize=15, fontweight="bold", color=RED, path_effects=halo,
             arrowprops=dict(arrowstyle="->", color=RED, lw=2.0))

fig.text(LEFT, 0.925, "How to catch an AI guessing", fontsize=29,
         fontweight="bold", color=INK, ha="left")
fig.text(LEFT, 0.845,
         "Run the guard's map of you twice. Once with ears and eyes, once deaf and blind.",
         fontsize=14, color="#4a4945", ha="left")

fig.text(LEFT, 0.285, "Above 1, you put it there.", fontsize=15.5,
         fontweight="bold", color=BLUE, ha="left")
fig.text(LEFT, 0.215, "Below 1, it is guessing.", fontsize=15.5,
         fontweight="bold", color=RED, ha="left")
fig.text(0.40, 0.295,
         f"After this guard searched the room and found nothing,\n"
         f"{below} of the {total_cells} floor tiles fell below 1. Almost anywhere\n"
         f"it goes next, it is making up.",
         fontsize=12, color=MUTED, ha="left", va="top", linespacing=1.6)
fig.text(LEFT, 0.038, "Day 5 of 7.  Every tile and both numbers are real output from the filter.",
         fontsize=9.5, color=MUTED, ha="left")

path = OUT / "day5-attribution.png"
fig.savefig(path)
print("wrote", path)
print(f"hot {hot_cell} {hot_value:.2f}x   cold {cold_cell} {ratio[cold_cell]:.2f}x")
print(f"below 1.0: {below}/{total_cells}")
