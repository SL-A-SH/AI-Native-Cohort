"""The day 3 figure: the parameter I could have turned up, and did not.

Three runs of the same filter from the same observation. The only thing that changes between
the second and the third is one number, `poi_weight`, which decides how much more likely the
player is to be moving through a doorway or a corridor than sitting on open floor.

Everything drawn here is computed by `belief.py` at render time, including the two
disagreement figures in the captions. Nothing is typed in by hand, so the picture cannot drift
away from the claim the post makes about it.

    python src/day3_tuning_figure.py     # writes figures/day3-the-knob.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import belief as belief_mod
import world

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

NPC = (11, 3)      # where the guard is standing
NOISE = (4, 17)    # where the sound came from
STEPS = 20         # ticks between the observation and the picture

# --------------------------------------------------------------------------- the three runs
def run(poi_weighted: bool, weight: float | None = None) -> np.ndarray:
    params = dict(belief_mod.PARAMS)
    if weight is not None:
        params["poi_weight"] = weight
    grid = belief_mod.BeliefGrid(poi_weighted=poi_weighted, params=params)
    grid.update(belief_mod.Observation("sound", NOISE, tick=0), agent_cell=NPC)
    grid.predict(steps=STEPS)
    return grid.belief


def total_variation(a: np.ndarray, b: np.ndarray) -> float:
    """How much two beliefs disagree, as a single number between 0 and 1.

    Half the summed absolute difference, which is the standard definition and is what the
    post means by "the two maps disagree by about 5%".
    """
    return 0.5 * float(np.abs(a - b).sum())


uniform = run(poi_weighted=False)
weighted_2 = run(poi_weighted=True, weight=2.0)
weighted_8 = run(poi_weighted=True, weight=8.0)

tv_2 = total_variation(uniform, weighted_2)
tv_8 = total_variation(uniform, weighted_8)

# ------------------------------------------------------------------------------- the plot
# Same palette, type and furniture as the day 1 figure, so the week reads as one series.
RAMP = ["#eef5fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
cmap = LinearSegmentedColormap.from_list("belief_blue", RAMP)
WALL_CMAP = LinearSegmentedColormap.from_list("w", ["#33332f", "#33332f"])
INK, MUTED, SURFACE = "#0b0b0b", "#6f6d68", "#fcfcfb"
KEEP, REJECT = "#2e7d5b", "#c4501f"

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": SURFACE})
FIG_W, FIG_H = 9.0, 9.4
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=150, facecolor=SURFACE)

# One scale across all three panels. Normalising each one to its own maximum would make the
# three look equally confident, which is the opposite of what the figure is about.
vmax = max(float(np.nanmax(b)) for b in (uniform, weighted_2, weighted_8)) * 1.02

panels = [
    {
        "belief": uniform,
        "eyebrow": "A  ·  uniform",
        "caption": "The player could be\nanywhere they could\nhave walked.",
        "note": None,
        "colour": MUTED,
    },
    {
        "belief": weighted_2,
        "eyebrow": "B  ·  weight 2",
        "caption": "Doorways and corridors\ncount double.",
        "note": f"{tv_2:.0%} different from A\nDefensible",
        "colour": KEEP,
    },
    {
        "belief": weighted_8,
        "eyebrow": "B  ·  weight 8",
        "caption": "Identical code. One\nnumber changed.",
        "note": f"{tv_8:.0%} different from A\nNo argument for it",
        "colour": REJECT,
    },
]

# Side by side rather than stacked: the map is 24 by 14, so three of them down a portrait page
# is either tiny or four screens tall, and a comparison is easier to make across than down.
#
# The panel height is derived from the map's own aspect rather than chosen. An axes box whose
# shape does not match the data letterboxes the image inside it, which is what put a band of
# empty page either side of every map in the first version of this figure.
LEFT, PANEL_W, COL_GAP = 0.055, 0.285, 0.0225
PANEL_H = (PANEL_W * FIG_W) * (world.H / world.W) / FIG_H
PANEL_Y = 0.545

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

    ax.plot(NOISE[1], NOISE[0], marker="*", markersize=11, color="#eb6834",
            markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=5)

    fig.text(x, PANEL_Y + PANEL_H + 0.018, panel["eyebrow"], fontsize=12.5,
             fontweight="bold", color=panel["colour"], ha="left")
    fig.text(x, PANEL_Y - 0.030, panel["caption"], fontsize=11,
             color=INK, ha="left", va="top", linespacing=1.5)
    if panel["note"]:
        fig.text(x, PANEL_Y - 0.128, panel["note"], fontsize=12,
                 fontweight="bold", color=panel["colour"], ha="left", va="top",
                 linespacing=1.4)

fig.text(LEFT, 0.930, "The number I could have changed", fontsize=27,
         fontweight="bold", color=INK, ha="left")
fig.text(LEFT, 0.862,
         "Two versions of the same AI, guessing where a player went after one sound.\n"
         "How different they look is set by one number, and I chose it myself.",
         fontsize=13.5, color="#4a4945", ha="left", va="top", linespacing=1.55)
fig.text(LEFT, 0.775, "★ = where the sound came from.  Darker = more likely.  "
                      f"{STEPS} ticks later.",
         fontsize=10.5, color=MUTED, ha="left")

halo = [pe.withStroke(linewidth=3.0, foreground=SURFACE)]
fig.text(LEFT, 0.310,
         "Going from 2 to 8 makes my comparison look\n"
         f"{tv_8 / tv_2:.1f} times stronger. Nobody would ever catch it.",
         fontsize=17, color=INK, ha="left", va="top", linespacing=1.5, fontweight="bold",
         path_effects=halo)
fig.text(LEFT, 0.190,
         "A person crossing a room really does go through the door, so double is an argument\n"
         "someone can have with me. Eight times is me picking the number that makes the chart\n"
         "bigger. So 2 stays, and the comparison stays underwhelming.",
         fontsize=11.5, color=MUTED, ha="left", va="top", linespacing=1.6)
fig.text(LEFT, 0.050,
         "Day 3 of 7.  Every cell and both percentages are output from the filter, not drawn by hand.",
         fontsize=10, color=MUTED, ha="left")

path = OUT / "day3-the-knob.png"
fig.savefig(path)
print("wrote", path)
print(f"TV(A, B@2) = {tv_2:.4f}   TV(A, B@8) = {tv_8:.4f}   ratio = {tv_8 / tv_2:.2f}")
print(f"mass on chokepoints: A={
    sum(uniform[c] for c in world.POINTS_OF_INTEREST):.3f}"
      f"  B@2={sum(weighted_2[c] for c in world.POINTS_OF_INTEREST):.3f}"
      f"  B@8={sum(weighted_8[c] for c in world.POINTS_OF_INTEREST):.3f}")
