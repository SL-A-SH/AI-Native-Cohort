"""The day 4 video: one guard, one noise, one failed search.

The same scenario as `day4_search_figure.py`, played out a tick at a time instead of shown as
a before and after. Nothing is staged: the guard walks the real shortest path, the belief
diffuses once per step it takes, and the sweep is the same likelihood the filter always
applies. What the video shows is what the filter did.

Writes an MP4 for LinkedIn and a GIF as a fallback for anywhere that will not take video.

    python src/day4_animation.py
"""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
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

NPC_START = (11, 3)
NOISE = (4, 17)
SEARCH_RADIUS = 5.0
FPS = 10

# ------------------------------------------------------------------------ build every frame
# The whole animation is computed first and then drawn, so a frame is only ever a snapshot of
# state the filter actually passed through. Nothing is interpolated or smoothed for looks.
frames: list[dict] = []


def hold(state: dict, count: int) -> None:
    for _ in range(count):
        frames.append(dict(state))


grid = belief_mod.BeliefGrid()
guard = NPC_START

# Beat 1. It knows only where the walls are.
hold({"belief": grid.belief.copy(), "guard": guard, "heard": False, "swept": False,
      "caption": "Heard nothing yet. You could be anywhere."}, 14)

# Beat 2. The noise.
grid.update(belief_mod.Observation("sound", NOISE, tick=0), agent_cell=guard)
hold({"belief": grid.belief.copy(), "guard": guard, "heard": True, "swept": False,
      "caption": "Then it hears something, right across the level."}, 16)

# Beat 3. The walk. One diffusion step per step taken, so the trail really does go cold at the
# rate the trip actually costs.
path = [guard]
while path[-1] != NOISE:
    path.append(world.step_towards(path[-1], NOISE))

for cell in path[1:]:
    guard = cell
    grid.predict(steps=1)
    frames.append({"belief": grid.belief.copy(), "guard": guard, "heard": True,
                   "swept": False,
                   "caption": f"It walks. Every step it takes, the trail goes colder."})

before_peak = grid.most_likely()
before_peak_p = float(grid.belief[before_peak])
searched_mask = world.visible_from(NOISE, radius=SEARCH_RADIUS)
searched_cells = tuple((int(r), int(c)) for r, c in zip(*np.where(searched_mask)))
before_mass = float(grid.belief[searched_mask].sum())

hold({"belief": grid.belief.copy(), "guard": guard, "heard": True, "swept": True,
      "caption": f"It arrives. That room holds {before_mass:.0%} of what it believes."}, 18)

# Beat 4. The failed search.
grid.update(
    belief_mod.Observation("search", NOISE, tick=len(path), cells=searched_cells),
    agent_cell=NOISE,
)
after_mass = float(grid.belief[searched_mask].sum())
after_peak = grid.most_likely()
after_old_peak_p = float(grid.belief[before_peak])

hold({"belief": grid.belief.copy(), "guard": guard, "heard": True, "swept": True,
      "caption": f"It looks. Nothing there. That room drops to {after_mass:.0%}."}, 22)

# Beat 5. Where it goes next, which is the point of the whole thing.
onward = [guard]
while onward[-1] != after_peak:
    onward.append(world.step_towards(onward[-1], after_peak))

for cell in onward[1:]:
    guard = cell
    grid.predict(steps=1)
    frames.append({"belief": grid.belief.copy(), "guard": guard, "heard": True,
                   "swept": True,
                   "caption": "So it keeps going, past the room it just cleared."})

hold({"belief": grid.belief.copy(), "guard": guard, "heard": True, "swept": True,
      "caption": "The failed search did not end the chase. It aimed it."}, 30)

# --------------------------------------------------------------------------------- the plot
RAMP = ["#eef5fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
cmap = LinearSegmentedColormap.from_list("belief_blue", RAMP)
WALL_CMAP = LinearSegmentedColormap.from_list("w", ["#33332f", "#33332f"])
INK, MUTED, SURFACE = "#0b0b0b", "#6f6d68", "#fcfcfb"
ATTENTION = "#eb6834"

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": SURFACE})
FIG_W, FIG_H = 8.0, 8.0
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=135, facecolor=SURFACE)

# One colour scale for the whole video, fixed before the first frame. Rescaling per frame
# would make every moment look equally certain and would hide the collapse this is about.
VMAX = max(float(f["belief"].max()) for f in frames) * 1.02

PANEL_W = 0.88
PANEL_H = (PANEL_W * FIG_W) * (world.H / world.W) / FIG_H
PANEL_X, PANEL_Y = 0.06, 0.315

ax = fig.add_axes([PANEL_X, PANEL_Y, PANEL_W, PANEL_H])
ax.set_xticks(np.arange(-0.5, world.W, 1), minor=True)
ax.set_yticks(np.arange(-0.5, world.H, 1), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=0.7)
ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
for spine in ax.spines.values():
    spine.set_visible(False)

image = ax.imshow(np.where(world.FREE, frames[0]["belief"], np.nan), cmap=cmap,
                  interpolation="nearest", vmin=0, vmax=VMAX)
ax.imshow(np.where(world.WALLS, 1.0, np.nan), cmap=WALL_CMAP, interpolation="nearest")

sweep_patches = [
    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=ATTENTION,
                           linewidth=1.0, zorder=4, visible=False))
    for r, c in searched_cells
]
star, = ax.plot([], [], marker="*", markersize=15, color=ATTENTION,
                markeredgecolor=SURFACE, markeredgewidth=1.0, linestyle="none", zorder=6)
guard_dot, = ax.plot([], [], marker="o", markersize=13, color=INK,
                     markeredgecolor=SURFACE, markeredgewidth=1.8, linestyle="none", zorder=7)
peak_ring, = ax.plot([], [], marker="o", markersize=15, markerfacecolor="none",
                     markeredgecolor=INK, markeredgewidth=2.0, linestyle="none", zorder=6)

fig.text(PANEL_X, 0.945, "Looking and finding nothing", fontsize=26,
         fontweight="bold", color=INK, ha="left")
fig.text(PANEL_X, 0.898, "What a guard believes, one tick at a time.",
         fontsize=13.5, color="#4a4945", ha="left")
fig.text(PANEL_X, 0.855,
         "● the guard     ★ the sound     ▢ swept and empty     ◯ where it thinks you are",
         fontsize=10.5, color=MUTED, ha="left")

CAPTION_WRAP = 52
halo = [pe.withStroke(linewidth=3.0, foreground=SURFACE)]
caption = fig.text(PANEL_X, 0.235, "", fontsize=15.5, color=INK, ha="left", va="top",
                   linespacing=1.5, fontweight="bold", path_effects=halo)
fig.text(PANEL_X, 0.125,
         "Darker means more likely. A swept cell is not ruled out, only multiplied down by the\n"
         "chance the guard looked straight at you and missed.",
         fontsize=10.5, color=MUTED, ha="left", va="top", linespacing=1.6)
fig.text(PANEL_X, 0.035, "Day 4 of 7.  Real output from the filter, one frame per tick.",
         fontsize=9.5, color=MUTED, ha="left")


def draw(i: int):
    f = frames[i]
    image.set_data(np.where(world.FREE, f["belief"], np.nan))

    guard_dot.set_data([f["guard"][1]], [f["guard"][0]])
    if f["heard"]:
        star.set_data([NOISE[1]], [NOISE[0]])
        peak_r, peak_c = np.unravel_index(np.argmax(f["belief"]), f["belief"].shape)
        peak_ring.set_data([peak_c], [peak_r])
    else:
        star.set_data([], [])
        peak_ring.set_data([], [])

    for patch in sweep_patches:
        patch.set_visible(f["swept"])

    caption.set_text(textwrap.fill(f["caption"], CAPTION_WRAP))
    return [image, star, guard_dot, peak_ring, caption, *sweep_patches]


anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 // FPS, blit=False)

mp4 = OUT / "day4-failed-search.mp4"
anim.save(mp4, writer=animation.FFMpegWriter(fps=FPS, bitrate=3200), dpi=135)
print("wrote", mp4)

gif = OUT / "day4-failed-search.gif"
anim.save(gif, writer=animation.PillowWriter(fps=FPS), dpi=90)
print("wrote", gif)

print(f"{len(frames)} frames, {len(frames) / FPS:.1f}s at {FPS}fps")
print(f"walk {len(path) - 1} steps, swept {len(searched_cells)} cells")
print(f"swept area {before_mass:.3f} -> {after_mass:.3f}")
print(f"old peak {before_peak} {before_peak_p:.4f} -> {after_old_peak_p:.4f}")
print(f"peak moves {before_peak} -> {after_peak}")
