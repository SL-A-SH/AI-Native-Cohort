"""The day 6 video: the same guard, the same level, two very different fines.

One episode replayed twice. On the left the penalty for abandoning a target is the registered
[ASSUMED] 4.0. On the right it is 32.0, eight times larger. Everything else is identical:
same case, same seeds, same player walking the same path.

The point is that both counters climb. A viewer does not need to be told the fine failed,
because they can watch it fail.

Nothing is staged. Both runs are real episodes from `experiments/run.py`'s own loop, replayed
frame by frame, and the counters use exactly the switch definition the results file uses.

    python src/day6_dither_video.py     # writes figures/day6-dither.mp4 and .gif
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import world                                  # noqa: E402
from agent import COSTS, TacticalAgent        # noqa: E402
from belief import BeliefGrid, Observation    # noqa: E402
import run as R                               # noqa: E402

OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# Case 19 of the forty. Chosen because it is the whole finding in one episode: at the
# registered fine the guard changes its mind 15 times and catches the player; at eight times
# the fine it changes its mind 4 times and loses them. Four of the forty cases show that
# reversal, and this is the shortest.
CASE_ID = 19
FPS = 10


def replay(case, dither: float) -> tuple[list[dict], bool]:
    """Re-run one episode, keeping a snapshot of every tick.

    This mirrors `run_episode` rather than calling it, because the video needs the belief and
    the live target at each step and that function returns only summary rows. The switch
    counting below is copied from it exactly so the counters on screen agree with the CSV.
    """
    costs = dict(COSTS)
    costs["dither"] = dither

    player_rng = np.random.default_rng(case.seed)
    world_rng = np.random.default_rng(case.seed + 104729)
    player = world.Player(position=case.player_start, rng=player_rng)
    agent = TacticalAgent(start=case.post, home_post=case.post,
                          costs=costs, sight_radius=R.SIGHT_RADIUS)

    agent.observe(Observation("sound", case.player_start, tick=0))

    frames, switches, previous_target = [], 0, None
    for tick in range(R.MAX_TICKS):
        true_cell = player.step()
        dist = world.distances_from(agent.cell)
        if (world.has_line_of_sight(agent.cell, true_cell)
                and dist[true_cell] <= R.SIGHT_RADIUS
                and world_rng.random() > R.SEARCH_MISS):
            agent.observe(Observation("sighting", true_cell, tick=tick))
        elif (dist[true_cell] <= R.HEARING_RADIUS
              and world_rng.random() < R.STEP_NOISE_CHANCE):
            agent.observe(Observation("sound", true_cell, tick=tick))

        decision = agent.decide()
        target = decision.target
        changed = False
        if decision.action in ("search", "flank", "call_reinforcements") and target is not None:
            if (previous_target is not None and target != previous_target
                    and agent.cell != previous_target):
                switches += 1
                changed = True
            previous_target = target

        frames.append({
            "belief": agent.belief.belief.copy(),
            "guard": agent.cell,
            "target": target,
            "switches": switches,
            "changed": changed,
        })

        agent.execute(decision)
        if (world.has_line_of_sight(agent.cell, true_cell)
                and world.distances_from(agent.cell)[true_cell] <= 1.5
                and world_rng.random() > R.SEARCH_MISS):
            return frames, True
        if agent.finished:
            break
    return frames, False


case = R.build_cases(40)[CASE_ID]
left, left_caught = replay(case, 4.0)
right, right_caught = replay(case, 32.0)
n = max(len(left), len(right))
left += [left[-1]] * (n - len(left))
right += [right[-1]] * (n - len(right))
HOLD = 22          # frames to sit on the final counters
n_total = n + HOLD

# ------------------------------------------------------------------------------------ plot
RAMP = ["#eef5fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
cmap = LinearSegmentedColormap.from_list("belief_blue", RAMP)
WALL_CMAP = LinearSegmentedColormap.from_list("w", ["#33332f", "#33332f"])
INK, MUTED, SURFACE = "#0b0b0b", "#6f6d68", "#fcfcfb"
FLASH = "#eb6834"

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": SURFACE})
FIG_W, FIG_H = 10.0, 6.4
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=135, facecolor=SURFACE)

VMAX = max(max(f["belief"].max() for f in left),
           max(f["belief"].max() for f in right)) * 1.02

LEFT, PANEL_W, GAP = 0.055, 0.42, 0.055
PANEL_H = (PANEL_W * FIG_W) * (world.H / world.W) / FIG_H
PANEL_Y = 0.375

sides = []
for i, (frames, heading, caught) in enumerate([(left, "Fine: 4 points", left_caught),
                                               (right, "Fine: 32 points", right_caught)]):
    x = LEFT + i * (PANEL_W + GAP)
    ax = fig.add_axes([x, PANEL_Y, PANEL_W, PANEL_H])

    image = ax.imshow(np.where(world.FREE, frames[0]["belief"], np.nan), cmap=cmap,
                      interpolation="nearest", vmin=0, vmax=VMAX)
    ax.imshow(np.where(world.WALLS, 1.0, np.nan), cmap=WALL_CMAP, interpolation="nearest")

    ax.set_xticks(np.arange(-0.5, world.W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, world.H, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=0.6)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    guard, = ax.plot([], [], marker="o", markersize=12, color=INK,
                     markeredgecolor=SURFACE, markeredgewidth=1.6,
                     linestyle="none", zorder=7)
    target, = ax.plot([], [], marker="X", markersize=14, color=FLASH,
                      markeredgecolor=SURFACE, markeredgewidth=1.4,
                      linestyle="none", zorder=6)

    fig.text(x, PANEL_Y + PANEL_H + 0.020, heading, fontsize=14,
             fontweight="bold", color=INK, ha="left")
    counter = fig.text(x, PANEL_Y - 0.115, "", fontsize=27, fontweight="bold",
                       color=INK, ha="left")
    fig.text(x, PANEL_Y - 0.165, "changes of mind", fontsize=12,
             color=MUTED, ha="left")
    outcome = fig.text(x + 0.155, PANEL_Y - 0.105, "", fontsize=17, fontweight="bold",
                       ha="left")
    sides.append({"frames": frames, "image": image, "guard": guard, "target": target,
                  "counter": counter, "outcome": outcome, "caught": caught})

fig.text(LEFT, 0.930, "I fined my AI for changing its mind", fontsize=27,
         fontweight="bold", color=INK, ha="left")
fig.text(LEFT, 0.878, "Same guard, same level, same hiding place. Only the penalty differs.",
         fontsize=13, color="#4a4945", ha="left")

halo = [pe.withStroke(linewidth=3.0, foreground=SURFACE)]
caption = fig.text(LEFT, 0.130, "", fontsize=15, fontweight="bold", color=INK,
                   ha="left", va="top", path_effects=halo)
fig.text(LEFT, 0.038,
         "Day 6 of 7.  X marks where it is heading. Both runs are real output, replayed tick by tick.",
         fontsize=9.5, color=MUTED, ha="left")


def draw(i: int):
    k = min(i, n - 1)
    artists = []
    for side in sides:
        f = side["frames"][k]
        side["image"].set_data(np.where(world.FREE, f["belief"], np.nan))
        side["guard"].set_data([f["guard"][1]], [f["guard"][0]])
        if f["target"]:
            side["target"].set_data([f["target"][1]], [f["target"][0]])
            # A change of mind flashes bigger for one frame, so the counter ticking up is
            # something the viewer sees happen rather than has to notice afterwards.
            side["target"].set_markersize(24 if f["changed"] else 14)
        else:
            side["target"].set_data([], [])
        side["counter"].set_text(str(f["switches"]))
        artists += [side["image"], side["guard"], side["target"], side["counter"]]

    for side in sides:
        if i >= n:
            side["outcome"].set_text("caught you" if side["caught"] else "lost you")
            side["outcome"].set_color("#2e7d5b" if side["caught"] else "#c0392b")
        else:
            side["outcome"].set_text("")
        artists.append(side["outcome"])

    if i >= n:
        caption.set_text("The decisive one stopped second-guessing. And lost.")
    elif k < 8:
        caption.set_text("It heard something. Now it has to pick a room.")
    else:
        caption.set_text("Every X is a new plan. The big fine buys fewer of them.")
    artists.append(caption)
    return artists


anim = animation.FuncAnimation(fig, draw, frames=n_total, interval=1000 // FPS, blit=False)

mp4 = OUT / "day6-dither.mp4"
anim.save(mp4, writer=animation.FFMpegWriter(fps=FPS, bitrate=3200), dpi=135)
print("wrote", mp4)

gif = OUT / "day6-dither.gif"
anim.save(gif, writer=animation.PillowWriter(fps=FPS), dpi=85)
print("wrote", gif)

print(f"case {CASE_ID}: {n} ticks, {n_total} frames, {n_total / FPS:.1f}s")
print(f"final switches   fine 4: {left[-1]['switches']} (caught={left_caught})   "
      f"fine 32: {right[-1]['switches']} (caught={right_caught})")
