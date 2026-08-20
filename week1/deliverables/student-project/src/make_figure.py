"""Generate the preprint's results figure directly from the simulator.

The figure is never hand-typed: both panels are computed by importing the
simulator, so a change to the model or the cost matrix updates the figure and can
never silently disagree with the numbers in the paper.

    python src/make_figure.py        # writes paper/figures/results.{pdf,png}
"""

import contextlib
import io
import os
import runpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

# --- run the simulator, quietly, and borrow its functions ---------------------
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sim = runpy.run_path(os.path.join(HERE, "simulator.py"))

results = sim["results"]
expected_cost = sim["expected_cost"]
belief_from_p = sim["belief_from_p"]
boundaries = sim["boundaries"]
ACT_NOW = sim["ACT_NOW"]

POLICIES = ["B0", "P1", "P2", "P3"]
LABELS = {"B0": "B0\nreactive\nbaseline", "P1": "P1\neye-tuned\nbands",
          "P2": "P2\n+ evidence\nhierarchy", "P3": "P3\ncost-derived\n+ VoI"}
totals = {p: float(results[f"cost_{p}"].sum()) for p in POLICIES}

CANON_AMT, CANON_PROD = 19.99, "consumable"
grid = np.arange(0, 1.0001, 0.005)
curves = {a: [expected_cost(a, belief_from_p(p), CANON_AMT, CANON_PROD) for p in grid]
          for a in ACT_NOW}
bounds = boundaries(CANON_AMT, CANON_PROD)

# --- palette (validated: all-pairs CVD ΔE 9.2, normal-vision 24.0, light) -----
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY_MARK = "#c3c2b7"     # de-emphasised marks
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)

# --- panel A: emphasis bar — one hue for the point, gray for context ---------
xs = np.arange(len(POLICIES))
vals = [totals[p] for p in POLICIES]
colors = [GRAY_MARK, GRAY_MARK, GRAY_MARK, BLUE]
ax1.bar(xs, vals, width=0.62, color=colors, zorder=3)
for x, v in zip(xs, vals):
    ax1.text(x, v + 3, f"{v:.2f}", ha="center", va="bottom", fontsize=8,
             color=INK if x == 3 else INK_2,
             fontweight="bold" if x == 3 else "normal")
ax1.set_xticks(xs)
ax1.set_xticklabels([LABELS[p] for p in POLICIES], fontsize=7.2, linespacing=1.35)
ax1.set_ylabel("total decision cost over 50 cases\n([ASSUMED] units)")
ax1.set_ylim(0, max(vals) * 1.18)
ax1.set_title("(a) Cost-derived thresholds are cheapest", fontsize=8.5, loc="left", pad=8)
ax1.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax1.set_axisbelow(True)
ax1.tick_params(axis="x", length=0)

# --- panel B: three action costs vs belief; boundaries where the winner flips -
series = [("approve", BLUE), ("question", ORANGE), ("stop", AQUA)]
for name, color in series:
    ax2.plot(grid * 100, curves[name], color=color, linewidth=1.8, zorder=3,
             label=name)

# direct labels (relief rule: aqua is sub-3:1 on white, so it must carry a label)
label_at = {"approve": (0.97, (-4, 7), "right"),
            "question": (0.97, (-4, -13), "right"),
            "stop": (0.10, (4, 6), "left")}
for name, color in series:
    xf, off, ha = label_at[name]
    xi = int(xf * (len(grid) - 1))
    ax2.annotate(name, (grid[xi] * 100, curves[name][xi]),
                 textcoords="offset points", xytext=off, ha=ha,
                 color=color, fontsize=8, fontweight="bold")

for p, frm, to in bounds:
    ax2.axvline(p * 100, color=MUTED, linewidth=0.9, linestyle=(0, (3, 3)), zorder=2)
    ax2.annotate(f"{p:.1%}", (p * 100, ax2.get_ylim()[1]),
                 textcoords="offset points", xytext=(3, -9),
                 color=INK_2, fontsize=7.5)

ax2.set_xlabel("P(hostile) = P(H2) + P(H3)   [%]")
ax2.set_ylabel("expected cost of the action\n([ASSUMED] units)")
ax2.set_title("(b) Where the cheapest action changes", fontsize=8.5, loc="left", pad=8)
ax2.set_xlim(0, 100)
ax2.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax2.set_axisbelow(True)
ax2.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, -0.24), handlelength=1.6, labelcolor=INK_2)

fig.savefig(os.path.join(OUT, "results.pdf"))
fig.savefig(os.path.join(OUT, "results.png"), dpi=220)
print("wrote", os.path.join(OUT, "results.pdf"))
print("panel A totals:", {k: round(v, 2) for k, v in totals.items()})
print("panel B boundaries:", [(f"{p:.1%}", f"{a}->{b}") for p, a, b in bounds])
