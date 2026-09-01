"""The central-result figure: outer (transport) and inner (payload) layers of the
same signal sit on opposite sides of the written-English Zipf value.

Replaces a stale figure that still showed the deleted payload-era architecture
(a three-step "bulk -0.173 / full frame -0.890 / payload only -0.990" progression;
the -0.990 object was removed from the paper on 2026-08-09 and the -0.173 bulk row
was replaced by the script's imputation-free construction on 2026-08-13).

All four plotted values are printed by frb_three_part.py:
PART I  : FRB 20201124A bulk -0.141, FRB 121102 bulk -0.356
PART III: stream 1 -0.890, stream 2 -1.057
"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent / "figures"

GRAY, BLUE, GREEN = "#757575", "#1565c0", "#2e7d32"

labels = ["outer layer\nFRB 20201124A", "outer layer\nFRB 121102",
          "payload stream 1", "payload stream 2"]
vals = [-0.141, -0.356, -0.890, -1.057]
cols = [GRAY, GRAY, BLUE, BLUE]

fig, ax = plt.subplots(figsize=(8.0, 4.6))

# written-English reference: dashed line with a tolerance band
ax.axhspan(-1.15, -0.85, color=GREEN, alpha=0.10, zorder=0)
ax.axhline(-1.0, color=GREEN, ls="--", lw=1.6, zorder=1)
ax.text(-0.45, -1.03, "written English", color=GREEN, fontsize=10,
        fontweight="bold", va="bottom", ha="left")

for i, (v, c) in enumerate(zip(vals, cols)):
    ax.vlines(i, 0, v, color=c, lw=4, zorder=2)
    ax.plot(i, v, "o", color=c, ms=13, zorder=3)
    ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=10.5, fontweight="bold", color=c)

from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], marker="o", ls="", color=GRAY, ms=10,
           label="transport layer (bulk parameters)"),
    Line2D([], [], marker="o", ls="", color=BLUE, ms=10,
           label="payload (transport removed)"),
], loc="center left", bbox_to_anchor=(0.02, 0.72), fontsize=9.5, frameon=False)

ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9.5)
ax.set_xlim(-0.55, 3.9)
ax.set_ylim(0.02, -1.28)          # inverted: toward language is up
ax.set_ylabel("Zipf slope", fontsize=11)
ax.set_title("The outer and inner layers sit on opposite sides\nof the language value",
             fontsize=11.5, fontweight="bold", pad=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=0.22)

fig.tight_layout()
fig.savefig(f"{FIG}/fig_progression.png", dpi=200)
print("fig_progression.png written")
