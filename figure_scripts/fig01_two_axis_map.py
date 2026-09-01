"""The two-axis communication-complexity map (Doyle/McCowan), neutral labels.

Replaces a stale figure that used the source analysis's interpretive stream
names. All plotted values are the canonical ones from SCRIPT_OUTPUT.txt and
the paper's own tables: content streams (Zipf top-20 rank slope, contiguous
depth), outer transport layers, and literature reference systems.
"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent / "figures"
BLUE, GRAY, GREEN = "#1565c0", "#757575", "#2e7d32"

# (label, zipf, depth, kind)  kind: ref / outer / content
pts = [
    ("Vela pulsar",        -0.30,  0, "ref"),
    ("squirrel monkey",    -0.60,  1, "ref"),
    ("bottlenose dolphin", -1.00,  4, "ref"),
    ("written English",    -1.00,  9, "ref"),
    ("outer layer\nFRB 20201124A", -0.141, 1,  "outer"),
    ("outer layer\nFRB 121102",    -0.356, 20, "outer"),
    ("stream 1", -0.890, 18, "content"),
    ("stream 2", -1.057,  8, "content"),
]

fig, ax = plt.subplots(figsize=(8.2, 5.2))
ax.axvspan(-1.2, -0.8, color=GREEN, alpha=0.08, zorder=0)
ax.text(-1.0, 22.6, "language-range slope", color=GREEN, fontsize=9,
        ha="center", style="italic")
ax.axhline(9, color=GRAY, ls=":", lw=1.2, zorder=0)

off = {"Vela pulsar": (6, -13), "squirrel monkey": (6, 6),
       "bottlenose dolphin": (8, -3), "written English": (8, 3),
       "outer layer\nFRB 20201124A": (8, 8), "outer layer\nFRB 121102": (8, -8),
       "stream 1": (10, -3), "stream 2": (10, -3)}
for lab, x, y, kind in pts:
    if kind == "ref":
        ax.plot(x, y, "o", color=GRAY, ms=8)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=off[lab],
                    fontsize=9, color="#616161")
    elif kind == "outer":
        ax.plot(x, y, "x", color=GRAY, ms=10, mew=2.5)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=off[lab],
                    fontsize=9, color="#616161", style="italic")
    else:
        ax.plot(x, y, "o", color=BLUE, ms=11)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=off[lab],
                    fontsize=10.5, color=BLUE, fontweight="bold")

ax.set_xlim(0.03, -1.28)          # language toward the right
ax.set_ylim(-1.2, 24)
ax.set_xlabel("Zipf slope (rank-frequency)", fontsize=11)
ax.set_ylabel("depth (deepest contiguous lag, $z>3$)", fontsize=11)
ax.set_title("The two-axis map: dots are content, crosses are transport",
             fontsize=11.5, fontweight="bold", pad=10)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(FIG / "fig01_intelligence_filter.png", dpi=200)
print("fig01_intelligence_filter.png written")
