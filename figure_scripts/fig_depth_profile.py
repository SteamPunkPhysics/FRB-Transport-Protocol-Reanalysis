"""Depth-axis figure: sequential constraint by lag on both content streams.

This is the presentational device of the McCowan/Doyle school (constraint vs
distance), applied to the layer that passes the language test. The z values are
the canonical per-lag profiles printed by frb_three_part.py PART IV
(SCRIPT_OUTPUT.txt, rng_depth seed 106, 200 surrogates per lag), hardcoded here
so that text, table and figure all quote one generator. The within-session
points are PART IV.b (rng_sess seed 107).

Palette #1565c0 / #2e7d32 validated CVD-safe (protan dE 23.6, normal dE 24.4);
marker shape is the secondary encoding.
"""
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent / "figures"

lags = np.arange(1, 41)
# canonical z by lag, frb_three_part.py PART IV
z1 = np.array([487.2, 425.0, 269.4, 194.7, 136.7, 95.1, 77.1, 60.5, 46.3, 32.8,
               24.0, 19.4, 15.6, 12.8, 9.0, 6.6, 5.2, 4.9, 2.5, 3.0,
               5.6, 3.2, 1.8, 3.9, 5.1, 7.7, 11.7, 13.9, 13.3, 11.9,
               12.4, 11.0, 10.8, 12.0, 11.3, 11.1, 8.2, 8.5, 7.1, 6.6])
z2 = np.array([228.9, 109.0, 58.7, 40.3, 24.4, 8.9, 9.2, 4.7, 1.0, -0.2,
               -0.5, -0.7, 0.9, 1.4, 0.7, 2.3, 2.2, 3.7, 3.6, 3.8,
               7.1, 3.4, 2.1, 2.4, 0.1, 2.1, 1.8, 0.7, -1.0, 1.5,
               2.6, 3.3, 2.5, 2.5, 4.3, -0.3, 2.2, 0.9, 2.9, 0.6])
# canonical within-session control, PART IV.b (stream 1): (lag, z, pairs).
# The null permutes y WITHIN each observing session (session-preserving).
sess = [(28, 4.9), (32, 6.5)]

C1, C2, RED, MUT = "#1565c0", "#2e7d32", "#c62828", "#9e9e9e"

fig, ax = plt.subplots(figsize=(10.4, 4.4))

# reference verticals: comparison-system depths and the frame period
for x, lab in [(4, "bottlenose dolphin depth $\\sim$4"),
               (9, "written English depth $\\sim$9")]:
    ax.axvline(x, color=MUT, ls=":", lw=1.2, zorder=1)
    ax.text(x - 0.55, 560, lab, rotation=90, va="top", ha="center",
            fontsize=8, color="#757575", style="italic")
ax.axvline(32, color="#616161", ls="--", lw=1.4, zorder=1)
ax.text(32 - 0.55, 560, "frame period 32 (Section 2)", rotation=90, va="top",
        ha="center", fontsize=8, color="#424242", style="italic")

# detection threshold
ax.axhline(3, color=RED, ls="--", lw=1.4, zorder=1)
ax.text(0.7, 3.6, "3$\\sigma$", color=RED, va="bottom", fontsize=10,
        fontweight="bold")

ax.plot(lags, z1, "-o", color=C1, lw=2, ms=4.5, zorder=3, label="stream 1")
ax.plot(lags, z2, "-s", color=C2, lw=2, ms=4, zorder=2, label="stream 2")
ax.plot([s[0] for s in sess], [s[1] for s in sess], "D", mfc="none", mec=C1,
        mew=2, ms=9, zorder=4, label="stream 1, within-session pairs only")

# selective direct labels
ax.annotate("stream 1", (2, 425.0), textcoords="offset points", xytext=(8, 4),
            fontsize=10, fontweight="bold", color=C1)
ax.annotate("stream 2", (2, 109.0), textcoords="offset points", xytext=(8, -14),
            fontsize=10, fontweight="bold", color=C2)

ax.set_yscale("symlog", linthresh=3, linscale=0.6)
ax.set_ylim(-2, 650)
ax.set_yticks([0, 3, 10, 30, 100, 300])
ax.set_yticklabels(["0", "3", "10", "30", "100", "300"])
ax.set_xlim(0.3, 41.5)
ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])
ax.set_xlabel("lag (bursts back)", fontsize=11)
ax.set_ylabel("lagged MI, $z$ vs order-shuffle", fontsize=11)
ax.set_title("Constraint by distance on the content streams",
             fontsize=11.5, fontweight="bold", pad=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.55, 1.0), fontsize=9,
          frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=0.22)

fig.tight_layout()
fig.savefig(f"{FIG}/fig_depth_profile.png", dpi=200)
print("fig_depth_profile.png written")
