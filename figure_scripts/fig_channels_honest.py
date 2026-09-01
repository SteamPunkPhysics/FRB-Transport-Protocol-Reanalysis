"""Honest channel figure. My recomputation does NOT reproduce the source paper's
claim that PC1 is featureless: all six components carry sequential structure. The
multiplexing claim does not depend on the carrier being flat; it depends on the
channels being independent of each other. Plot what is measured.

The z annotations are the canonical values printed by frb_three_part.py PART II.a
(rng_pc, seed 102), which is the paper's single source of numbers. This script
previously ran its own permutation null (seed 0), which gave 463/523/252/303/230/275:
the same measurement under a different draw. Text, figure and script now all quote
one generator. The MI values are deterministic and are computed here; they match the
script exactly (0.708 / 0.853 / 0.421 / 0.534 / 0.345 / 0.439)."""
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
sp = np.load(ROOT / "spectra.npy").astype(float)
U, S, _ = np.linalg.svd(sp - sp.mean(0), full_matrices=False)
var = (S ** 2) / (S ** 2).sum()

def H(c):
    c = np.asarray(list(c), float); p = c / c.sum(); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())
def MI1(s):
    x, y = s[:-1], s[1:]
    return H(Counter(x).values()) + H(Counter(y).values()) - H(Counter(zip(x, y)).values())

mi = []
for pc in range(6):
    v = U[:, pc] * S[pc]
    q = np.digitize(v, np.quantile(v, [.25, .5, .75]))
    mi.append(MI1(list(q)))
mi = np.array(mi)
# canonical z vs order-shuffle, from frb_three_part.py PART II.a (SCRIPT_OUTPUT.txt)
zs = np.array([401.7, 481.1, 227.5, 305.6, 213.9, 271.1])

# cross-channel MI, to show the pairing that carries the multiplexing claim
def xmi(i, j):
    a = np.digitize(U[:, i]*S[i], np.quantile(U[:, i]*S[i], [.25,.5,.75]))
    b = np.digitize(U[:, j]*S[j], np.quantile(U[:, j]*S[j], [.25,.5,.75]))
    return H(Counter(a).values()) + H(Counter(b).values()) - H(Counter(zip(a, b)).values())

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3),
                              gridspec_kw={"width_ratios": [1.35, 1]})
cols = ["#9e9e9e"] + ["#1565c0"]*5
for i, (m, c) in enumerate(zip(mi, cols)):
    ax.vlines(i, 0, m, color=c, lw=5, zorder=2)
    ax.plot(i, m, "o", color=c, ms=12, zorder=3)
    ax.annotate(f"z={zs[i]:.0f}", (i, m), textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=9, fontweight="bold", color="#0d47a1")
ax.set_xticks(range(6))
ax.set_xticklabels([f"PC{i+1}\n{var[i]*100:.0f}%" for i in range(6)], fontsize=9.5)
ax.set_ylabel("lag-1 mutual information (bits)", fontsize=10.5)
ax.set_ylim(0, max(mi)*1.28)
ax.set_title("Every component carries sequential structure,\nincluding the carrier",
             fontsize=11, fontweight="bold")
ax.text(0, 0.04, "carrier", ha="center", fontsize=8.5, style="italic", color="#424242")
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=0.25)

pairs = [("PC2\u2013PC3", xmi(1,2)), ("PC4\u2013PC5", xmi(3,4)), ("PC2\u2013PC4", xmi(1,3))]
c2 = ["#2e7d32", "#9e9e9e", "#9e9e9e"]
ax2.barh([2,1,0], [p[1] for p in pairs], color=c2, height=.55)
ax2.set_yticks([2,1,0]); ax2.set_yticklabels([p[0] for p in pairs], fontsize=10.5)
for k,(lab,v) in enumerate(pairs):
    ax2.text(v+0.008, 2-k, f"{v:.3f}", va="center", fontsize=10, fontweight="bold")
ax2.set_xlabel("cross-channel mutual information (bits)", fontsize=10.5)
ax2.set_xlim(0, max(p[1] for p in pairs)*1.35)
ax2.set_title("PC2–PC3 pairs strongly;\nPC4–PC5 does not separate",
              fontsize=11, fontweight="bold")
for s in ("top","right"): ax2.spines[s].set_visible(False)
ax2.grid(axis="x", alpha=0.25)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_channels.png", dpi=200)
print("z:", " ".join(f"PC{i+1}={z:.0f}" for i, z in enumerate(zs)))
print("MI:", " ".join(f"{m:.3f}" for m in mi))
print("pairs:", {a: round(b,3) for a,b in pairs})
