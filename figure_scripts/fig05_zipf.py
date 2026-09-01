"""Zipf rank-frequency figure, both content streams, neutral labels.

Replaces a stale figure whose legend still used the source analysis's
interpretive stream names ("Teaching" / "Phase-diff"). This paper's terms are
stream 1 and stream 2. Word frequencies are computed here by the paper's own
pipeline (identical to frb_three_part.py PART III); the annotated slopes are
the canonical top-20 rank-frequency values from SCRIPT_OUTPUT.txt.
"""
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sp = np.load(ROOT / "spectra.npy").astype(float)
U, S, _ = np.linalg.svd(sp - sp.mean(0), full_matrices=False)
a = np.angle(U[:, 1] * S[1] + 1j * U[:, 2] * S[2])
b = np.angle(U[:, 3] * S[3] + 1j * U[:, 4] * S[4])
d = (a - b + np.pi) % (2 * np.pi) - np.pi
q = lambda x: ((x + np.pi) / (np.pi / 2)).astype(int) % 4

def word_freqs(sym):
    words, cur, run = [], int(sym[0]), 1
    for x in sym[1:]:
        if x == cur:
            run += 1
        else:
            words.append((cur, run)); cur, run = int(x), 1
    words.append((cur, run))
    return np.sort(np.array(list(Counter(words).values()), float))[::-1]

f1, f2 = word_freqs(q(a)), word_freqs(q(d))
C1, C2, MUT = "#1565c0", "#2e7d32", "#9e9e9e"

fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.loglog(np.arange(1, len(f1) + 1), f1, "o", color=C1, ms=6,
          label="stream 1 (slope $-0.890$)")
ax.loglog(np.arange(1, len(f2) + 1), f2, "s", color=C2, ms=5,
          label="stream 2 (slope $-1.057$)")
r = np.arange(1, 70)
ax.loglog(r, f1[0] * r**-1.0, "--", color=MUT, lw=1.6,
          label="slope $-1.0$ (written English)")
ax.set_xlabel("word rank", fontsize=11)
ax.set_ylabel("word frequency", fontsize=11)
ax.set_title("Word rank-frequency on the two content streams",
             fontsize=11.5, fontweight="bold", pad=10)
ax.legend(loc="lower left", fontsize=9.5, frameon=False)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.grid(alpha=0.22, which="both")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig05_zipf_plots.png", dpi=200)
print("fig05_zipf_plots.png written")
