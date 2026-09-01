"""The 32-offset standing-wave profile, both sources, no interpretive labels.

Replaces a stale figure that carried "Header / Body / Payload" region labels;
the paper's field-segmentation section explicitly does not establish such a
split and nothing downstream uses one. Left panel: FRB 20201124A stream 1
(the paper's r = 0.770 fit). Right panel: FRB 20220912A on today's archive,
the pipeline of frb_three_part.py PART X (r = 0.496, p = 0.004; the
pre-registered outcome on the 894 spectra then public was r = 0.595).
"""
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
W = 32
WAVE = {3: 1.0, 2: 0.5, 1: 0.0, 0: -1.0}
BLUE, RED = "#1565c0", "#c62828"

def profile(sym):
    nr = len(sym) // W
    M = np.asarray(sym[:nr * W]).reshape(nr, W)
    return np.array([np.mean([WAVE[int(v)] for v in M[:, j]]) for j in range(W)])

def fit(prof):
    t = np.arange(W)
    X = np.column_stack([np.sin(2*np.pi*t/W), np.cos(2*np.pi*t/W), np.ones(W)])
    beta, *_ = np.linalg.lstsq(X, prof, rcond=None)
    r, p = stats.pearsonr(prof, X @ beta)
    return X @ beta, abs(r), p

# FRB 20201124A stream 1
sp = np.load(ROOT / "spectra.npy").astype(float)
U, S, _ = np.linalg.svd(sp - sp.mean(0), full_matrices=False)
a = np.angle(U[:, 1] * S[1] + 1j * U[:, 2] * S[2])
s1 = ((a + np.pi) / (np.pi / 2)).astype(int) % 4
p1 = profile(list(s1)); f1, r1, pv1 = fit(p1)

# FRB 20220912A, PART X construction (PC2 quartiles)
panels = [("FRB 20201124A, stream 1", p1, f1, r1, pv1)]
f22 = ROOT / "spectra_20220912a.npy"
if f22.exists():
    sp22 = np.load(f22).astype(float)
    mu, sd = sp22.mean(0), sp22.std(0); sd[sd < 1e-12] = 1.0
    Xn = (sp22 - mu) / sd; Xn = Xn - Xn.mean(0)
    U2, S2, _ = np.linalg.svd(Xn, full_matrices=False)
    v = U2[:, 1] * S2[1]
    s22 = np.digitize(v, np.nanquantile(v, [0.25, 0.5, 0.75])).astype(int)
    p2 = profile(list(s22)); ff2, r2, pv2 = fit(p2)
    panels.append(("FRB 20220912A (prediction source)", p2, ff2, r2, pv2))

fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.0),
                         sharey=True)
axes = np.atleast_1d(axes)
for ax, (title, prof, ft, r, pv) in zip(axes, panels):
    ax.plot(np.arange(W), prof, "o-", color=BLUE, ms=5, lw=1.6,
            label="mean wave-state amplitude")
    ax.plot(np.arange(W), ft, "--", color=RED, lw=2,
            label="best-fit period-32 sinusoid")
    ptxt = "p < 10^{-5}" if pv < 1e-5 else f"p = {pv:.3f}"
    ax.set_title(f"{title}\n$r = {r:.3f}$, ${ptxt}$", fontsize=10.5,
                 fontweight="bold")
    ax.set_xlabel("offset within 32-symbol frame", fontsize=10.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(alpha=0.22)
axes[0].set_ylabel("mean wave-state amplitude", fontsize=10.5)
axes[0].legend(loc="lower left", fontsize=8.5, frameon=False)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig04_standing_wave.png", dpi=200)
print("fig04_standing_wave.png written")
