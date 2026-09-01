"""Large-N bulk-parameter test: depth gain on the 11,553-burst FAST
catalogue of FRB 20240114A (Zhang, J.-S., et al. 2025, arXiv:2507.14707;
table from doi:10.57760/sciencedb.Fastro.00030).

The instrument is the held-out depth gain of the 2026-07 record, recovered
verbatim from the session that froze its calibration: contiguous 6-fold CV
(train on the other five folds, both sides of the test fold), add-0.5
smoothing, unseen context -> uniform 1/A, gain = CE(order 1) - min CE(2..5),
quantile-digitised log10 energies, AAFT surrogates (amplitude distribution
preserved exactly, power spectrum of the gaussianised series preserved).
Validation gate: FRB 121102 (tables1.dat.txt, N=1652, A=4) must reproduce
the frozen gain +0.0251 exactly (the estimator is deterministic given the
sequence); its AAFT z reproduces within surrogate noise.

Calibrated tracks (A=4): language (English through the identical estimator)
rises to ~+0.14 at N~10k; a first-order physical process (AR(1)) sits at
+0.015; noise at 0. The frozen prediction this experiment answers: at
N = 5,000-10,000 the two hypotheses diverge hard.

Design pre-stated before any 20240114A number was computed: Block 1 =
MJD < 60500 (first activity episode), Block 2 = MJD >= 60500 (the July-
August 2024 re-brightening; regime shift per arXiv:2607.01576); storm week
MJD 60374-60382 scored alone. Nulls: global AAFT, within-block AAFT
(surrogates built separately per block, so the regime shift is inside the
null), per-block runs. Family-wise alphabets A = 2..5 plus the 9-class
energy x width scheme.

Deterministic: seeds 401 (calibration) and 402 (experiment). Skips
gracefully if the catalogue CSV is absent. Committed output:
DEPTHGAIN_OUTPUT.txt.  Runtime ~20 minutes (pure Python).
"""
import io
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CAT = ROOT / "data" / "frb20240114a" / "FRB20240114A_SuppTab2.csv"
rng_cal = np.random.default_rng(401)
rng_exp = np.random.default_rng(402)


# ---- the frozen instrument, verbatim ---------------------------------------
def cv_xent(seq, order, A, folds=6, alpha=0.5):
    n = len(seq)
    b = np.linspace(0, n, folds + 1).astype(int)
    tot = 0.0
    cnt = 0
    for f in range(folds):
        lo, hi = b[f], b[f + 1]
        tr = np.concatenate([np.arange(0, lo), np.arange(hi, n)])
        tab = defaultdict(lambda: np.zeros(A))
        for i in tr:
            if i >= order:
                tab[tuple(seq[i - order:i])][seq[i]] += 1
        for i in range(lo, hi):
            if i < order:
                continue
            c = tab.get(tuple(seq[i - order:i]))
            p = ((c[seq[i]] + alpha) / (c.sum() + alpha * A)) if c is not None else 1.0 / A
            tot -= np.log2(p)
            cnt += 1
    return tot / cnt


def depth_gain(seq, A, maxo=5):
    c = [cv_xent(seq, k, A) for k in range(maxo + 1)]
    return c[1] - min(c[2:])


def aaft(x, rng):
    n = len(x)
    g = np.sort(rng.normal(size=n))
    r = np.argsort(np.argsort(x))
    y = g[r]
    F = np.fft.rfft(y)
    ph = rng.uniform(0, 2 * np.pi, len(F))
    ph[0] = 0
    if n % 2 == 0:
        ph[-1] = 0
    return np.sort(x)[np.argsort(np.argsort(np.fft.irfft(np.abs(F) * np.exp(1j * ph), n)))]


def q(x, A):
    return np.digitize(x, np.quantile(x, np.linspace(0, 1, A + 1)[1:-1]))


def zrun(x, A, nsur, rng, blocks=None):
    real = depth_gain(q(x, A), A)
    sur = np.empty(nsur)
    for j in range(nsur):
        if blocks is None:
            xs = aaft(x, rng)
        else:
            xs = np.concatenate([aaft(x[b], rng) for b in blocks])
        sur[j] = depth_gain(q(xs, A), A)
    z = (real - sur.mean()) / sur.std()
    return real, sur.mean(), sur.std(), z, int((sur >= real).sum())


# ---- data ------------------------------------------------------------------
def e_121102_log():
    cs = [(0, 4), (5, 20), (21, 26), (27, 30), (32, 37), (38, 42), (43, 48),
          (51, 59), (60, 64), (65, 71), (72, 78), (79, 88), (89, 98)]
    nm = ['Burst', 'MJD', 'DM', 'e_DM', 'Width', 'e_Width', 'Bandwidth',
          'Fp', 'e_Fp', 'Fluence', 'e_Fluence', 'E', 'e_E']
    df = pd.read_fwf(io.StringIO(open(HERE / "tables1.dat.txt").read()),
                     colspecs=cs, names=nm)
    df = df.dropna(subset=['MJD']).sort_values('MJD').reset_index(drop=True)
    return np.log10(df['E'].astype(float).clip(lower=1e-30)).to_numpy()


def english_moby():
    import re
    from collections import Counter
    raw = io.open(HERE / "english_sample.txt", encoding="utf-8",
                  errors="ignore").read()
    if "*** START" in raw:
        raw = raw.split("*** START", 1)[1]
    if "*** END" in raw:
        raw = raw.split("*** END", 1)[0]
    L = re.sub(r' +', ' ', re.sub(r'[^a-z ]', '', raw.lower()))
    vals, counts = np.unique(list(L), return_counts=True)
    o = np.argsort(-counts)
    cum = np.cumsum(counts[o]) / counts.sum()
    cls = {}
    for i, ix in enumerate(o):
        cls[vals[ix]] = min(int(cum[i] * 4), 3)
    return np.array([cls[ch] for ch in L])


def ar1(n, rng, rho=0.9, A=4):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal() * np.sqrt(1 - rho ** 2)
    return q(x, A)


if not CAT.exists():
    print(f"depth-gain experiment SKIPPED: {CAT} not found.")
    print("Download the supplementary table of arXiv:2507.14707 from")
    print("doi:10.57760/sciencedb.Fastro.00030 to that path.")
    raise SystemExit(0)

print("=" * 76)
print("PART A  VALIDATION GATE + calibrated tracks, exact 2026-07 instrument")
print("=" * 76)

E121 = e_121102_log()
real = depth_gain(q(E121, 4), 4)
print(f"\nA.1  FRB 121102, N={len(E121)}, A=4")
print(f"     gain = {real:+.4f}    frozen record: +0.0251  "
      f"{'REPRODUCED' if abs(real - 0.0251) < 0.0005 else 'MISMATCH'}")
sur = np.array([depth_gain(q(aaft(E121, rng_cal), 4), 4) for _ in range(150)])
z = (real - sur.mean()) / sur.std()
print(f"     AAFT null {sur.mean():+.4f} +/- {sur.std():.4f}   z = {z:+.2f}   "
      f"{int((sur >= real).sum())}/150 beat    frozen: z=+3.96, 0/150")

eng = english_moby()
print(f"\nA.2  English track (Moby-Dick, the bundle corpus, same coarsening")
print("     rule as the 2026-07 md-corpus; corpus-sensitivity was +/-0.015)")
eng_track = {}
NS_LIST = [400, 800, 1652, 1863, 3000, 5000, 8000, 11553, 20000]
for N in NS_LIST:
    reps = min(3, len(eng) // N)
    g = np.mean([depth_gain(eng[i * N:(i + 1) * N], 4) for i in range(reps)])
    eng_track[N] = g
    tgt = {1863: "+0.078/+0.093", 5000: "+0.154", 20000: "+0.212"}.get(N, "")
    print(f"     N={N:<7} gain {g:+.4f}   {tgt}")

print(f"\nA.3  AR(1) rho=0.9 physics track (3 realisations each)")
ar_track = {}
for N in NS_LIST:
    g = np.mean([depth_gain(ar1(N, rng_cal), 4) for _ in range(3)])
    ar_track[N] = g
    tgt = {1863: "+0.011", 5000: "+0.014", 20000: "+0.016"}.get(N, "")
    print(f"     N={N:<7} gain {g:+.4f}   {tgt}")

print()
print("=" * 76)
print("PART B  FRB 20240114A, N = 11,553, exact instrument")
print("=" * 76)

df = pd.read_csv(CAT)
mjd = df['MJD(bary@inf freq.)'].to_numpy(float)
o = np.argsort(mjd, kind="stable")
E = np.log10(df['Energy(erg)'].to_numpy(float)[o])
W = np.log10(np.clip(df['Weff(ms)'].to_numpy(float)[o], 1e-6, None))
mjd = mjd[o]
n = len(E)
b1 = mjd < 60500.0
b2 = ~b1
storm = (mjd >= 60374.0) & (mjd < 60383.0)
print(f"\n  N={n}, MJD {mjd.min():.2f}-{mjd.max():.2f};  block1 {b1.sum()}, "
      f"block2 {b2.sum()}, storm {storm.sum()}")

print("\nB.1  Full sequence, global AAFT null")
print(f"     {'alphabet':<22}{'gain':>9}{'null':>9}{'sd':>8}{'z':>8}{'beat':>10}")
for A in (2, 3, 4, 5):
    nsur = 150 if A == 4 else 100
    r, mu, sd, z, beat = zrun(E, A, nsur, rng_exp)
    print(f"     A={A:<20}{r:>+9.4f}{mu:>+9.4f}{sd:>8.4f}{z:>+8.2f}"
          f"{beat:>6}/{nsur}")
r9 = depth_gain(np.digitize(E, np.quantile(E, [1/3, 2/3])) * 3
                + np.digitize(W, np.quantile(W, [1/3, 2/3])), 9)
s9 = []
for _ in range(100):
    Es, Ws = aaft(E, rng_exp), aaft(W, rng_exp)
    s9.append(depth_gain(np.digitize(Es, np.quantile(E, [1/3, 2/3])) * 3
                         + np.digitize(Ws, np.quantile(W, [1/3, 2/3])), 9))
s9 = np.array(s9)
print(f"     {'energy x width (A=9)':<22}{r9:>+9.4f}{s9.mean():>+9.4f}"
      f"{s9.std():>8.4f}{(r9 - s9.mean()) / s9.std():>+8.2f}"
      f"{int((s9 >= r9).sum()):>6}/100")

print("\nB.2  Full sequence, WITHIN-BLOCK AAFT (regime shift inside the null),")
print("     A=4, 150 surrogates")
blocks = [np.flatnonzero(b1), np.flatnonzero(b2)]
r, mu, sd, z, beat = zrun(E, 4, 150, rng_exp, blocks=blocks)
print(f"     gain {r:+.4f}   null {mu:+.4f} +/- {sd:.4f}   z = {z:+.2f}   "
      f"{beat}/150")

print("\nB.3  Per-block, A=4, 100 surrogates each")
for name, mask in (("block 1", b1), ("block 2", b2), ("storm wk", storm)):
    r, mu, sd, z, beat = zrun(E[mask], 4, 100, rng_exp)
    print(f"     {name:<9} N={mask.sum():<6} gain {r:+.4f}   "
          f"null {mu:+.4f} +/- {sd:.4f}   z = {z:+.2f}   {beat}/100")

print("\nB.4  Depth gain vs N (A=4): chronological prefixes, each with its own")
print("     100-surrogate AAFT z, next to the tracks measured in PART A.")
print(f"     {'N':>7}{'FRB gain':>11}{'z':>8}{'English':>10}{'AR(1)':>9}")
for N in NS_LIST:
    if N > n:
        continue
    r, mu, sd, z, beat = zrun(E[:N], 4, 100, rng_exp)
    print(f"     {N:>7}{r:>+11.4f}{z:>+8.2f}{eng_track[N]:>+10.4f}"
          f"{ar_track[N]:>+9.4f}")

print("\nB.5  VERDICT LINE. Frozen prediction at N ~ 10k:")
print("     language ~ +0.15   |   physics ~ +0.016   |   noise ~ 0.00")
print("     The B.1 A=4 row and the B.4 track are the measured answer.")
