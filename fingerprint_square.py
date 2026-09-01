"""Fingerprint square 2x2: does the payload fingerprint travel with the
SOURCE or with the TELESCOPE?

Cells: A = FRB 121102 @ Arecibo (Hewitt et al. 2022, Zenodo 7181266,
Bursts_npys unpacked); B = FRB 121102 @ FAST (per-file 60 x 4096 subint
spectra built from the FAST-FREX archive, doi:10.57760/sciencedb.15070, by
build_spectra_frb121102_fast.py); C = FRB 20201124A @ FAST (spectra.npy);
D = FRB 20220912A @ FAST (spectra_20220912a.npy).

Design (frozen before computation; see the paper's fingerprint-square
section): all four cells reduced to per-burst spectra on a common 28-bin
12.5-MHz grid over 1150-1500 MHz (the Arecibo/FAST overlap), identical
pipeline (PCA, discard PC1, the two quadrant streams), nine
relabeling-invariant stream statistics per 352-burst window, distance =
median |difference| in units of pooled within-cell window scatter.
Criteria: informative only if different-source/same-telescope distances
exceed 1.0; MATCH if the same-source/cross-telescope distance is smaller
than both.

Variants: R1 erases the burst signal from cell B (whole-file average; a
~ms burst diluted over ~6 s leaves the receiver noise floor) — the
instrument-signature positive control. R2 extracts cell B with a uniform
metadata-free rule (power-outlier subints minus the rest), the cleanest
extraction and the one a bundle user can reproduce without the extraction
metadata.

Data paths resolve relative to the repository root (parent of this
directory) and can be overridden:
  python fingerprint_square.py [arecibo_npys_dir] [fast121102_spectra_dir]
Cells whose data are absent are skipped with a note. Deterministic (the
computation draws no random numbers). Committed output:
FINGERPRINT_OUTPUT.txt.
"""
import json
import sys
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NW = 352

A_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT / "data" / "frb121102_arecibo" / "Hewitt_etal_2021_bursts"
    / "Bursts_npys")
B_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else None
if B_DIR is None:
    for cand in (ROOT / "investigation-2" / "output" / "frb121102" / "spectra",
                 HERE / "frb121102_fast_spectra"):
        if cand.exists():
            B_DIR = cand
            break
B_META = (B_DIR.parent / "extraction_metadata.json") if B_DIR else None

EDGES = np.arange(1150.0, 1500.1, 12.5)          # 28 bins


def rebin(spec, freqs):
    out = np.zeros(28)
    for b in range(28):
        m = (freqs >= EDGES[b]) & (freqs < EDGES[b + 1])
        out[b] = spec[m].mean() if m.any() else 0.0
    return out


def load_A():
    files = sorted(A_DIR.rglob("*.npy"), key=lambda p: float(p.stem.split("_")[0]))
    freqs = 1780.0 - 12.5 * (np.arange(64) + 0.5)   # frequency DESCENDS
    nb = 2432
    on = slice(int(nb * 0.45), int(nb * 0.55))
    rows = []
    for f in files:
        a = np.load(f, allow_pickle=True)
        onp = np.ma.mean(a[:, on], axis=1)
        base = np.ma.mean(np.ma.concatenate(
            [a[:, :nb // 4], a[:, -nb // 4:]], axis=1), axis=1)
        rows.append(rebin(np.ma.filled(onp - base, 0.0), freqs))
    return np.array(rows)


def b_files():
    return sorted(B_DIR.glob("*.npy"), key=lambda p: int(p.stem.split("_")[1]))


B_FREQS = 1000.0 + 500.0 * (np.arange(4096) + 0.5) / 4096


def load_B_primary():
    """Baseline-subtracted using the extraction metadata's has_burst flags
    (falls back to the whole-file mean where a file has no flags)."""
    meta = json.load(open(B_META))
    flags = defaultdict(dict)
    for m in meta:
        flags[m["filename"]][m["sub_idx"]] = bool(m["has_burst"])
    rows, nofb = [], 0
    for f in b_files():
        a = np.load(f)
        fl = flags.get(f.stem + ".fits", {})
        bidx = [i for i in range(a.shape[0]) if fl.get(i, False)]
        if bidx and len(bidx) < a.shape[0]:
            oidx = [i for i in range(a.shape[0]) if i not in set(bidx)]
            spec = a[bidx].mean(axis=0) - a[oidx].mean(axis=0)
        else:
            spec = a.mean(axis=0)
            nofb += 1
        rows.append(rebin(spec, B_FREQS))
    return np.array(rows), nofb


def load_B_r1():
    """Whole-file average: burst signal diluted ~1e-3 -> noise floor."""
    return np.array([rebin(np.load(f).mean(axis=0), B_FREQS)
                     for f in b_files()])


def load_B_r2():
    """Uniform metadata-free rule: burst subints = broadband power above
    median + 3*1.4826*MAD (fallback: the single max-power subint)."""
    rows, nfall = [], 0
    for f in b_files():
        a = np.load(f)
        p = a.mean(axis=1)
        med = np.median(p)
        mad = np.median(np.abs(p - med)) * 1.4826
        bidx = np.flatnonzero(p > med + 3 * mad) if mad > 0 else np.array([], int)
        if len(bidx) == 0:
            bidx = np.array([int(np.argmax(p))])
            nfall += 1
        mask = np.zeros(a.shape[0], bool)
        mask[bidx] = True
        rows.append(rebin(a[mask].mean(axis=0) - a[~mask].mean(axis=0),
                          B_FREQS))
    return np.array(rows), nfall


def load_shipped(name, nchan=512):
    a = np.load(HERE / name)
    freqs = 1000.0 + 500.0 * (np.arange(nchan) + 0.5) / nchan
    return np.array([rebin(r, freqs) for r in a])


# ---- the paper's pipeline --------------------------------------------------
def read_streams(spectra):
    U, S, _ = np.linalg.svd(spectra - spectra.mean(axis=0),
                            full_matrices=False)
    a = np.angle(U[:, 1] * S[1] + 1j * U[:, 2] * S[2])
    b = np.angle(U[:, 3] * S[3] + 1j * U[:, 4] * S[4])
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    q = lambda x: ((x + np.pi) / (np.pi / 2)).astype(int) % 4
    return q(a), q(d), S[0] ** 2 / (S ** 2).sum()


# ---- features --------------------------------------------------------------
def plug_h(counts):
    tot = sum(counts.values())
    p = np.array([v / tot for v in counts.values()])
    return float(-(p * np.log2(p)).sum())


def redundancy(seq, k):
    h0 = plug_h(Counter(seq))
    ctx = defaultdict(Counter)
    for i in range(k, len(seq)):
        ctx[tuple(seq[i - k:i])][seq[i]] += 1
    n = len(seq) - k
    hk = sum((sum(c.values()) / n) * plug_h(c) for c in ctx.values())
    return 1.0 - hk / h0


def persistence(seq):
    s = np.asarray(seq)
    return float((s[1:] == s[:-1]).mean())


def cv_xent(seq, order, A, folds=6, alpha=0.5):
    n = len(seq)
    b = np.linspace(0, n, folds + 1).astype(int)
    tot, cnt = 0.0, 0
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


def depth_gain(seq, A=4, maxo=5):
    c = [cv_xent(seq, k, A) for k in range(maxo + 1)]
    return c[1] - min(c[2:])


WAVE = {3: 1.0, 2: 0.5, 1: 0.0, 0: -1.0}
DIHEDRAL = [lambda x, k=k, r=r: ((k + (x if not r else (4 - x))) % 4)
            for k in range(4) for r in (False, True)]


def wave_r32(seq):
    best = -1.0
    t = np.arange(32)
    X = np.column_stack([np.sin(2 * np.pi * t / 32),
                         np.cos(2 * np.pi * t / 32), np.ones(32)])
    s = np.asarray(seq)
    nr = len(s) // 32
    if nr < 2:
        return np.nan
    for g in DIHEDRAL:
        v = np.array([WAVE[int(g(x))] for x in s[:nr * 32]]).reshape(nr, 32)
        amp = v.mean(axis=0)
        beta, *_ = np.linalg.lstsq(X, amp, rcond=None)
        best = max(best, np.corrcoef(amp, X @ beta)[0, 1])
    return best


FEATS = ["r1(s1)", "r2(s1)", "pers(s1)", "gain(s1)", "wave(s1)",
         "r1(s2)", "r2(s2)", "pers(s2)", "wave(s2)"]


def window_features(s1, s2):
    w1, w2 = list(s1), list(s2)
    return np.array([
        redundancy(w1, 1), redundancy(w1, 2), persistence(w1),
        depth_gain(w1), wave_r32(w1),
        redundancy(w2, 1), redundancy(w2, 2), persistence(w2),
        wave_r32(w2)])


def cell_features(spectra):
    s1, s2, pc1 = read_streams(spectra)
    nwin = len(s1) // NW
    F = np.array([window_features(s1[j * NW:(j + 1) * NW],
                                  s2[j * NW:(j + 1) * NW])
                  for j in range(nwin)])
    return F, pc1


# ---- run -------------------------------------------------------------------
missing = []
if not A_DIR.exists():
    missing.append(f"cell A: {A_DIR}")
if B_DIR is None or not B_DIR.exists():
    missing.append("cell B: no FAST-121102 spectra directory found")
for f in ("spectra.npy", "spectra_20220912a.npy"):
    if not (HERE / f).exists():
        missing.append(f"cell C/D: {f}")
if missing:
    print("fingerprint square SKIPPED, data not found:")
    for m in missing:
        print("  " + m)
    print("See the docstring for archive DOIs and builders.")
    raise SystemExit(0)

print("loading cells...")
Aspec = load_A()
has_meta = B_META is not None and B_META.exists()
if has_meta:
    Bspec, nofb = load_B_primary()
else:
    Bspec, nofb = None, -1
Cspec = load_shipped("spectra.npy")
Dspec = load_shipped("spectra_20220912a.npy")
bshape = Bspec.shape if Bspec is not None else "(skipped: no metadata)"
print(f"  A {Aspec.shape}  B {bshape} (files w/o burst flags: {nofb})"
      f"  C {Cspec.shape}  D {Dspec.shape}")

cellspecs = [("A:121102@AO", Aspec)]
if Bspec is not None:
    cellspecs.append(("B:121102@FAST", Bspec))
cellspecs += [("C:20201124A@FAST", Cspec), ("D:20220912A@FAST", Dspec)]

cells = {}
for name, spec in cellspecs:
    F, pc1 = cell_features(spec)
    cells[name] = F
    print(f"  {name:<18} windows={len(F)}  PC1 share={pc1 * 100:.1f}%")

vars_ = [F.var(axis=0, ddof=1) for F in cells.values() if len(F) >= 2]
sd = np.sqrt(np.mean(vars_, axis=0))

print("\nPer-feature cell means (window sd in the pooled row):")
print(f"{'feature':<10}" + "".join(f"{n.split(':')[0]:>12}" for n in cells)
      + f"{'pooled sd':>12}")
means = {n: F.mean(axis=0) for n, F in cells.items()}
for i, f in enumerate(FEATS):
    print(f"{f:<10}" + "".join(f"{means[n][i]:>12.4f}" for n in cells)
          + f"{sd[i]:>12.4f}")


def dist(x, y):
    return float(np.median(np.abs(means[x] - means[y]) / sd))


names = list(cells)
print("\nDistance matrix (median |delta|/sd over 9 features):")
print(f"{'':<18}" + "".join(f"{n.split(':')[0]:>10}" for n in names))
for x in names:
    print(f"{x:<18}" + "".join(
        f"{dist(x, y):>10.2f}" if x != y else f"{'-':>10}" for y in names))

if "B:121102@FAST" in cells:
    D_same = dist("A:121102@AO", "B:121102@FAST")
    D_d1 = dist("B:121102@FAST", "C:20201124A@FAST")
    D_d2 = dist("B:121102@FAST", "D:20220912A@FAST")
    print(f"\nD_same (A,B) = {D_same:.2f}   D_diff (B,C) = {D_d1:.2f}   "
          f"D_diff (B,D) = {D_d2:.2f}")
    print("Frozen criteria: POWER GATE min(D_diff) > 1.0; "
          "MATCH if D_same < min(D_diff); INSTRUMENT if D_same > max(D_diff).")
    if min(D_d1, D_d2) <= 1.0:
        print("VERDICT: UNDECIDED (power gate failed)")
    elif D_same < min(D_d1, D_d2):
        print("VERDICT: MATCH - fingerprint travels with the SOURCE; "
              "instrumental class killed")
    elif D_same > max(D_d1, D_d2):
        print("VERDICT: INSTRUMENT-FAVOURED (or source evolution; see design)")
    else:
        print("VERDICT: MIXED")

    print("\nR1 control: cell B with the burst signal erased (whole-file")
    print("  average; ~ms burst over ~6 s of data leaves the receiver noise")
    print("  floor). The telescope signature should appear here if and only")
    print("  if the primary match is carried by burst content.")
    F2, _ = cell_features(load_B_r1())
    means["B2"] = F2.mean(axis=0)
    print(f"  D_same = {float(np.median(np.abs(means['A:121102@AO'] - means['B2']) / sd)):.2f}"
          f"   D_diff(B,C) = {float(np.median(np.abs(means['B2'] - means['C:20201124A@FAST']) / sd)):.2f}"
          f"   D_diff(B,D) = {float(np.median(np.abs(means['B2'] - means['D:20220912A@FAST']) / sd)):.2f}")

print("\nR2: cell B, uniform power-outlier burst extraction (no metadata)")
Bspec3, nfall = load_B_r2()
F3, pc1c = cell_features(Bspec3)
means["B3"] = F3.mean(axis=0)
print(f"  files using single-max-subint fallback: {nfall}/{len(Bspec3)};"
      f"  PC1 share={pc1c * 100:.1f}%")
D_same3 = float(np.median(np.abs(means["A:121102@AO"] - means["B3"]) / sd))
D_d1c = float(np.median(np.abs(means["B3"] - means["C:20201124A@FAST"]) / sd))
D_d2c = float(np.median(np.abs(means["B3"] - means["D:20220912A@FAST"]) / sd))
print(f"  D_same = {D_same3:.2f}   D_diff(B,C) = {D_d1c:.2f}   "
      f"D_diff(B,D) = {D_d2c:.2f}")
if min(D_d1c, D_d2c) <= 1.0:
    print("  R2 verdict: UNDECIDED (power gate)")
elif D_same3 < min(D_d1c, D_d2c):
    print("  R2 verdict: MATCH")
elif D_same3 > max(D_d1c, D_d2c):
    print("  R2 verdict: INSTRUMENT-FAVOURED")
else:
    print("  R2 verdict: MIXED")
