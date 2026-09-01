"""Cross-check the load-bearing numbers between paper.tex and SCRIPT_OUTPUT.txt.

Pure text check: each entry requires a paper-side substring in paper.tex AND a
script-side substring in SCRIPT_OUTPUT.txt. Paths are resolved relative to this
file, so it runs from anywhere in the unpacked bundle.
"""
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
tex = io.open(ROOT / "paper.tex", encoding="utf-8").read()
out = io.open(ROOT / "SCRIPT_OUTPUT.txt", encoding="utf-8").read()

checks = [
    ("frame r=0.770",       r"\bnum{0.770}",                  "32    0.770    0.00000"),
    ("frame r=0.281",       r"8  & 0.281",                    "8    0.281    0.11978"),
    ("frame r=0.176",       r"16 & 0.176",                    "16    0.176    0.33522"),
    ("stream1 Zipf",        r"$-0.890$",                      "454     61    0.442   -0.890"),
    ("stream2 Zipf",        r"$\mathbf{-1.057}$",             "772     44    0.187   -1.057"),
    ("MLE s1 2.498/0.855",  r"\bnum{2.498}",                  "2.498    12   0.090   0.855"),
    ("MLE s2 2.386/0.852",  r"\bnum{0.852}",                  "2.386    23   0.095   0.852"),
    ("shuffle z 512",       r"$z = \mathbf{512}$",            "z= 511.5"),
    ("shuffle z 234",       r"$z = \mathbf{234}$",            "z= 233.6"),
    ("gain G1 s1 +9.4",     r"$\mathbf{+9.4}$",               "0.0128 +/-0.0031     9.4"),
    ("gain G1 s2 +8.4",     r"$\mathbf{+8.4}$",               "0.0146 +/-0.0034     8.4"),
    ("depth 18",            r"\bnum{18}",                     "from lag 1) : 18"),
    ("depth s2 = 8",        r"& 8 & 35",                      "from lag 1) : 8"),
    ("session lag32",       r"$\mathbf{z = +6.5}$",           "z =   +6.5"),
    ("circular check",      r"$r = 0.705$",                   "sin(phase):  r = 0.705"),
    ("persistence 75.7",    r"\bnum{75.7\%}",                 "holds state 75.7%"),
    ("marginal-matched 30.8", r"\bnum{30.8\%}",               "its own marginal: 30.8%"),
    ("memoryless z range",  r"$z = 48$--$54$",                "1     1.027     1.834   0.807    48.4"),
    ("iid gap 0.807",       r"0.807",                         "0.807"),
    ("bulk 121102 -0.356",  r"$\mathbf{-0.356}$",             "FRB 121102                           -0.356"),
    ("MI bias floor",       r"\bnum{0.003 bits}",             "= 0.0035 bits"),
    ("field G-test",        r"$G = 91.4$",                    "G = 91.4, dof = 93, p = 0.53"),
    ("x-source channels",   r"$z = 201$--$335$",              "range: z = 201-335"),
    ("x-source wave",       r"$r = 0.496$",                   "0.496"),
    ("matched Eng depth",   r"$2.3 \pm 0.8$",                 "contiguous depth: 2.3 +/- 0.8"),
    ("adversarial mapping", r"$2.5 \pm 0.4$\%",               "2.5 +/- 0.4%"),
    ("math gradient",       r"$2.5 \pm 0.7$\%",               "k=1 2.5 +/- 0.7%"),
]

bad = []
for name, t, s in checks:
    it, isc = (t in tex), (s in out)
    if not (it and isc):
        bad.append((name, it, isc))
    print(f"  {'OK      ' if it and isc else 'MISMATCH'}  {name:20s} paper={str(it):5s} script={isc}")

print(f"\n{len(checks)-len(bad)}/{len(checks)} agree")
if bad:
    print("\nNEEDS ATTENTION:")
    for n, it, isc in bad:
        where = "paper" if not it else "script"
        print(f"  {n}: not found in {where}")

# ── companion experiment outputs (fingerprint square, large-N test) ─────
# Both editions quote these numbers; each must match its committed output.
fp_out = ROOT / "FINGERPRINT_OUTPUT.txt"
dg_out = ROOT / "DEPTHGAIN_OUTPUT.txt"
cross_p = ROOT / "paper_crossdisciplinary.tex"
ctex_p = io.open(cross_p, encoding="utf-8").read() if cross_p.exists() else ""
exp_checks = [
    ("square D_same 1.27",   "distance 1.27",     fp_out, "D_same (A,B) = 1.27"),
    ("square R2 0.70",       r"\bnum{0.70}",      fp_out, "D_same = 0.70"),
    ("square diff 3.78",     "3.78 and 5.47",     fp_out, "D_diff (B,C) = 3.78"),
    ("square diff 5.47",     "3.78 and 5.47",     fp_out, "D_diff (B,D) = 5.47"),
    ("square verdict",       "travel with the source", fp_out,
     "VERDICT: MATCH"),
    ("depth-gain -0.0006",   "$-0.0006$ bits",    dg_out, "gain -0.0006"),
    ("depth-gain gate",      "11{,}553",          dg_out,
     "gain = +0.0251    frozen record: +0.0251  REPRODUCED"),
]
print("\ncompanion experiment outputs vs both editions:")
ebad = 0
for name, t, opath, s in exp_checks:
    if not opath.exists():
        print(f"  MISSING   {name}: {opath.name} not found")
        ebad += 1
        continue
    otxt = io.open(opath, encoding="utf-8").read()
    ok = (t in tex) and (not ctex_p or t in ctex_p) and (s in otxt)
    ebad += 0 if ok else 1
    print(f"  {'OK      ' if ok else 'MISMATCH'}  {name}")
print(f"{len(exp_checks)-ebad}/{len(exp_checks)} experiment checks agree")

# ── cross-disciplinary edition, if present ──────────────────────────────
# Read-only check of the numbers that edition imports from the Cretchen
# recurrent-forms record, against that record's committed results file.
cross = ROOT / "paper_crossdisciplinary.tex"
det = ROOT / "Cretchen" / "frb_20201124a_six_identity_continuous_detector.md"
if cross.exists() and det.exists():
    ctex = io.open(cross, encoding="utf-8").read()
    dmd = io.open(det, encoding="utf-8").read()
    xchecks = [
        ("six-forms 21/30 70.0%", "21 of 30 later events correctly (70.0\\%)", "| late | 30 | 70.0%"),
        ("six-forms median 26.7", "26.7\\%", "matched median `26.7%`"),
        ("six-forms tail 5e-5",   "0.00005", "tail `0.00005`"),
        ("six-forms early 70.3",  "70.3\\%", "| early | 37 | 70.3%"),
    ]
    print("\ncross-disciplinary edition vs Cretchen detector record:")
    xbad = 0
    for name, t, s in xchecks:
        ok = (t in ctex) and (s in dmd)
        xbad += 0 if ok else 1
        print(f"  {'OK      ' if ok else 'MISMATCH'}  {name}")
    print(f"{len(xchecks)-xbad}/{len(xchecks)} cross-edition checks agree")
elif cross.exists():
    print("\ncross-disciplinary edition present; Cretchen record not found -- skipped")
