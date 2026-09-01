# A Transport Protocol in Repeating Fast Radio Bursts, and the Language-Grade Signal Inside It

A reproducible reanalysis of Meucci, *Evidence for Structured Signal
Traffic in Fast Radio Burst Data* (doi:10.5281/zenodo.19443570).

The paper is `paper_crossdisciplinary.pdf` (source:
`paper_crossdisciplinary.tex`, compiles with `pdflatex`, run twice); it
includes cross-disciplinary explanation boxes and an interpretation appendix
kept explicitly separate from the measurements. `paper.pdf` (`paper.tex`) is
the frozen measurement-only edition: the same paper with every box, appendix
and interpretive sentence removed.

## Reproducing every number

One deterministic script prints every number computed for this reanalysis
(the handful of results the paper quotes from the source analysis are cited
as such where they appear):

```
pip install numpy scipy pandas
python frb_three_part.py
```

Every stochastic block carries its own fixed seed, so repeated runs are
byte-identical. `SCRIPT_OUTPUT.txt` is the committed reference output; diff
your run against it. `analysis_scripts/verify_agree.py` cross-checks selected
load-bearing numbers between the papers and the committed outputs; it verifies
paper-output agreement, not the statistical validity of the estimators, which
is the paper's own subject.

Two companion experiment scripts follow the same discipline, each with a
committed reference output:

```
python fingerprint_square.py      # -> FINGERPRINT_OUTPUT.txt
python depth_gain_20240114a.py    # -> DEPTHGAIN_OUTPUT.txt  (~20 min)
```

`fingerprint_square.py` is the cross-telescope 2x2 (does the payload
fingerprint travel with the source or the telescope?); it needs the Arecibo
and FAST 121102 spectra below and skips per-cell if data are absent.
`depth_gain_20240114a.py` is the large-N bulk-parameter test (11,553 bursts);
it validates its estimator by reproducing the FRB 121102 value +0.0251
exactly before measuring anything new.

## Data

The FRB data files are not distributed with this bundle. All inputs come from
public archives:

| file the script expects | contents | where to get it |
|---|---|---|
| `spectra.npy` | FRB 20201124A, 1,863 time-averaged dynamic spectra x 512 channels | FAST atlas, Wang et al. 2023: https://doi.org/10.57760/sciencedb.j00113.00076 — rebuild with `build_spectra.py` (needs `astropy`) |
| `frb20201124a_parameters.csv` | FRB 20201124A burst-parameter table (1,863 rows) | same archive as above |
| `tables1.dat.txt` | FRB 121102 burst catalogue, 1,652 bursts (Li et al. 2021) | https://cdsarc.cds.unistra.fr/ftp/J/Nature/598/267/tables1.dat |
| `spectra_20220912a.npy` | FRB 20220912A, time-averaged dynamic spectra x 512 channels (949 public at this writing) | Zhang et al. 2023: https://doi.org/10.57760/sciencedb.08058 — rebuild with `build_spectra_20220912a.py` |
| `english_sample.txt` | English control corpus (public domain, ships with this bundle) | Project Gutenberg ebook #2701: https://www.gutenberg.org/files/2701/2701-0.txt |
| `math_sample_prose.txt` | mathematical-prose control corpus (ships with this bundle) | Project Gutenberg ebook #58881: https://www.gutenberg.org/files/58881/58881-0.txt |
| `math_sample_puzzles.txt` | structured-mathematics control corpus (ships with this bundle) | Project Gutenberg ebook #16713: https://www.gutenberg.org/files/16713/16713-8.txt |
| `../data/frb121102_arecibo/.../Bursts_npys/` | FRB 121102 Arecibo burst arrays, 404 files, 64 ch x 2432 bins (Hewitt et al. 2022) | https://doi.org/10.5281/zenodo.7181266 — unpack `Bursts_npys.tar` |
| FAST 121102 sub-integration spectra | per-file 60 x 4096 arrays for the square's FAST cell | FAST-FREX PSRFITS (~90 GB): https://doi.org/10.57760/sciencedb.15070 — convert with `build_spectra_frb121102_fast.py` (needs `astropy`) |
| `../data/frb20240114a/FRB20240114A_SuppTab2.csv` | FRB 20240114A burst table, 11,553 rows (Zhang, J.-S., et al. 2025, arXiv:2507.14707) | https://doi.org/10.57760/sciencedb.Fastro.00030 |

The two `build_spectra*.py` scripts turn the raw archive downloads into the
matrices the analysis script reads; only `build_spectra.py` needs `astropy`
beyond the core three packages. If `spectra_20220912a.npy` or
`english_sample.txt` is absent, the corresponding script part (PART X, PART XI)
skips gracefully and prints where to get the file; everything else runs.

## Figures

`figure_scripts/` regenerates the figures into `figures/`. Where a figure
displays z values drawn from a permutation null, the annotations are the
canonical values from `SCRIPT_OUTPUT.txt` (documented in each script's
docstring), so text, tables and figures all quote one generator.
