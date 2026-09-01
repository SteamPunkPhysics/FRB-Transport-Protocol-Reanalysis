"""Build spectra_20220912a.npy from the public FAST archive of FRB 20220912A.

Source data: "FAST Observations of FRB 20220912A: Burst Properties and
Polarization Characteristics" (Zhang et al. 2023), doi:10.57760/sciencedb.08058.
Download the dynamic-spectra .npy files (named 220912-NNN-MJD.npy; each a
time x frequency array with 512 frequency channels, dedispersed at
DM = 220.0 pc/cm^3) into a directory and pass that directory as the first
argument (default: ../data/frb20220912a).

Each file contributes its time-averaged spectrum (mean over the time axis);
rows are sorted by the MJD encoded in the filename. The original analysis used
the 894 spectra public at the time; the archive has since grown, and this
script uses whatever it finds.
"""
import sys
from pathlib import Path
import numpy as np

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../data/frb20220912a")
files = sorted(DATA.glob("*.npy"))
if not files:
    raise SystemExit(f"no .npy files found in {DATA}")
print(f"{len(files)} .npy files in {DATA}")
mjds = [float(f.stem.split("-")[2]) for f in files]
specs = [np.load(f).mean(axis=0) for f in files]
spectra = np.array(specs, dtype=np.float32)[np.argsort(mjds)]
out = Path(__file__).resolve().parent / "spectra_20220912a.npy"
np.save(out, spectra)
print(f"{out.name} written: {spectra.shape}")
