#!/usr/bin/env python3
"""
Build spectra.npy from the public FRB 20201124A archive.

The data files are not distributed with this bundle (see README.md): download
the archive below, then run this script to produce spectra.npy.
Requires astropy in addition to numpy/pandas.

Source: Wang et al. (2023), "Atlas of dynamic spectra of FRB 20201124A",
        Chinese Physics B 32, 029801.  FAST telescope, 1,863 bursts.
        Archive: https://psr.pku.edu.cn/index.php/publications/frb20201124a/

Expects:
    <archive>/CPB/*.ar                       PSRFITS files, one per burst
    <archive>/frb20201124a_parameters.csv    burst list with MJD + filename

Usage:
    python build_spectra.py /path/to/archive
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
params = pd.read_csv(root / "frb20201124a_parameters.csv")

# Time order is the whole point: the analysis measures sequential structure.
params = params.sort_values("MJD").reset_index(drop=True)

spectra = []
for _, row in params.iterrows():
    try:
        with fits.open(str(root / "CPB" / row["filename"])) as hdul:
            sub = hdul["SUBINT"]
            d = sub.data
            nbin = sub.header.get("NBIN", 508)
            nchan = sub.header.get("NCHAN", 512)
            raw = d["DATA"][0]
            wts, scl, offs = d["DAT_WTs"][0], d["DAT_SCL"][0], d["DAT_OFFS"][0]

            # PSRFITS stores data scaled and offset per channel; undo that.
            sp = raw.reshape(nbin, nchan).astype(float)
            ds = sp * scl[np.newaxis, :] + offs[np.newaxis, :]

            # Apply channel weights; zero-weight channels are RFI-flagged.
            ds *= wts[np.newaxis, :]
            ds[:, wts <= 0] = 0

            # One row per burst: the time-averaged spectrum.
            spectra.append(ds.mean(axis=0))
    except Exception:
        continue

spectra = np.array(spectra, dtype=np.float32)
np.save("spectra.npy", spectra)
print(f"wrote spectra.npy  {spectra.shape}")
