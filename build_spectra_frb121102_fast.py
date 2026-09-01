"""Build per-file FAST FRB 121102 subint spectra for fingerprint_square.py.

Source data: the FAST-FREX release of the Li et al. (2021) FRB 121102
observations, doi:10.57760/sciencedb.15070 — PSRFITS files named
FRB20121102_NNNN.fits, each ~244 MB with 60 sub-integrations x 1024 time
bins x 4096 frequency channels (1.0-1.5 GHz). The full archive is ~90 GB;
this script processes whatever .fits files it finds.

For each file it writes <name>.npy of shape (nsub, 4096): one spectrum per
sub-integration. Where a sub-integration contains a burst (peak SNR of the
frequency-summed time series >= 3), the spectrum is averaged over the
contiguous SNR > 2 window around the peak (minimum +/-2 bins); otherwise
over all time bins. This reproduces the extraction used for the committed
FINGERPRINT_OUTPUT.txt. fingerprint_square.py's R2 variant needs only these
.npy files (its burst-subint selection is metadata-free).

Usage: python build_spectra_frb121102_fast.py <fits_dir> [out_dir]
Requires astropy. Default out_dir: ./frb121102_fast_spectra
"""
import sys
import numpy as np
from pathlib import Path
from astropy.io import fits

SNR_THRESHOLD = 3.0

fits_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    Path(__file__).resolve().parent / "frb121102_fast_spectra"
out_dir.mkdir(exist_ok=True)

files = sorted(fits_dir.glob("*.fits"))
if not files:
    raise SystemExit(f"no .fits files in {fits_dir}")

for fpath in files:
    try:
        with fits.open(str(fpath), memmap=True) as hdul:
            sub = hdul['SUBINT']
            d = sub.data
            hdr = sub.header
            nbin = hdr.get('NBIN', 1024)
            nchan = hdr.get('NCHAN', 4096)
            rows = []
            for isub in range(len(d)):
                raw = d['DATA'][isub].squeeze()
                if raw.ndim == 1:
                    raw = raw.reshape(nbin, nchan)
                elif raw.ndim > 2:
                    raw = raw.reshape(-1, nchan)
                scl = d['DAT_SCL'][isub]
                offs = d['DAT_OFFS'][isub]
                wts = d['DAT_WTs'][isub] if 'DAT_WTs' in d.names else d['DAT_WTS'][isub]
                cal = raw.astype(np.float32) * scl[None, :] + offs[None, :]
                cal[:, wts <= 0] = 0
                good = wts > 0
                if good.sum() == 0:
                    rows.append(cal.mean(axis=0))
                    continue
                ts = cal[:, good].mean(axis=1)
                std = np.std(ts)
                if std == 0:
                    rows.append(cal.mean(axis=0))
                    continue
                snr = (ts - np.median(ts)) / std
                peak = int(np.argmax(snr))
                if snr[peak] >= SNR_THRESHOLD:
                    mask = snr > 2.0
                    mask[peak] = True
                    start = peak
                    while start > 0 and mask[start - 1]:
                        start -= 1
                    end = peak
                    while end < len(mask) - 1 and mask[end + 1]:
                        end += 1
                    start = max(0, min(start, peak - 2))
                    end = min(len(snr) - 1, max(end, peak + 2))
                    rows.append(cal[start:end + 1, :].mean(axis=0))
                else:
                    rows.append(cal.mean(axis=0))
            np.save(out_dir / (fpath.stem + ".npy"),
                    np.array(rows, dtype=np.float32))
            print(f"{fpath.name}: {len(rows)} subint spectra")
    except Exception as e:
        print(f"{fpath.name}: FAILED ({e})")
