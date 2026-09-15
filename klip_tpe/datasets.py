"""Public example data sets for the tutorials (downloaded on first use, cached).

* ``naco_betapic`` -- VLT/NACO L' ADI sequence of beta Pictoris (61 frames, 101x101 px,
  0.02719"/px) with an off-axis PSF and derotation angles, the VIP tutorial data set
  (Absil et al. 2013).  beta Pic b sits at ~0.45", PA ~210 deg in this epoch.
* ``sphere_sao206462`` -- VLT/SPHERE IRDIS sequence of SAO 206462 with a *reference-star*
  cube (RDI / ARDI), from the same repository.
* ``nircam_pds70_f480m`` / ``nircam_pds70_f187n`` -- JWST/NIRCam PDS 70 roll images with
  PA arrays (VIP tutorial data; two frames only -- an illustration of the JWST path, not
  an optimization test case).
* ``sphere_hd95086`` -- VLT/SPHERE IRDIS DB_K12 sequence of HD 95086 (2019-04-13, ESO
  1100.C-0481, 63 frames x 2 channels, 241x241 px crops, 33.8 deg of rotation) with the
  unsaturated flux frames; HD 95086 b at 0.62", PA ~145 deg.  Reduced with K. Wagner's
  IDL SPHERE pipeline; distributed from the klip-tpe GitHub release ``tutorial-data``.

The VIP files come from https://github.com/vortex-exoplanet/VIP_extras (data distribution
of the VIP project); see that repository for provenance.  The cache directory is
``$KLIP_TPE_DATA`` or ``~/.klip_tpe/data``; ``KLIP_TPE_DATA_URL`` overrides the base URL
of the klip-tpe-hosted sets (e.g. a local mirror).
"""
from __future__ import annotations

import os
import sys
import urllib.request
from typing import Dict

__all__ = ["DATASETS", "data_dir", "fetch"]

_BASE = "https://raw.githubusercontent.com/vortex-exoplanet/VIP_extras/master/datasets/"
_KT_BASE = os.environ.get("KLIP_TPE_DATA_URL",
                          "https://github.com/astrowagner/KLIP-TPE/releases/download/tutorial-data/")

DATASETS: Dict[str, Dict[str, str]] = {
    "naco_betapic": {
        "cube": "naco_betapic_cube_cen.fits",
        "angles": "naco_betapic_derot_angles.fits",
        "psf": "naco_betapic_psf.fits",
    },
    "sphere_sao206462": {
        "cube": "sphere_SAO206462_cube.fits",
        "angles": "sphere_SAO206462_derot_angles.fits",
        "ref_cube": "sphere_SAO206462_ref_cube.fits",
    },
    "nircam_pds70_f480m": {
        "cube": "nircam_PDS70_cube_F480M.fits",
        "angles": "nircam_PDS70_pa_F480M.fits",
    },
    "nircam_pds70_f187n": {
        "cube": "nircam_PDS70_cube_F187N.fits",
        "angles": "nircam_PDS70_pa_F187N.fits",
    },
    "sphere_hd95086": {
        "cube_K1": "hd95086_irdis_K1_cube.fits",
        "cube_K2": "hd95086_irdis_K2_cube.fits",
        "psf_K1": "hd95086_irdis_K1_psf.fits",
        "psf_K2": "hd95086_irdis_K2_psf.fits",
        "angles": "hd95086_irdis_angles.fits",
    },
}
#: which base URL serves each set
_SOURCE = {name: ("kt" if name.startswith("sphere_hd95086") else "vip") for name in DATASETS}

#: instrument constants that go with the data sets (arcsec/px, wavelength m, aperture m)
INSTRUMENT = {
    "naco_betapic": {"pxscale": 0.02719, "lam_m": 3.8e-6, "diam_m": 8.2, "truenorth": 0.0},
    "sphere_sao206462": {"pxscale": 0.01225, "lam_m": 1.593e-6, "diam_m": 8.2, "truenorth": 0.0},
    "nircam_pds70_f480m": {"pxscale": 0.063, "lam_m": 4.83e-6, "diam_m": 6.5, "truenorth": 0.0},
    "nircam_pds70_f187n": {"pxscale": 0.031, "lam_m": 1.874e-6, "diam_m": 6.5, "truenorth": 0.0},
    # SPHERE IRDIS DB_K12: K1 2.110 um, K2 2.251 um; science DIT 96 s without ND, flux DIT 0.837464 s with ND_1.0
    "sphere_hd95086": {"pxscale": 0.01225, "lam_m": 2.110e-6, "lam_m_K2": 2.251e-6, "diam_m": 8.2, "truenorth": 0.0,
                       "dit_science": 96.0, "dit_flux": 0.837464, "nd_transmission": 0.0851},
}


def data_dir() -> str:
    d = os.environ.get("KLIP_TPE_DATA") or os.path.join(os.path.expanduser("~"), ".klip_tpe", "data")
    os.makedirs(d, exist_ok=True)
    return d


def fetch(name: str, quiet: bool = False) -> Dict[str, str]:
    """Download (once) the files of data set ``name`` and return ``{role: local path}``
    (roles: ``cube``, ``angles``, and ``psf`` / ``ref_cube`` when the set has them)."""
    if name not in DATASETS:
        raise KeyError(f"unknown data set {name!r}; known: {sorted(DATASETS)}")
    out = {}
    d = data_dir()
    for role, fn in DATASETS[name].items():
        path = os.path.join(d, fn)
        if not os.path.exists(path) or os.path.getsize(path) < 2880:
            url = (_KT_BASE if _SOURCE.get(name) == "kt" else _BASE) + fn
            if not quiet:
                print(f"downloading {url} -> {path}", file=sys.stderr)
            # Every fetch gets its own scratch file, so two of them running at once against
            # a cold cache cannot write the same partial download.  They would otherwise
            # interleave their bytes and one would rename the mixture into place -- a cube
            # of exactly the right size and the wrong contents, which nothing downstream
            # would notice.  mkstemp rather than a pid suffix: threads in one process share
            # a pid, and the paper rerun's two halves read the same cubes either way.
            import tempfile
            fd, tmp = tempfile.mkstemp(dir=d, prefix=fn + ".", suffix=".part")
            os.close(fd)
            try:
                urllib.request.urlretrieve(url, tmp)
                os.replace(tmp, path)          # atomic: readers see old or new, never partial
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
        out[role] = path
    return out
