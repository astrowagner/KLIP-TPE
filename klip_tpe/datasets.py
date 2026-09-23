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

__all__ = ["DATASETS", "INSTRUMENT", "PHOTOMETRY", "data_dir", "fetch"]

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
    # SPHERE IRDIS DB_K12: K1 2.110 um, K2 2.251 um.  The DIT and ND entries are PROVENANCE,
    # not a correction to apply: the distributed flux frames are already on the science
    # frames' scale.  Multiplying the flux frame by dit_science/dit_flux/nd_transmission a
    # second time over-counts the star by 1347x and moves the whole contrast axis with it --
    # it cost tutorial 02 and paper run C a contrast axis until HD 95086 b's published
    # brightness was used to check it.  star_flux = the flux frame's own sum.
    "sphere_hd95086": {"pxscale": 0.01225, "lam_m": 2.110e-6, "lam_m_K2": 2.251e-6, "diam_m": 8.2, "truenorth": 0.0,
                       "dit_science": 96.0, "dit_flux": 0.837464, "nd_transmission": 0.0851},
}

#: Absolute stellar photometry for the sets that HAVE some, in the science frames' own
#: units: ``starphot`` = the star's flux inside ``aperture_px`` of the distributed template's
#: centre.  Feed it to :func:`klip_tpe.instruments.generic.star_flux_from_aperture_photometry`
#: with the template to get the ``star_flux`` :class:`TemplatePSF` wants; without it the
#: contrast axis is in template units, not contrast.
#:
#: ``naco_betapic``: the distributed ``naco_betapic_psf.fits`` is NORMALISED -- its flux
#: inside r = 2.000 px is 1.000000000 to 2e-9 -- so its own counts say nothing about beta
#: Pic's brightness, and ``star_flux`` left unset (= the stamp's sum, 4.349) puts the axis
#: 9.4e5 away from a real contrast.  VIP's metrics tutorial publishes ``starphot = 764939.6``
#: for this very cube, "obtained from the non-coronagraphic PSF before normalization and
#: after rescaling to the integration time used in the coronagraphic observations", i.e. in
#: the aperture the template is normalised in.  CHECKED, not assumed: with it, beta Pic b
#: measures dL' = 7.79 against the 8.01 +/- 0.16 Absil et al. (2013) published from these
#: same data -- 1.2 sigma.  See ``scripts/check_betapic_contrast.py`` and docs/FLUX_CALIBRATION.md.
#:
#: The other sets are calibrated differently and are NOT listed here: ``sphere_hd95086``
#: distributes flux frames already on the science scale (``star_flux`` = the frame's own
#: sum), and the JWST set has no stellar photometry at all.
#: ``hip65426_f444w``: there is no off-axis stellar image anywhere in ERS 1386 -- HIP 65426
#: and the reference star phi Cen are both behind MASK335R in every exposure -- so the star
#: is given as a FLUX DENSITY and the frames' own MJy/sr calibration converts it
#: (:func:`klip_tpe.stpsf_psf.star_flux_from_flux_density`).  0.4026 Jy is synthetic
#: photometry of a Planck spectrum at Carter et al. (2023)'s Teff = 8600 K through the F444W
#: bandpass, normalised to 2MASS Ks = 6.771 (A2V, J = 6.826, J-K = 0.055); +/- 3%, of which
#: 1.9% is Ks and <2% the unknown detail of the system response shape.  Ks and F444W are both
#: in the Rayleigh-Jeans tail, and the JWST flux calibrators are A dwarfs too, so the colour
#: term against the standards is negligible.
PHOTOMETRY = {
    "naco_betapic": {
        "starphot": 764939.6, "aperture_px": 2.0,
        "ref": "VIP tutorial 04_metrics (vip.readthedocs.io), NACO L' beta Pic; "
               "off-axis PSF rescaled to the coronagraphic DIT",
        "check": "beta Pic b -> dL' 7.79 vs 8.01 +/- 0.16 (Absil et al. 2013, A&A 559, L12)",
    },
    "hip65426_f444w": {
        "flux_density_jy": 0.40259, "flux_density_err_frac": 0.03, "filter": "F444W",
        "ref": "synthetic photometry: Planck(Teff=8600 K, Carter et al. 2023) through the "
               "STPSF F444W bandpass, normalised to 2MASS Ks = 6.771",
        # Transmissive losses of the coronagraphic optics that the model PSF does not carry
        # (COM sapphire substrate, BaF2 Lyot substrate): NONE are left to supply.  PHOTMJSR
        # for PUPIL=MASKRND (2.486, against ~0.4 for CLEAR imaging in the same filter) is
        # derived from standards observed through this very optical train, so the MJy/sr
        # in a calints file already puts an off-mask point source at its true flux; the
        # only things a model adds are the PSF's shape and the occulter's T(rho).  The
        # "no occulting-mask column in the photom file" argument confused the occulter
        # (spatially varying, not in photom) with the substrate (uniform, in it).
        #
        # HISTORY: 0.561, "anchored on HIP 65426 b", until 2026-09-16.  It was not an
        # optics number at all: the loader's sigma-clip repair had median-filtered the
        # companion's core down to 39% of its peak while the fakes were injected AFTER the
        # repair and kept theirs, so the companion looked 1.9x too faint relative to them,
        # and 0.561 = 1/1.9 was what hid it.  With the repair fixed and 1.0 here, HIP 65426 b
        # measures dF444W = 8.796 +/- 0.092 against Carter et al. (2023)'s 8.703 +/- 0.055 --
        # 1.0 sigma, with nothing tuned -- an INDEPENDENT check of the whole axis (S, PIXAR_SR,
        # EE, T(rho), centring, recovery).  See scripts/check_hip65426_contrast.py and
        # docs/FLUX_CALIBRATION.md.  (It read 8.74 until 2026-09-22, when the injector stopped
        # rotating the template by the source's azimuth; that moved it +0.056 mag, away from the
        # published value but well inside the combined error, and the physics decided it, not
        # the agreement.)
        "optics_transmission": 1.0,
        "optics_transmission_source": "PHOTMJSR (PUPIL=MASKRND) already carries the coronagraphic optics",
        # GEOMETRY rather than photometry, but it lives here because it is the same
        # per-programme calibration block and the flux check depends on it.  CRPIX
        # (149.2, 173.6) is the APERTURE reference point -- identical in every file of the
        # programme, dithers included -- and misses the star by 0.78 px, which puts the
        # companion inside its own separation and mismatches its KLIP throughput against
        # the fakes injected to calibrate it.  Solved from the companion's position in each
        # roll against Carter et al. (2023)'s F444W astrometry (820 mas, 149.9 deg; see
        # scripts/check_hip65426_contrast.py); the two rolls agree to 0.12 px.  The value
        # before 2026-09-16, (150.54, 172.98), was solved on median-filtered frames whose
        # smeared companion peak had moved: the rolls disagreed by 0.71 px then.  Replace
        # with spaceKLIP's STARCENX/Y when those products exist.
        "star_center": (149.65, 172.96), "star_center_source": "solved from both rolls (agree to 0.12 px)",
        "check": "HIP 65426 b -> dF444W vs 8.703 +/- 0.055 (Carter et al. 2023, ApJL 951, L20, Table 3)",
    },
    # ---- the same star through MIRI's three 4QPM filters (ERS 1386, obs 4/5 at F1140C).
    # Same star, same method, and deliberately anchored to the F444W entry above rather
    # than computed from scratch: the Planck-through-the-bandpass recipe reproduces that
    # value to -3.3% (photon-weighted) / -2.4% (energy-weighted), which is inside its own
    # +/-3%, and the residual is the 2MASS zero-point and effective-wavelength convention.
    # Taking the RATIO to F444W cancels that convention and leaves only the model's shape
    # between 4.4 and 11.3 um -- a Rayleigh-Jeans tail, where Planck and a real atmosphere
    # differ by a per cent or two.  Teff = 8600 K is Carter et al. (2023)'s own PHOENIX fit
    # (8600 +/- 200 K, L = 16 +/- 1 Lsun), so both entries rest on the same model.
    #
    # The photosphere is the right thing to use here: the only excess these authors find is
    # 3.5 sigma at 24 um (Chen et al. 2012) fitted with T_dust ~ 300 K, which contributes
    # far less at 11 um than at 24 and is marginal even there.
    #
    # Cross-check, independent of the recipe: Carter et al. (2023) Table 3 gives the companion
    # at dF1140C = 8.264 +/- 0.021 (a contrast of 4.95e-4) and (7.40 +/- 1.16)e-19 W m^-2 um^-1,
    # i.e. 31.5 uJy at 11.3 um -- which puts THEIR star at 0.0637 Jy.  This entry is 8.3%
    # higher, against their +/-3.5% on m* and its own +/-5%: 1.3 sigma.  Their quoted
    # background-limited sensitivity of ~2.7 uJy is then a contrast floor of 3.9e-5, in line
    # with their ~5e-5 beyond 3".  (An earlier version of this note called ~2e-4 "the
    # companion" -- that is their 5-sigma LIMIT at 1", and the companion is 2.5x brighter.)
    **{f"hip65426_{f.lower()}": {
        "flux_density_jy": s, "flux_density_err_frac": 0.05, "filter": f,
        "ref": "synthetic photometry: Planck(Teff=8600 K, Carter et al. 2023 PHOENIX fit) "
               "through the STPSF bandpass, normalised to 2MASS Ks = 6.771, ratio-anchored "
               "to the F444W entry above",
        "optics_transmission": 1.0,
        "optics_transmission_source": "PHOTMJSR of the coronagraphic mode already carries its optics",
        "check": "Carter et al. (2023, ApJL 951, L20) Table 3, F1140C: dF1140C = 8.264 +/- 0.021 "
                 "and 7.40e-19 W m^-2 um^-1 (31.5 uJy) imply a star of 0.0637 Jy, 8.3% below "
                 "this entry (1.3 sigma); their ~2.7 uJy sensitivity is a 3.9e-5 contrast floor",
    } for f, s in (("F1065C", 0.07813), ("F1140C", 0.06899), ("F1550C", 0.03739))},
    # ---- HR 8799 through the same three MIRI filters (GO 1194, Boccaletti et al. 2024,
    # A&A 686, A33).  Computed from scratch rather than anchored, because this star has an
    # INDEPENDENT check available and passing it is worth more than the anchor: those authors
    # interpolate the stellar flux density at 15.5 um from WISE and AKARI photometry and get
    # 154.2 mJy, and Planck(7600 K) through the F1550C bandpass normalised to 2MASS
    # Ks = 5.240 gives 155.8 -- agreement to 1.0%, with nothing in common between the two
    # routes.  Over Teff = 7200-7800 K the F1550C value moves 159.6 -> 154.0 mJy, so the
    # agreement also pins the inputs: an error in Ks would scale the answer straight through.
    **{f"hr8799_{f.lower()}": {
        "flux_density_jy": s, "flux_density_err_frac": 0.04, "filter": f,
        "ref": "synthetic photometry: Planck(Teff=7600 K, A5V) through the STPSF bandpass, "
               "normalised to 2MASS Ks = 5.240",
        "optics_transmission": 1.0,
        "optics_transmission_source": "PHOTMJSR of the coronagraphic mode already carries its optics",
        "check": "F1550C 155.8 mJy against Boccaletti et al. (2024, A&A 686, A33) 154.2 mJy "
                 "interpolated from WISE + AKARI -- 1.0%, by an independent route",
    } for f, s in (("F1065C", 0.3244), ("F1140C", 0.2866), ("F1550C", 0.1558))},
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
