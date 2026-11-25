"""A synthetic reducer for unit tests and demos, with no telescope data but the same API.

TPE, calibration, validation and checkpointing can be exercised with it in seconds.  The
"reduction" makes a noise image whose injected-source throughput and residual noise depend smoothly,
and multimodally, on the parameters, with a whole ``k_klip`` optimum that differs per partition and
a decoy: a small ``k`` gives high throughput but high noise, like a real trade-off between over- and
under-subtraction.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .injection import GaussianPSF, inject_sources
from .metrics import Source
from .reducer import Dataset, KLIPReducer, ReductionRequest, ReductionResult, Reducer, _flag, _zone_ext

__all__ = ["SyntheticReducer", "make_synthetic_partitions", "synthetic_klip_dataset"]

#: Settings every request carries that the synthetic images do not depend on.  They stay out of
#: :meth:`SyntheticReducer._rng`'s seed, so a synthetic run draws what it drew before requests
#: carried them (the eigenvalue cut and the stamps' normalization aperture since 2026-11-25).  Its
#: Gaussian model sums to 1 over its stamp and takes no aperture.
_UNSEEDED = ("klip_eig_cut", "ee_aperture")


def _outside_zone(rr: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    """Return the pixels outside the zone a KLIP reduction would cover.

    The zone is the annulus padded by 2 px on each edge the runner marks (``pad_in`` and ``pad_out``,
    both by default), and widened by ``zone_in`` and ``zone_out`` as the KLIP reducer widens it.

    Parameters
    ----------
    rr : ndarray, shape (ny, nx)
        Each pixel's radius, in pixels.
    p : dict
        The reduction's parameters.

    Returns
    -------
    outside : ndarray of bool, shape (ny, nx)
        True outside the zone.
    """
    pin = 2.0 if _flag(p.get("pad_in", True)) else 0.0
    pout = 2.0 if _flag(p.get("pad_out", True)) else 0.0
    z_in, z_out = _zone_ext(p)
    return (rr < float(p.get("inrad", 0)) - pin - z_in) | (rr > float(p.get("outrad", 45)) + pout + z_out)


class SyntheticReducer(Reducer):
    """Reducer of a synthetic night, whose throughput and noise are smooth functions of the parameters.

    Parameters
    ----------
    size : int, default 100
        Width of the images, in pixels.
    pxscale : float, default 0.05
        Plate scale, in arcsec per pixel.
    fwhm : float, default 4.0
        FWHM of the PSF, in pixels.
    k_opt : int, default 8
        The KL modes of best throughput.
    bin_opt : float, default 12.0
        The binning of best throughput and lowest noise.
    filter_opt : float, default 10.0
        The high-pass of best throughput.
    noise : float, default 1.0
        The noise's base level.
    star_flux : float, default 1e6
        The star's flux, the injection model's flux unit.
    seed : int, default 0
        Seed of the noise, with the request.
    partition : object, default None
        The partition's id.
    texp : float, default 1000.0
        Exposure time, whose square root is the partition's weight.
    fm : bool, default False
        Whether to advertise ``supports_fm``, a request with ``fm_sources`` then also returning the
        noiseless, throughput-scaled model response, the analogue of KLIP-FM.
    """
    name = "synthetic"
    supports_kscan = True

    def __init__(self, size: int = 100, pxscale: float = 0.05, fwhm: float = 4.0, k_opt: int = 8,
                 bin_opt: float = 12.0, filter_opt: float = 10.0, noise: float = 1.0, star_flux: float = 1e6,
                 seed: int = 0, partition: Any = None, texp: float = 1000.0, fm: bool = False):
        self.size, self.pxscale, self.fwhm = int(size), float(pxscale), float(fwhm)
        self.supports_fm = bool(fm)
        self.lam_over_d_px = self.fwhm / 1.028
        self.k_opt, self.bin_opt, self.filter_opt = k_opt, bin_opt, filter_opt
        self.noise0, self.star_flux, self.seed = noise, star_flux, seed
        self.partition = partition
        self.texp = texp
        self.model = GaussianPSF(self.fwhm, star_flux=star_flux)
        self.angles = np.linspace(-20, 20, 200)

    def frame_angles(self, pid=None):
        """Return the frames' parallactic angles, 200 from -20 to 20 degrees.

        Parameters
        ----------
        pid : object, default None
            The partition, unused.

        Returns
        -------
        angles : ndarray
            The angles, in degrees.
        """
        return self.angles

    def partition_weight(self, pid=None) -> float:
        """Return the partition's weight, the square root of its exposure time.

        Parameters
        ----------
        pid : object, default None
            The partition, unused.

        Returns
        -------
        weight : float
            The weight.
        """
        return float(np.sqrt(self.texp))

    # smooth "physics"
    def throughput(self, k: float, bin_: float, filt: float) -> float:
        """Return the synthetic throughput of a configuration, peaked near ``k_opt``, ``bin_opt`` and ``filter_opt``.

        Parameters
        ----------
        k : float
            Number of KL modes.
        bin_ : float
            The temporal binning.
        filt : float
            Width of the high-pass, in pixels.

        Returns
        -------
        throughput : float
            The throughput.
        """
        t_k = 0.25 + 0.75 * np.exp(-0.5 * ((k - self.k_opt) / 4.0) ** 2) * (1 - 0.5 * np.exp(-k / 3.0))
        t_b = 0.7 + 0.3 * np.exp(-0.5 * ((bin_ - self.bin_opt) / 6.0) ** 2)
        t_f = 0.8 + 0.2 * np.cos(np.pi * (filt - self.filter_opt) / 12.0) ** 2
        return float(t_k * t_b * t_f)

    def noise_level(self, k: float, bin_: float, filt: float, angsep: float = 0.5) -> float:
        """Return the synthetic noise of a configuration, high at small ``k`` and growing mildly at large ``k``.

        Parameters
        ----------
        k : float
            Number of KL modes.
        bin_ : float
            The temporal binning.
        filt : float
            Width of the high-pass, in pixels.
        angsep : float, default 0.5
            The exclusion, in units of the arc.

        Returns
        -------
        noise : float
            The noise's standard deviation.
        """
        n_k = 1.0 + 3.0 * np.exp(-k / 4.0) + 0.02 * k       # under-subtraction at small k, mild growth at large k
        n_b = 1.0 + 0.3 * np.abs(bin_ - self.bin_opt) / 10.0
        n_a = 1.0 + 0.5 * (angsep - 1.0) ** 2
        return float(self.noise0 * n_k * n_b * n_a)

    def _rng(self, req: ReductionRequest) -> np.random.Generator:
        """Return the generator of a request, seeded by its parameters, sources, k-scan flag, seed and partition.

        The parameters in :data:`_UNSEEDED`, which the synthetic images do not depend on, stay out of
        the seed.

        Parameters
        ----------
        req : ReductionRequest
            The request.

        Returns
        -------
        rng : numpy.random.Generator
            The generator, the same for the same request.
        """
        key = repr((sorted((k, round(float(v), 6) if isinstance(v, (int, float)) else str(v))
                           for k, v in req.params.items() if k not in _UNSEEDED),
                    None if not req.injections else [s.as_tuple() for s in req.injections],
                    req.k_scan, self.seed, str(self.partition)))
        h = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
        return np.random.default_rng(h)

    def matched_filter_kernel(self, rho_as: float):
        """Return the injection model's matched filter at a separation.

        Parameters
        ----------
        rho_as : float
            The separation, in arcsec.

        Returns
        -------
        kernel : ndarray
            The kernel.
        """
        return self.model.matched_filter_kernel(rho_as, self.fwhm)

    def _model_image(self, p: Dict[str, Any], k: int, sources) -> np.ndarray:
        """Return the noiseless model response, the sources at ``throughput * contrast`` on zeros, NaN outside the zone.

        Parameters
        ----------
        p : dict
            The parameters.
        k : int
            Number of KL modes.
        sources : sequence of Source
            The sources.

        Returns
        -------
        image : ndarray, shape (size, size)
            The response, float32.
        """
        bin_ = float(p.get("bin", 10)); filt = float(p.get("filter", 10))
        thr = self.throughput(k, bin_, filt)
        frame = np.zeros((1, self.size, self.size), np.float32)
        src = [Source(s.rho, s.theta, s.contrast * thr) for s in sources]
        frame = inject_sources(frame, np.zeros(1), src, self.model, self.pxscale, truenorth=0.0, copy=False)
        img = np.asarray(frame[0], np.float32).copy()
        yy, xx = np.mgrid[0:self.size, 0:self.size]
        c = (self.size - 1) / 2.0
        rr = np.hypot(xx - c, yy - c)
        img[_outside_zone(rr, p)] = np.nan
        return img

    def _image(self, p: Dict[str, Any], k: int, injections, rng) -> np.ndarray:
        """Return a synthetic reduced image: noise, a few static speckles and the injected sources at their throughput.

        Parameters
        ----------
        p : dict
            The parameters.
        k : int
            Number of KL modes.
        injections : sequence of Source or None
            The sources injected.
        rng : numpy.random.Generator
            The generator.

        Returns
        -------
        image : ndarray, shape (size, size)
            The image, float32, NaN outside the zone.
        """
        bin_ = float(p.get("bin", 10)); filt = float(p.get("filter", 10)); asep = float(p.get("angsep", 0.5))
        img = rng.standard_normal((self.size, self.size)) * self.noise_level(k, bin_, filt, asep)
        # a few static speckles so clean-subtraction has something to do
        yy, xx = np.mgrid[0:self.size, 0:self.size]
        c = (self.size - 1) / 2.0
        for (dx, dy, amp) in ((9, 4, 6.0), (-7, 8, 4.0), (3, -11, 5.0)):
            img += amp * self.noise0 * np.exp(-0.5 * ((xx - c - dx) ** 2 + (yy - c - dy) ** 2) / (self.fwhm / 2.3548) ** 2)
        rr = np.hypot(xx - c, yy - c)
        img[_outside_zone(rr, p)] = np.nan
        if injections:
            thr = self.throughput(k, bin_, filt)
            frame = np.zeros((1, self.size, self.size), np.float32)
            src = [Source(s.rho, s.theta, s.contrast * thr) for s in injections]
            frame = inject_sources(frame, np.zeros(1), src, self.model, self.pxscale, truenorth=0.0, copy=False)
            img = img + frame[0]
        return img.astype(np.float32)

    def reduce(self, req: ReductionRequest) -> ReductionResult:
        """Reduce a request, an image, or one per k up to ``k_klip`` for a k-scan.

        Parameters
        ----------
        req : ReductionRequest
            The request.

        Returns
        -------
        result : ReductionResult
            The image, the partition's weight, ``k`` and ``partition``, and with ``fm`` the model response.
        """
        rng = self._rng(req)
        p = req.params
        k = int(max(round(float(p.get("k_klip", 5))), 1))
        fm_img = None
        if req.k_scan:
            img = np.stack([self._image(p, kk, req.injections, rng) for kk in range(1, k + 1)])
        else:
            img = self._image(p, k, req.injections, rng)
            if self.supports_fm and req.fm_sources:
                fm_img = self._model_image(p, k, req.fm_sources)
        return ReductionResult(img, self.partition_weight(), {"k": k, "partition": self.partition}, fm_image=fm_img)


def make_synthetic_partitions(ids: Sequence[Any], k_opts: Optional[Sequence[int]] = None, **kw) -> Dict[Any, SyntheticReducer]:
    """Make one :class:`SyntheticReducer` per partition, each with its own ``k_opt`` and seed.

    Parameters
    ----------
    ids : sequence
        The partitions' ids.
    k_opts : sequence of int or None, default None
        Each partition's ``k_opt``.  None takes 6, 9, 12 and so on.
    kw : dict
        Keywords of :class:`SyntheticReducer`.

    Returns
    -------
    reducers : dict
        The reducers, by partition id.
    """
    k_opts = k_opts or [6 + 3 * i for i in range(len(ids))]
    return {pid: SyntheticReducer(k_opt=int(k), seed=i, partition=pid, **kw)
            for i, (pid, k) in enumerate(zip(ids, k_opts))}


def synthetic_klip_dataset(nframes: int = 60, size: int = 64, pa_span: float = 40.0, seed: int = 1,
                           speckle_amp: float = 50.0, noise: float = 1.0) -> Dataset:
    """Make a tiny pupil-tracking cube with a quasi-static speckle field, for testing the real :class:`~klip_tpe.reducer.KLIPReducer` end to end.

    Parameters
    ----------
    nframes : int, default 60
        Number of frames.
    size : int, default 64
        Width of the frames, in pixels.
    pa_span : float, default 40.0
        Field rotation over the sequence, in degrees.
    seed : int, default 1
        Seed of the speckles and the noise.
    speckle_amp : float, default 50.0
        Amplitude of the speckles.
    noise : float, default 1.0
        Standard deviation of the white noise.

    Returns
    -------
    dataset : Dataset
        The cube, its angles, frame tags (``corrs``, ``noises``, ``coronoise``) and exposure time.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    rr = np.hypot(xx - c, yy - c)
    static = np.zeros((size, size))
    for _ in range(25):
        x0, y0 = rng.uniform(c - 20, c + 20, 2)
        static += rng.uniform(0.3, 1.0) * np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / 1.5 ** 2)
    static *= speckle_amp * np.exp(-rr / 15.0)
    cube = np.empty((nframes, size, size), np.float32)
    for i in range(nframes):
        drift = 1.0 + 0.1 * np.sin(i / 7.0)
        cube[i] = static * drift + rng.standard_normal((size, size)) * noise
    angles = np.linspace(-pa_span / 2, pa_span / 2, nframes)
    tags = {"corrs": np.clip(rng.normal(0.98, 0.01, nframes), 0, 1),
            "noises": np.abs(rng.normal(1.0, 0.1, nframes)), "coronoise": np.abs(rng.normal(1.0, 0.1, nframes))}
    return Dataset(cube, angles, tags, texp=float(nframes), name=f"synthetic{seed}")
