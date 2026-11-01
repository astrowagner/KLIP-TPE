"""klip-tpe's NEAR-style reduction of one night on the card (CuPy), step by step.

The chain gpu_bench.py / gpu_validate.py have run since 2026-10-12, lifted out of the benchmark so a
reducer can call it: the frames' selection, the injection of sources, the high-pass, the binning,
annular KLIP in the Gram-matrix formulation, the residuals' high-pass, the derotation and the
noise-weighted combination.  Everything runs on the device; the night's image comes back.

Numerics, as validated (gpu_benchmark_plan.md sections 6, 9, 12; gpu_validate.py --check on every
run, <= 1e-5 of the image noise): frames in float32 on the device, the high-pass in the frames'
precision, the Gram matrix in float64, the eigendecomposition in float64 with cuSOLVER's batched
divide-and-conquer solver (``eigh="xbatched64"``), the projection, derotation and combination in
float64, klip-tpe's absolute 1e-6 as the only eigenvalue threshold.

New here: the per-target eigenproblems are solved on each target's own ``n_ref x n_ref``
sub-matrix, targets grouped into size classes (``compact`` = the class step; 0 = the benchmark's
padded M x M stack).  The same eigenpairs to rounding, (n_ref/M)^3 of the work.

With no CuPy the same code runs on NumPy / SciPy (``Backend("numpy")``), for tests.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

EIG_TOL = 1e-6          # klip_basis's absolute eigenvalue threshold: the only one anywhere


# ----------------------------------------------------------------------------- backend
class Backend:
    """numpy/scipy or cupy/cupyx behind one name."""

    def __init__(self, name: str = "cupy"):
        self.name = name
        if name == "cupy":
            import cupy as cp                      # noqa: WPS433
            import cupyx                           # noqa: WPS433
            import cupyx.scipy.ndimage as cndi     # noqa: WPS433
            self.xp, self.ndi, self._cupyx, self.cp = cp, cndi, cupyx, cp
            self.is_gpu = True
        else:
            import scipy.ndimage as sndi           # noqa: WPS433
            self.xp, self.ndi, self._cupyx, self.cp = np, sndi, None, None
            self.is_gpu = False
        self._xsyev = None

    def sync(self) -> None:
        if self.is_gpu:
            self.cp.cuda.Device().synchronize()

    def asnumpy(self, a):
        return self.cp.asnumpy(a) if self.is_gpu else np.asarray(a)

    def to_dev(self, a, dtype=None):
        a = np.asarray(a) if dtype is None else np.asarray(a, dtype)
        return self.xp.asarray(a)

    def scatter_add(self, target_flat, idx, vals) -> None:
        if self.is_gpu:
            self._cupyx.scatter_add(target_flat, idx, vals)
        else:
            np.add.at(target_flat, idx, vals)

    def free_pool(self) -> None:
        if self.is_gpu:
            self.cp.get_default_memory_pool().free_all_blocks()

    def device_name(self) -> str:
        if not self.is_gpu:
            return "numpy"
        props = self.cp.cuda.runtime.getDeviceProperties(self.cp.cuda.Device().id)
        name = props["name"]
        return name.decode() if isinstance(name, bytes) else str(name)

    def eigh(self, A, mode: str = "xbatched64"):
        """Eigendecomposition of a C-contiguous stack (N, n, n): ``(w ascending (N, n), V (N, n, n))``
        with ``V[i][:, j]`` the j-th eigenvector.  ``A`` may be overwritten.
        xbatched64 / xbatched32: cuSOLVER's batched divide-and-conquer (cusolverDnXsyevBatched);
        jacobi64 / jacobi32: cupy.linalg.eigh on the stack (cusolverDn<t>syevjBatched).
        On NumPy every mode is numpy.linalg.eigh."""
        xp = self.xp
        dt = xp.float64 if mode.endswith("64") else xp.float32
        A = xp.ascontiguousarray(A, dtype=dt)
        if not self.is_gpu:
            return np.linalg.eigh(A)
        if mode.startswith("xbatched"):
            if self._xsyev is None:
                self._xsyev = XsyevBatched(self.cp)
            return self._xsyev.eigh(A)
        return xp.linalg.eigh(A)


class XsyevBatched:
    """cusolverDnXsyevBatched through ctypes: one call eigendecomposes a contiguous stack of
    symmetric matrices with a batched divide-and-conquer solver (cuSOLVER >= 11.7.1, CUDA >= 12.6.2).
    The library is the one CuPy has loaded (found by its soname) and the handle is CuPy's."""

    CUDA_R_32F, CUDA_R_64F = 0, 1
    EIG_MODE_VECTOR = 1          # CUSOLVER_EIG_MODE_VECTOR
    FILL_LOWER = 0               # CUBLAS_FILL_MODE_LOWER
    STATUS = {0: "SUCCESS", 1: "NOT_INITIALIZED", 2: "ALLOC_FAILED", 3: "INVALID_VALUE", 4: "ARCH_MISMATCH",
              5: "MAPPING_ERROR", 6: "EXECUTION_FAILED", 7: "INTERNAL_ERROR", 8: "MATRIX_TYPE_NOT_SUPPORTED",
              9: "NOT_SUPPORTED", 31: "INVALID_WORKSPACE"}

    def __init__(self, cp):
        import ctypes
        self.cp, self.ct = cp, ctypes
        try:
            h = cp.cuda.Device().cusolver_handle                 # what cupy.linalg itself uses
        except AttributeError:
            h = cp.cuda.device.get_cusolver_handle()
        self.handle = ctypes.c_void_p(int(h))
        self.lib = self._load()
        ver = ctypes.c_int(0)
        self.lib.cusolverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.cusolverGetVersion(ctypes.byref(ver))
        self.version = int(ver.value)
        if not hasattr(self.lib, "cusolverDnXsyevBatched"):
            raise RuntimeError(f"libcusolver {self.version} has no cusolverDnXsyevBatched (needs >= 11701, CUDA >= 12.6.2)")
        vp, i64, sz, ci = ctypes.c_void_p, ctypes.c_int64, ctypes.c_size_t, ctypes.c_int
        L = self.lib
        L.cusolverDnXsyevBatched_bufferSize.restype = ci
        L.cusolverDnXsyevBatched_bufferSize.argtypes = [vp, vp, ci, ci, i64, ci, vp, i64, ci, vp, ci,
                                                        ctypes.POINTER(sz), ctypes.POINTER(sz), i64]
        L.cusolverDnXsyevBatched.restype = ci
        L.cusolverDnXsyevBatched.argtypes = [vp, vp, ci, ci, i64, ci, vp, i64, ci, vp, ci, vp, sz, vp, sz, vp, i64]
        L.cusolverDnCreateParams.restype = ci
        L.cusolverDnCreateParams.argtypes = [ctypes.POINTER(vp)]
        L.cusolverDnSetStream.restype = ci
        L.cusolverDnSetStream.argtypes = [vp, vp]
        self.params = vp()
        self._check(L.cusolverDnCreateParams(ctypes.byref(self.params)), "cusolverDnCreateParams")

    def _load(self):
        import glob
        import sysconfig
        cands = ["libcusolver.so.11", "libcusolver.so.12", "libcusolver.so"]      # already loaded by CuPy
        for sp in {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}:
            cands += sorted(glob.glob(os.path.join(sp, "nvidia", "cusolver", "lib", "libcusolver.so*")))
        cands += sorted(glob.glob("/usr/local/cuda*/lib64/libcusolver.so*"))
        errs = []
        for c in cands:
            try:
                return self.ct.CDLL(c)
            except OSError as exc:
                errs.append(f"{c}: {exc}")
        raise RuntimeError("libcusolver not found: " + "; ".join(errs))

    def _check(self, status: int, what: str) -> None:
        if status != 0:
            raise RuntimeError(f"{what}: cuSOLVER status {status} ({self.STATUS.get(status, '?')})")

    def eigh(self, A):
        """(w ascending (N, n), V (N, n, n) with V[i][:, j] the j-th eigenvector), as linalg.eigh.
        A must be a C-contiguous (N, n, n) float32/float64 stack; it is overwritten."""
        cp, ct = self.cp, self.ct
        if A.ndim != 3 or A.shape[1] != A.shape[2]:
            raise ValueError(f"need a stack (N, n, n), got {A.shape}")
        A = cp.ascontiguousarray(A)
        N, n = int(A.shape[0]), int(A.shape[1])
        if A.dtype == cp.float32:
            code = self.CUDA_R_32F
        elif A.dtype == cp.float64:
            code = self.CUDA_R_64F
        else:
            raise TypeError(f"float32 or float64 only, got {A.dtype}")
        W = cp.empty((N, n), A.dtype)
        info = cp.zeros(N, cp.int32)
        self._check(self.lib.cusolverDnSetStream(self.handle, ct.c_void_p(int(cp.cuda.get_current_stream().ptr))),
                    "cusolverDnSetStream")
        dev_bytes, host_bytes = ct.c_size_t(0), ct.c_size_t(0)
        self._check(self.lib.cusolverDnXsyevBatched_bufferSize(
            self.handle, self.params, self.EIG_MODE_VECTOR, self.FILL_LOWER, n, code,
            ct.c_void_p(int(A.data.ptr)), n, code, ct.c_void_p(int(W.data.ptr)), code,
            ct.byref(dev_bytes), ct.byref(host_bytes), N), "cusolverDnXsyevBatched_bufferSize")
        work_d = cp.empty(max(int(dev_bytes.value), 16), cp.uint8)
        work_h = np.empty(max(int(host_bytes.value), 16), np.uint8)
        self._check(self.lib.cusolverDnXsyevBatched(
            self.handle, self.params, self.EIG_MODE_VECTOR, self.FILL_LOWER, n, code,
            ct.c_void_p(int(A.data.ptr)), n, code, ct.c_void_p(int(W.data.ptr)), code,
            ct.c_void_p(int(work_d.data.ptr)), int(dev_bytes.value),
            work_h.ctypes.data_as(ct.c_void_p), int(host_bytes.value),
            ct.c_void_p(int(info.data.ptr)), N), "cusolverDnXsyevBatched")
        bad = int((info != 0).sum())                     # synchronises; work_h stays alive until here
        if bad:
            first = int(info[info != 0][0])
            raise RuntimeError(f"cusolverDnXsyevBatched: info != 0 for {bad} of {N} matrices (first: {first})")
        # column-major eigenvectors per matrix: the C view of each matrix holds them as rows
        return W, cp.swapaxes(A, 1, 2)


# ----------------------------------------------------------------------------- the per-frame steps
def inject(be: Backend, cube_dev, angles: np.ndarray, sources, model, pxscale: float, truenorth,
           angle_convention: str = "pa", fallback=None):
    """klip-tpe's inject_sources on the device: per source, the stamp rotated per frame (refpa
    models), shifted by the fractional offset and added at the integer one, every frame at once.
    ``cube_dev`` (n, ny, nx) float32 is not changed; the injected cube is returned (float32).
    ``truenorth``: one angle (deg), or one per frame of ``angles``."""
    xp, ndi = be.xp, be.ndi
    n, ny, nx = cube_dev.shape
    out = cube_dev.astype(xp.float64)            # klip-tpe adds each stamp in float64 and stores float32
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    angles = np.asarray(angles, float)
    if np.ndim(truenorth) > 0:
        truenorth = np.asarray(truenorth, float).ravel()
        if truenorth.shape[0] != angles.shape[0]:
            raise ValueError(f"{truenorth.shape[0]} true-north values for {angles.shape[0]} frames")
    for s in sources:
        st, sc, ok = model.stamp(s.rho)
        m = model
        if not ok:
            if fallback is None:
                raise ValueError(f"no stamp at rho={s.rho} and no fallback model")
            st, sc, ok = fallback.stamp(s.rho)
            m = fallback
        st = np.asarray(st, float)
        sh, sw = st.shape
        thru = float(model.throughput(s.rho))          # the model's throughput, the fallback's stamp and flux unit
        base = be.to_dev(st * (s.contrast * m.flux_unit * thru))
        az = s.theta - truenorth - 270.0 - angles if angle_convention == "pa" else s.theta - angles
        azr = np.deg2rad(az)
        xs = cx + (s.rho / pxscale) * np.cos(azr)
        ys = cy + (s.rho / pxscale) * np.sin(azr)
        refpa = getattr(m, "refpa_deg", None)
        syy, sxx = xp.meshgrid(xp.arange(sh, dtype=xp.float64), xp.arange(sw, dtype=xp.float64), indexing="ij")
        if refpa is not None:
            scx, scy = (sw - 1) / 2.0, (sh - 1) / 2.0
            dx, dy = (sxx - scx)[None], (syy - scy)[None]
            a = xp.asarray(np.deg2rad(az - refpa))
            ca, sa = xp.cos(a)[:, None, None], xp.sin(a)[:, None, None]
            sx = scx + ca * dx + sa * dy
            sy = scy - sa * dx + ca * dy
            stamps = ndi.map_coordinates(base, xp.stack([sy, sx]).reshape(2, -1), order=1, mode="constant", cval=0.0)
            stamps = stamps.reshape(n, sh, sw)
            inside = (sx >= 0) & (sx <= sw - 1) & (sy >= 0) & (sy <= sh - 1)
            stamps = stamps * inside
        else:
            stamps = xp.broadcast_to(base, (n, sh, sw))
        fx, fy = xs - sc[0], ys - sc[1]
        ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
        ddx, ddy = xp.asarray(fx - ix), xp.asarray(fy - iy)
        jj = xp.broadcast_to(xp.arange(n, dtype=xp.float64)[:, None, None], (n, sh, sw))
        coords = xp.stack([jj, syy[None] - ddy[:, None, None], sxx[None] - ddx[:, None, None]])
        shifted = ndi.map_coordinates(xp.ascontiguousarray(stamps), coords.reshape(3, -1), order=1,
                                      mode="constant", cval=0.0).reshape(n, sh, sw)
        ty = xp.asarray(iy)[:, None, None] + syy.astype(xp.int64)[None]
        tx = xp.asarray(ix)[:, None, None] + sxx.astype(xp.int64)[None]
        valid = (ty >= 0) & (ty < ny) & (tx >= 0) & (tx < nx)
        flat = xp.arange(n, dtype=xp.int64)[:, None, None] * (ny * nx) + ty * nx + tx
        be.scatter_add(out.reshape(-1), xp.where(valid, flat, 0).reshape(-1), xp.where(valid, shifted, 0.0).reshape(-1))
        out = out.astype(xp.float32).astype(xp.float64)        # the store to the float32 cube, source by source
    return out.astype(xp.float32)


def highpass(be: Backend, cube_dev, width: int, nan_aware: bool, dtype=np.float64):
    """highpass_stack on the device for a (..., ny, nx) stack: boxcar of odd width, IDL border
    rule, NaN kept.  ``dtype``: the precision of the filter."""
    xp, ndi = be.xp, be.ndi
    w = int(width)
    if w <= 1:
        return cube_dev.astype(xp.float32)
    if w % 2 == 0:
        w += 1
    img = cube_dev.astype(dtype)
    size = (1,) * (img.ndim - 2) + (w, w)
    fin = xp.isfinite(img)
    if nan_aware:
        num = ndi.uniform_filter(xp.where(fin, img, 0.0), size, mode="constant")
        den = ndi.uniform_filter(fin.astype(dtype), size, mode="constant")
        sm = xp.where(den > 0, num / xp.where(den > 0, den, 1.0), xp.nan)
    else:
        sm = ndi.uniform_filter(img, size, mode="constant")
    h = w // 2
    out = img - sm
    out[..., :h, :] = 0.0
    out[..., -h:, :] = 0.0
    out[..., :, :h] = 0.0
    out[..., :, -h:] = 0.0
    out[~fin] = xp.nan
    return out.astype(xp.float32)


def bin_matrix(grp: np.ndarray) -> np.ndarray:
    """(ng, n) averaging matrix of bin_by_groups (mean per group)."""
    ng = int(grp.max()) + 1
    B = np.zeros((ng, grp.size), np.float64)
    B[grp, np.arange(grp.size)] = 1.0
    return B / B.sum(axis=1, keepdims=True)


ROTATE_CHUNK = 192     # frames per gather: keeps the coordinate arrays at ~100 MB


def rotate(be: Backend, imgs_dev, angles_deg: np.ndarray, cval: float, validity: bool = True):
    """rotate_ccw of a stack (K, ny, nx) by angles (K,), a chunk of frames per gather; mode "zero"."""
    xp, ndi = be.xp, be.ndi
    K, ny, nx = imgs_dev.shape
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    yy, xx = xp.meshgrid(xp.arange(ny, dtype=xp.float64), xp.arange(nx, dtype=xp.float64), indexing="ij")
    dx, dy = (xx - cx)[None], (yy - cy)[None]
    ang = np.asarray(angles_deg, float)
    out = xp.empty((K, ny, nx), xp.float64)
    for lo in range(0, K, ROTATE_CHUNK):
        hi = min(lo + ROTATE_CHUNK, K)
        a = xp.asarray(np.deg2rad(ang[lo:hi]))
        ca, sa = xp.cos(a)[:, None, None], xp.sin(a)[:, None, None]
        xs = cx + ca * dx + sa * dy
        ys = cy - sa * dx + ca * dy
        kk = xp.broadcast_to(xp.arange(hi - lo, dtype=xp.float64)[:, None, None], (hi - lo, ny, nx))
        inside = (xs >= 0) & (xs <= nx - 1) & (ys >= 0) & (ys <= ny - 1)
        coords = xp.stack([kk, ys, xs]).reshape(3, -1)
        blk = imgs_dev[lo:hi]
        fin = xp.isfinite(blk)
        src = xp.where(fin, blk, 0.0).astype(xp.float64)
        o = ndi.map_coordinates(src, coords, order=1, mode="constant", cval=0.0).reshape(hi - lo, ny, nx)
        if validity and not bool(fin.all()):
            w = ndi.map_coordinates(fin.astype(xp.float64), coords, order=1, mode="constant", cval=0.0).reshape(hi - lo, ny, nx)
            o[(w <= 1e-6) & inside] = xp.nan
        o[~inside] = cval
        zero = np.flatnonzero(ang[lo:hi] == 0.0)       # rotate_ccw returns a frame rotated by 0 unchanged
        if zero.size:
            zi = xp.asarray(zero)
            o[zi] = blk[zi].astype(xp.float64)
        out[lo:hi] = o
    return out


def nw_ang_comb(be: Backend, dr_dev, angles: np.ndarray):
    """nw_ang_comb on the device for one derotated cube (M, ny, nx) -> (ny, nx) float64."""
    xp = be.xp
    M = dr_dev.shape[0]
    if M == 1:
        return dr_dev[0].astype(xp.float64)
    fin = xp.isfinite(dr_dev)
    A = rotate(be, xp.where(fin, dr_dev, 0.0), -np.asarray(angles, float), cval=0.0)
    A = xp.where(xp.isfinite(A), A, 0.0)
    var = xp.var(A, axis=0, ddof=1)
    pos = var[var > 0]
    vfloor = float(xp.median(pos)) * 1e-6 if int(pos.size) else 1e-30
    var = xp.maximum(var, vfloor)
    Ai = rotate(be, A, np.asarray(angles, float), cval=0.0)
    Vi = rotate(be, xp.broadcast_to(var, A.shape).copy(), np.asarray(angles, float), cval=0.0)
    Vi = xp.maximum(Vi, vfloor)
    Ai = xp.where(xp.isfinite(Ai), Ai, 0.0)
    Vi = xp.where(xp.isfinite(Vi), Vi, vfloor)
    inv = fin.astype(xp.float64) / Vi
    num = (Ai * inv).sum(axis=0)
    den = inv.sum(axis=0)
    out = num / xp.maximum(den, 1e-30)
    out[den <= 0] = xp.nan
    return out


# ----------------------------------------------------------------------------- KLIP
def reference_masks(angles: np.ndarray, k: int, angsep_deg: float, anglemax: float, n_min_ref: int,
                    fast: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Per-target reference masks (M, M) and the dropped-target mask, by klip_annular's rules (the
    auto-fast shortcut, the < 4 fallback, the n_min_ref drop with its 25 % floor).  ``angsep_deg``
    is the exclusion in degrees of rotation (``angsep`` times the arc at the annulus).  The same
    masks as klip_tpe.klip.reference_mask gives target by target, made at once."""
    angles = np.asarray(angles, float)
    n = int(angles.size)
    paspan = float(angles.max() - angles.min()) if n >= 2 else 0.0
    auto_fast = angsep_deg <= 0 and anglemax >= paspan
    info: Dict[str, Any] = {"fast": bool(fast or auto_fast), "auto_fast": bool(auto_fast and not fast),
                            "k_requested": int(k), "k_capped": False, "n_frames": n}
    if fast or auto_fast:
        if n >= 2 and k >= n:
            k, info["k_capped"] = n - 1, True
        masks = np.ones((n, n), bool)                            # every frame, the target included
        return masks, np.zeros(n, bool), dict(info, k=int(k), nref=np.full(n, n - 1), nref_used=np.full(n, n - 1),
                                              n_dropped=0)                 # the keys klip_annular's fast path writes
    dpa = np.abs(angles[:, None] - angles[None, :])
    masks = (dpa >= angsep_deg) & (dpa <= anglemax)
    np.fill_diagonal(masks, False)
    nref = masks.sum(axis=1)
    nstarv = int((nref < n_min_ref).sum())
    do_drop = (n - nstarv) >= max(int(n_min_ref), int(np.ceil(0.25 * n)))
    dropped = do_drop & (nref < n_min_ref)
    few = np.flatnonzero((~dropped) & (nref < 4))                # the < 4 fallback: every other frame
    if few.size:
        masks[few] = True
        masks[few, few] = False
    masks[dropped] = False
    nref_used = masks.sum(axis=1)
    return masks, dropped, dict(info, k=int(k), nref=nref, nref_used=nref_used, n_starved=nstarv,
                                drop_enabled=bool(do_drop), n_dropped=int(dropped.sum()))


class Problems:
    """The distinct eigenproblems of one night's KLIP: one per distinct reference set (the fast
    path has one for every target), each a list of reference frames; and which problem each
    target uses."""

    def __init__(self, masks: np.ndarray, dropped: np.ndarray):
        M = masks.shape[0]
        self.M = M
        self.targets = np.flatnonzero(~dropped)
        rows = masks[self.targets]
        uniq, first, inverse = np.unique(rows, axis=0, return_index=True, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        # problems in the order their first target appears (the order the one-by-one code had)
        order = np.argsort(first, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        prob_of = np.full(M, -1, np.int64)
        prob_of[self.targets] = rank[inverse]
        self.prob_of = prob_of
        self.rows = uniq[order]                                   # (P, M) bool, problem p's references
        self.ids = [np.flatnonzero(r) for r in self.rows]
        self.nref = np.array([len(i) for i in self.ids], np.int64)
        self.P = int(self.rows.shape[0])

    def classes(self, step: int) -> List[Tuple[int, np.ndarray]]:
        """Size classes: n_ref rounded up to a multiple of ``step`` -> [(size, problem indices)]."""
        step = max(int(step), 1)
        sz = ((self.nref + step - 1) // step) * step
        return [(int(s), np.flatnonzero(sz == s)) for s in np.unique(sz)]


def klip_weights(be: Backend, G, prob: Problems, k: int, eigh_mode: str, compact: int, rr: bool = False):
    """The KLIP weights of every target: ``W`` (T, M) float64 with row i the coefficients of the
    reference frames in target ``prob.targets[i]``'s subtraction, so that its residual is
    ``D[t] - W[i] @ D``; ``G`` (M, M) the zone's Gram matrix (float64, on the device).

    For target t with references r and the top-k eigenpairs (lam, V) of ``G[r, r]``:
    ``w_t[r] = V diag(1/lam) V^T G[t, r]`` (modes with lam <= 1e-6 dropped), which is
    klip_annular's zone-Gram projection.  ``compact`` > 0: the problems are solved on their own
    n_ref x n_ref sub-matrices in size classes of that step; 0: one padded M x M matrix per problem
    (the benchmark's way).  ``rr``: a float64 Rayleigh-Ritz step inside the solver's top-k subspace
    (for a float32 solver).  Returns ``(W, n_modes)``."""
    xp = be.xp
    M, T, P = prob.M, len(prob.targets), prob.P
    tx = xp.asarray(prob.targets)
    probx = xp.asarray(prob.prob_of[prob.targets])
    W = xp.zeros((T, M), xp.float64)
    n_modes = 0

    def topk(w, V):
        order = xp.argsort(-w, axis=1)[:, :k]
        lam = xp.take_along_axis(w, order, axis=1).astype(xp.float64)
        Vk = xp.take_along_axis(V, order[:, None, :], axis=2).astype(xp.float64)
        return lam, Vk

    def ritz(A64, lam, Vk):
        # float64 Rayleigh-Ritz in span(Vk): eigenpairs of Vk^T A Vk, vectors Vk Q
        Y = A64 @ Vk
        Bm = xp.swapaxes(Vk, -1, -2) @ Y
        Bm = 0.5 * (Bm + xp.swapaxes(Bm, -1, -2))
        th, Q = be.eigh(xp.ascontiguousarray(Bm), "xbatched64")
        th, Q = th[:, ::-1], Q[:, :, ::-1]
        return th, Vk @ Q

    if compact <= 0:
        masks = xp.asarray(prob.rows.astype(np.float64))
        A = G[None] * (masks[:, :, None] * masks[:, None, :])
        A64 = A if rr else None
        w, V = be.eigh(A, eigh_mode)
        lam, Vk = topk(w, V)
        if rr:
            lam, Vk = ritz(A64, lam, Vk)
        kept = lam > EIG_TOL
        inv = xp.where(kept, 1.0 / xp.where(kept, lam, 1.0), 0.0)
        g = G[tx] * masks[probx]                                        # (T, M): G[t, refs]
        c = xp.einsum("tm,tmk->tk", g, Vk[probx]) * inv[probx]
        W = xp.einsum("tk,tmk->tm", c, Vk[probx])
        return W, int(kept.sum())

    for size, probs in prob.classes(compact):
        n_p = len(probs)
        idx = np.zeros((n_p, size), np.int64)
        valid = np.zeros((n_p, size), bool)
        for j, p in enumerate(probs):
            ii = prob.ids[p]
            idx[j, :len(ii)] = ii
            valid[j, :len(ii)] = True
        idx_x, vx = xp.asarray(idx), xp.asarray(valid)
        A = G[idx_x[:, :, None], idx_x[:, None, :]] * (vx[:, :, None] & vx[:, None, :])
        A64 = A if rr else None
        w, V = be.eigh(A, eigh_mode)
        lam, Vk = topk(w, V)
        if rr:
            lam, Vk = ritz(A64, lam, Vk)
        kept = lam > EIG_TOL
        inv = xp.where(kept, 1.0 / xp.where(kept, lam, 1.0), 0.0)
        n_modes += int(kept.sum())
        # the targets of this class
        local = np.full(P, -1, np.int64)
        local[probs] = np.arange(n_p)
        tsel = np.flatnonzero(local[prob.prob_of[prob.targets]] >= 0)      # rows of W
        if tsel.size == 0:
            continue
        lp = xp.asarray(local[prob.prob_of[prob.targets[tsel]]])            # (Ts,) local problem index
        ii = idx_x[lp]                                                        # (Ts, size)
        g = G[tx[xp.asarray(tsel)][:, None], ii] * vx[lp]                     # (Ts, size): G[t, refs]
        c = xp.einsum("ts,tsk->tk", g, Vk[lp]) * inv[lp]
        Wc = xp.einsum("tk,tsk->ts", c, Vk[lp]) * vx[lp]                      # (Ts, size)
        rows = xp.repeat(xp.asarray(tsel), size)
        vflat = vx[lp].reshape(-1)
        W[rows[vflat], ii.reshape(-1)[vflat]] = Wc.reshape(-1)[vflat]
    return W, n_modes


def padded_weights(be: Backend, G, prob: "Problems", masks_x, lam, Vk):
    """The KLIP weights (T, M) from the top-k eigenpairs of the padded problems: ``lam`` (P, k)
    descending, ``Vk`` (P, M, k) in M-space, ``masks_x`` (P, M) float64 on the device."""
    xp = be.xp
    tx = xp.asarray(prob.targets)
    probx = xp.asarray(prob.prob_of[prob.targets])
    kept = lam > EIG_TOL
    inv = xp.where(kept, 1.0 / xp.where(kept, lam, 1.0), 0.0)
    g = G[tx] * masks_x[probx]                                          # (T, M): G[t, refs]
    c = xp.einsum("tm,tmk->tk", g, Vk[probx]) * inv[probx]
    W = xp.einsum("tk,tmk->tm", c, Vk[probx])
    return W, int(kept.sum())


#: the largest padded stack sent to the solver at once, in bytes (float64): more sectors are
#: batched together while they fit; the rest go in further calls
BATCH_BYTES = 6 * 1024 ** 3


def klip_annular(be: Backend, bc_dev, bang: np.ndarray, *, k: int, inrad: float, outrad: float, n_ang: int,
                 angsep: float, anglemax: float, n_min_ref: int, lam_over_d_px: float,
                 arc_radii: Optional[Tuple[float, float]] = None, zone_center=None, zone_angles: str = "ccw",
                 fast: bool = False, eigh_mode: str = "xbatched64", compact: int = 0, rr: bool = False,
                 batch_sectors: bool = True, zone_cache: Optional[Dict[Any, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
    """Annular KLIP of one binned, high-passed cube (M, ny, nx) on the device, in the Gram-matrix
    formulation, per-target reference sets as klip_annular's slow path makes them.  Pixels outside
    the zones are NaN; dropped targets are NaN frames.  Returns ``(residuals (M, ny, nx) float32,
    info)`` with klip_annular's info keys.

    ``batch_sectors`` (the padded path): the eigenproblems of every sector go to the solver in one
    call (the sectors share the frames and the reference sets, so their padded stacks have one
    shape), in chunks of at most ``BATCH_BYTES``.  cuSOLVER's batched solver is far from busy with
    one sector's hundred matrices of 100^2 at a time; eight sectors in one call cost little more
    than one."""
    from klip_tpe.klip import arcdist_deg, zone_indices
    xp = be.xp
    M, ny, nx = bc_dev.shape
    a_in, a_out = (inrad, outrad) if arc_radii is None else (float(arc_radii[0]), float(arc_radii[1]))
    arc = arcdist_deg(a_in, a_out, lam_over_d_px)
    angsep_deg = float(angsep) * abs(arc)
    masks, dropped, info = reference_masks(np.asarray(bang, float), int(k), angsep_deg, float(anglemax),
                                           int(n_min_ref), fast=fast)
    k_eff = int(min(info["k"], M))
    info.update(arcdist=arc, k_effective=k_eff, rdi=False, fm=False, library_as_adi=False,
                n_basis_frames=M, klip_gram="zone", dropped=dropped)
    prob = Problems(masks, dropped)
    flat = bc_dev.reshape(M, ny * nx)
    out = xp.full((M, ny * nx), xp.nan, xp.float32)
    keep_t = xp.asarray(prob.targets)
    n_modes = []
    sectors = []
    n_ang = int(max(n_ang, 1))
    zkey = (ny, nx, float(inrad), float(outrad), n_ang, None if zone_center is None else tuple(zone_center), zone_angles)
    zones = None if zone_cache is None else zone_cache.get(zkey)
    if zones is None:
        zones = []
        for jj in range(n_ang):
            a0 = jj * 360.0 / n_ang
            a1 = min((jj + 1) * 360.0 / n_ang, 360.0)
            idx = zone_indices((ny, nx), inrad, outrad, a0, a1, zone_center, zone_angles)
            if idx.size:
                zones.append(xp.asarray(idx))
        if zone_cache is not None:
            zone_cache[zkey] = zones
    for idx_d in zones:
        D = flat[:, idx_d].astype(xp.float64)
        D = xp.where(xp.isfinite(D), D, 0.0)
        sectors.append((idx_d, D, D @ D.T))                       # the Gram matrix (M, M) float64

    def put(idx_d, D, res, nm):
        n_modes.append(nm)
        tmp = xp.full((M, idx_d.size), xp.nan, xp.float64)
        tmp[keep_t] = res
        out[:, idx_d] = tmp.astype(xp.float32)

    if k_eff == 0:
        for idx_d, D, G in sectors:
            put(idx_d, D, D[keep_t], 0)
    elif compact > 0 or not batch_sectors:
        for idx_d, D, G in sectors:
            W, nm = klip_weights(be, G, prob, k_eff, eigh_mode, compact, rr)
            put(idx_d, D, D[keep_t] - W @ D, nm)
    else:
        P = prob.P
        masks_x = xp.asarray(prob.rows.astype(np.float64))             # (P, M)
        pair = masks_x[:, :, None] * masks_x[:, None, :]              # (P, M, M)
        per_sector = P * M * M * 8
        step = max(1, int(BATCH_BYTES // max(per_sector, 1)))
        for s0 in range(0, len(sectors), step):
            chunk = sectors[s0:s0 + step]
            A = xp.concatenate([G[None] * pair for _, _, G in chunk], axis=0)     # (S*P, M, M)
            A64 = A if rr else None
            w, V = be.eigh(A, eigh_mode)
            order = xp.argsort(-w, axis=1)[:, :k_eff]
            lam = xp.take_along_axis(w, order, axis=1).astype(xp.float64)
            Vk = xp.take_along_axis(V, order[:, None, :], axis=2).astype(xp.float64)
            if rr:
                Y = A64 @ Vk
                Bm = xp.swapaxes(Vk, -1, -2) @ Y
                Bm = 0.5 * (Bm + xp.swapaxes(Bm, -1, -2))
                th, Q = be.eigh(xp.ascontiguousarray(Bm), "xbatched64")
                lam, Vk = th[:, ::-1], Vk @ Q[:, :, ::-1]
            del A, w, V
            for si, (idx_d, D, G) in enumerate(chunk):
                W, nm = padded_weights(be, G, prob, masks_x, lam[si * P:(si + 1) * P], Vk[si * P:(si + 1) * P])
                put(idx_d, D, D[keep_t] - W @ D, nm)
            del lam, Vk
    info["n_modes"] = n_modes
    return out.reshape(M, ny, nx), info
