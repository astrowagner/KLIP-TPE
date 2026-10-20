"""mwc758_idl2022.py -- the LBTI/ALES KLIP reduction behind Wagner et al. (2023, Nature Astronomy 7,
1208) Fig. 1 (MWC 758, UT 2019-11-14), ported statement by statement from the IDL as it ran in 2022.

Sources (read-only):
  reduce_ales_mwc758_2.pro  github.com/astrowagner/MWC758_ALES @ 9ae925e (2022-08-24), the version the
                            paper's Code availability points to: lines 772-916 (binning, ADI-KLIP, the
                            second high-pass) and 1043-1174 (SDI-KLIP on the ADI-KLIP residual frames)
  adiklip.pro, sdiklip.pro  annular branch ("annmode"); the 2017 copies in idl.macbook/zonasphere (the
                            sphere-tools adiklip.pro was edited 2026-05-26; identical here, no NaN input)
  getzone.pro               2015: distances and angles about (sc/2-1, sc/2-1) = (30, 30).  NOT the
                            (31, 31) of NEAR2-code/code/getzone.pro, which was changed on 2026-07-16.
  get_klip_basis.pro        2015: each reference minus its mean; R R^T eigenvectors in decreasing order,
                            scaled by 1/sqrt(eigenvalue), eigenvalues below 5e-3 zeroed
  nw_ang_comb.pro           sphere-tools, 2019-10-22: inverse-variance combination (Bottom et al. 2017)
  angmask.pro, dist_circle.pro (astrolib), reduce_matrix.pro, psf_gaussian.pro and convolve.pro (astrolib)
IDL behaviour the port relies on, each checked against IDL output:
  ROT(img, a, mag, /INTERP): clockwise by a, about ((n-1)/2, (n-1)/2), bilinear, nearest edge pixel
      outside -- against MWC758_xyln_derot.fits (IDL, 2023), 63 frames x 3 channels, to 4e-5.
  SMOOTH(img, w): pixels within w//2 of the edge copied unchanged; with /NAN the mean of the finite
      values in the full w x w window -- the 2-d form reproduces the paper's SDI cubes, a separable
      form (what GDL does) misses them by 2 %.
Arrays are numpy [..., y, x] (IDL [x, y, ...]); IDL linear pixel order equals numpy's ravel order.
"""
import numpy as np
from scipy import ndimage

NX = 62
ZONE_CEN = NX / 2.0 - 1.0          # getzone.pro (2015): dist_circle / angmask about sc/2.-1
MASK_CEN = 31.0                    # the reduce script's masks: sqrt((xx-31.)^2+(yy-31.)^2)
ANN = (5, 27)                      # annmode_inout (IDL integers)
DIAM = 8.4
F32 = np.float32

# reduce_ales_mwc758_2.pro lines 3-19: the 98 cube wavelengths (single precision, as IDL holds them)
WAVES98 = np.array([
    2.7890406, 2.81217028, 2.83495221, 2.85740164, 2.8795327, 2.90135857, 2.92289153, 2.94414308, 2.96512396,
    2.98584427, 3.00631346, 3.02654047, 3.04653369, 3.06630103, 3.08585, 3.10518767, 3.12432074, 3.14325558,
    3.16199822, 3.1805544, 3.19892958, 3.21712895, 3.23515747, 3.25301985, 3.27072061, 3.28826406, 3.30565431,
    3.32289532, 3.33999086, 3.35694454, 3.37375985, 3.39044013, 3.40698856, 3.42340823, 3.43970209, 3.45587301,
    3.47192371, 3.48785684, 3.50367495, 3.51938049, 3.53497583, 3.55046326, 3.56584499, 3.58112316, 3.59629982,
    3.61137697, 3.62635655, 3.64124043, 3.65603042, 3.67072827, 3.68533568, 3.69985431, 3.71428574, 3.72863154,
    3.74289322, 3.75707222, 3.77116998, 3.78518788, 3.79912726, 3.81298941, 3.82677562, 3.84048712, 3.8541251,
    3.86769075, 3.88118519, 3.89460954, 3.90796487, 3.92125225, 3.93447269, 3.94762721, 3.96071677, 3.97374233,
    3.98670482, 3.99960515, 4.0124442, 4.02522283, 4.0379419, 4.05060222, 4.06320461, 4.07574986, 4.08823872,
    4.10067196, 4.11305031, 4.1253745, 4.13764522, 4.14986317, 4.16202903, 4.17414344, 4.18620707, 4.19822054,
    4.21018448, 4.22209949, 4.23396617, 4.2457851, 4.25755685, 4.26928198, 4.28096105, 4.29259459], dtype=F32)


def wave_index(limit):
    """max(where(waves le limit)): the script's start_wave / end_wave."""
    return int(np.max(np.flatnonzero(WAVES98 <= F32(limit))))


def idl_rot(img, angle, mag=1.0):
    """ROT(img, angle, mag, /INTERP): output pixel p takes the input at c + R(angle)(p - c)/mag,
    c = ((n-1)/2, (n-1)/2), R clockwise on the displayed (y up) image; bilinear (NaN neighbours,
    even at zero weight, propagate), the nearest edge pixel outside the array."""
    ny, nx = img.shape
    xc, yc = (nx - 1) / 2.0, (ny - 1) / 2.0
    th = np.deg2rad(angle)
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    dx, dy = (x - xc) / mag, (y - yc) / mag
    xi = xc + dx * np.cos(th) - dy * np.sin(th)
    yi = yc + dx * np.sin(th) + dy * np.cos(th)
    return ndimage.map_coordinates(np.asarray(img, float), [yi, xi], order=1, mode="nearest")


def _box(a, w):
    return ndimage.uniform_filter(np.asarray(a, float), size=int(w), mode="nearest")


def idl_smooth(img, w, nan=None):
    """SMOOTH(img, w[, /NAN]) with no EDGE keyword: pixels within w//2 of an edge are copied unchanged.
    nan=None  no /NAN: a NaN anywhere in the w x w window gives NaN.
    nan='2d'  /NAN: the mean of the finite values in the w x w window (NaN if none); NaN pixels get it too.
    nan='sep' /NAN as two 1-d passes (x then y), each over finite values -- GDL's form, kept for the record."""
    a = np.asarray(img, float)
    h = int(w) // 2
    fin = np.isfinite(a)
    if nan is None:
        sm = np.where(_box((~fin).astype(float), w) > 1e-9, np.nan, _box(np.where(fin, a, 0.0), w))
    elif nan == "2d":
        num, den = _box(np.where(fin, a, 0.0), w), _box(fin.astype(float), w)
        with np.errstate(invalid="ignore", divide="ignore"):
            sm = np.where(den > 1e-9, num / np.where(den > 1e-9, den, 1.0), np.nan)
    elif nan == "sep":
        sm = a
        for ax in (1, 0):
            f = np.isfinite(sm)
            num = ndimage.uniform_filter1d(np.where(f, sm, 0.0), int(w), axis=ax, mode="nearest")
            den = ndimage.uniform_filter1d(f.astype(float), int(w), axis=ax, mode="nearest")
            with np.errstate(invalid="ignore", divide="ignore"):
                sm = np.where(den > 1e-9, num / np.where(den > 1e-9, den, 1.0), np.nan)
    else:
        raise ValueError(nan)
    out = a.copy()
    out[h:a.shape[0] - h, h:a.shape[1] - h] = sm[h:a.shape[0] - h, h:a.shape[1] - h]
    return out


def zones(n_ang, ann=ANN, cen=None, shape=(NX, NX)):
    """getzone.pro (2015) with dist_circle and angmask, in single precision as IDL computes them:
    r = sqrt((x-c)^2 + (y-c)^2), ang = atan(y-c, x-c) + !Pi, c = 30; zone jj holds
    jj*(360./n_ang)*!DTOR <= ang <= (jj+1)*(360./n_ang)*!DTOR and ann[0] <= r <= ann[1], both ends
    inclusive (a pixel on a boundary is in both zones; the later zone writes it).  Flat indices."""
    ny, nx = shape
    c = F32(ZONE_CEN if cen is None else cen)
    x2 = (np.arange(nx, dtype=F32) - c) ** 2
    y2 = (np.arange(ny, dtype=F32) - c) ** 2
    r = np.sqrt(x2[None, :] + y2[:, None]).astype(F32)
    yy, xx = np.mgrid[0:ny, 0:nx].astype(F32)
    ang = (np.arctan2(yy - c, xx - c).astype(F32) + F32(np.pi)).astype(F32)
    dtor = F32(np.pi / 180.0)
    step = F32(360.0) / F32(n_ang)
    out = []
    for jj in range(n_ang):
        a0 = F32(F32(jj) * step) * dtor
        a1 = F32(F32(jj + 1) * step) * dtor
        out.append(np.flatnonzero(((ang >= a0) & (ang <= a1) & (r >= ann[0]) & (r <= ann[1])).ravel()))
    return out


def klip_project(t, R, k, tol=5e-3):
    """The projection in adiklip/sdiklip with get_klip_basis (2015): the target minus its zone mean; each
    reference minus its own mean; eigenvectors of R R^T by decreasing eigenvalue, eigenvalues below tol
    zeroed (their vectors then left unscaled), the others scaled by 1/sqrt(eigenvalue); the first
    k = min(k, n_ref) modes Z = v^T R; returns t - (t Z^T) Z."""
    t = np.asarray(t, float)
    t = t - t.mean()
    R = np.asarray(R, float)
    R = R - R.mean(axis=1, keepdims=True)
    w, v = np.linalg.eigh(R @ R.T)
    o = np.argsort(w)[::-1]
    w, v = w[o], v[:, o]
    k = min(int(k), R.shape[0])
    w = np.where(np.abs(w) < tol, 0.0, w)
    s = np.sqrt(np.abs(w[:k]))
    Z = (v[:, :k] / np.where(s > 0, s, 1.0)).T @ R
    return t - (t @ Z.T) @ Z


def bin_frames(cube, angles, nbin=15):
    """Script lines 776-828: frames sorted by angle, means of consecutive nbin frames (remainder
    dropped), angles likewise.  cube (nwave, nframe, y, x)."""
    o = np.argsort(angles, kind="stable")
    c, a = np.asarray(cube)[:, o], np.asarray(angles, float)[o]
    nb = len(a) // nbin
    return (c[:, :nb * nbin].reshape(c.shape[0], nb, nbin, *c.shape[2:]).mean(axis=2),
            a[:nb * nbin].reshape(nb, nbin).mean(axis=1))


def adiklip_channel(kcube, angles, wl_um, k, n_ang, angsep=0.01, anglemax=360.0, zl=None):
    """adiklip(kcube, k, target=ii, posang=-angles, wl=wl, diam=8.4, pixelscale=0.035, angsep, anglemax,
    n_ang, annmode_inout=[5,27]) for every target frame.  References: |dPA| >= angsep * arcdist (arcdist =
    360 lambda/D / (2 pi 16 px)) and <= anglemax; fewer than 4: all other frames.  NaN outside the zones."""
    nb = kcube.shape[0]
    flat = kcube.reshape(nb, -1).astype(float)
    posang = -np.asarray(angles, float)
    rmid = ANN[0] + abs(ANN[1] - ANN[0]) / 2.0
    arcdist = 360.0 * ((wl_um * 1.0e-6 / DIAM) * 206265.0 / 0.035) / (2.0 * np.pi * rmid)
    zl = zones(n_ang) if zl is None else zl
    out = np.full((nb, flat.shape[1]), np.nan)
    allf = np.arange(nb)
    for t in range(nb):
        d = np.abs(posang - posang[t])
        refs = np.flatnonzero((d >= angsep * abs(arcdist)) & (d <= anglemax))
        if refs.size < 4:
            refs = allf[allf != t]
        for idx in zl:
            out[t, idx] = klip_project(flat[t, idx], flat[refs][:, idx], k)
    return out.reshape(kcube.shape)


def nw_ang_comb(cube, angles):
    """nw_ang_comb.pro (2019): each frame rot(+angle), NaN -> 0, the per-pixel variance over frames
    (n-1); the variance image rot(-angle) per frame, the frames rot(-angle); sum(f/var) / sum(1/var)."""
    n = cube.shape[0]
    adi = np.stack([idl_rot(f, a) for f, a in zip(cube, angles)])
    adi[~np.isfinite(adi)] = 0.0
    var = adi.var(axis=0, ddof=1)
    adi = np.stack([idl_rot(f, -a) for f, a in zip(adi, angles)])
    varc = np.stack([idl_rot(var, -a) for a in angles])
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1.0 / (n * np.mean(1.0 / varc, axis=0))) * (n * np.mean(adi / varc, axis=0))


def radius31(shape=(NX, NX)):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return np.sqrt((xx - MASK_CEN) ** 2 + (yy - MASK_CEN) ** 2)


def mask_annulus(img, lo, hi, fill=0.0):
    """img with fill where r < lo or r > hi about (31, 31) (NaN -> 0 first when fill is 0)."""
    r = radius31(np.shape(img)[-2:])
    out = np.array(img, float, copy=True)
    if fill == 0.0:
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out[..., (r < lo) | (r > hi)] = fill
    return out


def adi_channel_products(kcube_hp, angles, wl_um, k, n_ang, zl=None):
    """One channel of script lines 853-901: the residual frames and the derotated, nw_ang_comb'd image
    (NaN within 5 or beyond 27 px of (31, 31)), offset 180."""
    res = adiklip_channel(kcube_hp, angles, wl_um, k, n_ang, zl=zl)
    km = nw_ang_comb(np.stack([idl_rot(f, -a - 180.0) for f, a in zip(res, angles)]), angles)
    r = radius31()
    km[(r < ANN[0]) | (r > ANN[1])] = np.nan
    return res, km


def highpass(cube, w=11, nan=None):
    """frame - smooth(frame, w[, /nan]) on every frame of a (nwave, nframe, y, x) cube: script line 850
    (klip_filt, before ADI) and line 905 (second_klip_filt, on the ADI residuals)."""
    return np.stack([[f - idl_smooth(f, w, nan=nan) for f in ch] for ch in np.asarray(cube, float)])


def final_adi_masks(medcube):
    """Script lines 910-914: the per-channel images zeroed within 6 and beyond 22 px of (31, 31)."""
    return mask_annulus(medcube, 6, 22)


def sdi_refs(waves, sdisep=1.5):
    """sdiklip's reference channels per target: |lambda - lambda_t| >= sdisep * FWHM / 16 with
    FWHM = 1.22 lambda/D in px at 0.0345"/px (microns against a unitless number, as the code has it);
    fewer than 4: all other channels."""
    w = np.asarray(waves, F32)
    rmid = ANN[0] + abs(ANN[1] - ANN[0]) / 2.0
    out = []
    for kt in range(len(w)):
        fwhm = F32(1.22) * w[kt] * F32(1e-6) / F32(DIAM) / F32(np.pi / 180.0) * F32(3600.0) / F32(0.0345)
        refs = np.flatnonzero(np.abs(w - w[kt]) >= F32(sdisep) * fwhm / F32(rmid))
        out.append(refs if refs.size >= 4 else np.flatnonzero(np.arange(len(w)) != kt))
    return out


def sdi_target(adicube, angles, waves, kt, k_sdi=2, n_ang=6, refsets=None, zl=None):
    """Script lines 1069-1125 for one target channel kt of the (nwave, nframe, y, x) ADI residual cube:
    per frame every channel ROT-magnified by lambda_t/lambda_c (single precision ratio), NaN -> 0,
    sdiklip (2 modes per zone), then the residual frames rotated by -(angle+180) and combined with
    nw_ang_comb(angles+180).  Unmasked."""
    nw, nb = adicube.shape[:2]
    w = np.asarray(waves, F32)
    zl = zones(n_ang) if zl is None else zl
    refs = (sdi_refs(w) if refsets is None else refsets)[kt]
    frames = np.full((nb, NX * NX), np.nan)
    for i in range(nb):
        sc = np.stack([idl_rot(adicube[kk, i], 0.0, float(F32(w[kt] / w[kk]))) for kk in range(nw)])
        sc[~np.isfinite(sc)] = 0.0
        flat = sc.reshape(nw, -1)
        for idx in zl:
            frames[i, idx] = klip_project(flat[kt, idx], flat[refs][:, idx], k_sdi)
    frames = frames.reshape(nb, NX, NX)
    derot = np.stack([idl_rot(f, -a - 180.0) for f, a in zip(frames, angles)])
    return nw_ang_comb(derot, np.asarray(angles) + 180.0)


def band_select(waves, lo, hi):
    """where(waves gt lo and waves lt hi), single precision."""
    w = np.asarray(waves, F32)
    return (w > F32(lo)) & (w < F32(hi))


def psf_gaussian_peak1(npix, fwhm):
    """psf_Gaussian(NPIX=npix, FWHM=[fwhm, fwhm]) / max: a unit-peak Gaussian centred at (npix-1)/2."""
    sd = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    x = np.arange(npix) - (npix - 1) / 2.0
    g = np.exp(-x ** 2 / (2.0 * sd * sd))
    return np.outer(g, g)


def idl_convolve(img, psf):
    """astrolib CONVOLVE(img, psf), default (zero-padded FFT): a linear convolution with zeros outside,
    the PSF's pixel floor((n-1)/2) at the origin."""
    from scipy.signal import fftconvolve
    ny, nx = img.shape
    py, px = psf.shape
    full = fftconvolve(np.nan_to_num(np.asarray(img, float)), psf, mode="full")
    oy, ox = (py - 1) // 2, (px - 1) // 2
    return full[oy:oy + ny, ox:ox + nx]


def lod_smooth(img, lam_um):
    """The display smoothing of Fig. 1 b, d-f, as the script's conv products: CONVOLVE with the unit-peak
    Gaussian of FWHM lambda/D at 0.0345"/px (NPIX 61)."""
    return idl_convolve(img, psf_gaussian_peak1(61, (lam_um * 1e-6) / DIAM * 206265.0 / 0.0345))
