# B. Reduction backend specification — `reduce_near_2.pro` → Python `KLIPReducer` + NEAR adapter

Source of truth: `/mnt/user-data/uploads/idl/reduce_near_2.pro` (2401 lines; all line numbers below refer to it
unless prefixed with a file name). Helpers read: `KLIP.pro`, `adklip.pro`, `getzone.pro`, `nw_ang_comb.pro`,
`nw_ang_comb_ref.pro`, `near2_n4psf.pro`, `near2_agpm_trans.pro`, `near2_texp.pro`, `near2_worker.pro`,
`near2_hpar.pro`, `near2_build_sav.pro`, `near2_refcensus.pro` (stub), plus the caller side in
`optimize_near_2_tpe.pro` (`near2m_reduce` L678, `near2m_avg` L450, `near2m_nightstack` L479,
`near2_ref_*` L4208–4331).

> **MISSING SOURCE — must be obtained before the KLIP core can be ported bit-faithfully.**
> The KLIP engine actually called by `reduce_near_2` is `multiklip(...)` (L1890, 1896, 1905, 1911, 1972, 1978,
> 1984) which in turn calls `get_klip_basis_new` (L238 comment; `optimize_near_2_tpe.pro` L4575, 4956) and
> `getzone`. **Neither `multiklip.pro` nor `get_klip_basis_new.pro` is in the upload.** They live in
> `/Users/kevinwagner/idl/` (also compiled into `<root>/near2_routines.sav`). Everything in §3.9–3.10 about
> the projection internals is reconstructed from (a) the call signatures, (b) the legacy `KLIP.pro`/`adklip.pro`
> (same author lineage, same `getzone` + `get_klip_basis` pattern), and (c) explicit comments in
> `reduce_near_2.pro`, `reduce_nircam_coro.pro` L2174–2178 and `optimize_near_2_tpe.pro` L4573–4576, 4788–4792.
> Items marked **[INFERRED]** need confirmation against the real file.

Conventions used in this document:
* IDL arrays are written `[nx, ny, nframes]` (x fastest). The same array in numpy is `(nframes, ny, nx)`, and
  IDL `img[x,y]` is numpy `img[y, x]`.
* IDL `rot(img, a, /interp)` rotates **clockwise** by `a` degrees about `((nx-1)/2, (ny-1)/2)` with bilinear
  interpolation; pixels rotated in from outside are 0 (author's own note, `nw_ang_comb.pro` L44; the injection
  path passes `missing=0.0` explicitly, L1378, 1402). In numpy/scipy: `scipy.ndimage.rotate(img, -a, reshape=False,
  order=1, cval=0)` rotates CCW by `a`... i.e. IDL `rot(img, a)` == `ndimage.rotate(img, a, ...)` **only if** you
  account for the array-axis transpose; the safest port is an explicit bilinear `map_coordinates` about
  `((n-1)/2, (n-1)/2)` with the rotation matrix written down (see §3.11).
* IDL `smooth(img, w)` is a **boxcar (moving-average) filter** of width `w` (even `w` → `w+1`); border pixels
  within `w/2` of the edge are copied unchanged from the input; without `/nan`, NaNs propagate through the box;
  with `/nan`, NaNs are excluded from the mean.
* IDL `fshift(img, dx, dy)`: integer part via wrap-around `shift`, fractional part via bilinear interpolation
  (see `near2_psfstamp`, L62–72, which reimplements it on a stamp).
* IDL `a > b` = max(a,b); `a < b` = min(a,b).

---

## 1. Inputs read from disk

Root directory: `root` keyword, default `/Volumes/RAID36TB/NEAR2/` (L163). Per night `seq` the folder is
`root + 'n' + str(seq) + '/'` (L289). Nights: `seq_range` default `[1,6]` (L178), or explicit `seq_values` (L181).
`seq == 20` is skipped outright (L299).

### 1.1 Optimizer-critical inputs (the `do_inject` / `do_klip` path; `init=0`, L199)

| File (per night `n<seq>/`) | Format / shape (IDL) | Variables / keywords used | Where read |
|---|---|---|---|
| `AB_cube_skysub_cen_clean.fits` | FITS primary, float `[432, 432, nf]` (`2*hsize` square, `hsize=216`, see §7) — one frame per raw file (chop A−B median), sky-subtracted, centred (KS5-centred, star at pixel 215.5,215.5), floor-cleaned | data only; header not used (NAXIS3 read by optimizer L4580 to auto-size `k_klip_max`) | L1310/1316 (inject), L1633/1641 (klip), L1442 (fm), L1520 (rotate) |
| `AB_parang_clean.sav` | IDL SAVE | `angles` — float `[nf]`, **parallactic angle in degrees** per frame, `(PARANG START+PARANG END)/2` (L477–479), same order/length as the clean cube | L1311/1317, 1634/1642 |
| `AB_frametags.sav` | IDL SAVE | `corrs` `[nf]` (peak cross-correlation of frame vs running-window pupil median, L976–981), `noises` `[nf]` (stddev in a 21×21 empty background patch at (182,92), L984), `coronoise` `[nf]` (stddev in the 0.3″–0.5″ annulus, L988–995). All aligned to the clean cube. **Optional**: if absent → no frame selection (L99). | `near2_framesel` L98–101 |
| `PSF_ACenB.fits` | FITS float `[16, 16]`, median B-star PSF template, **centred at (7.5, 7.5)** (L1283), raw ADU units | data | L1319 (`ref_file = acenb_file`, L369) |
| `texp.sav` | IDL SAVE | `texp` — float scalar, night's total on-source integration, seconds | header provenance only (`near2_provhdr` L39–42); **weights in `near2m_avg`** (optimizer L456–460); absent → TEXP=-1 / weight 1 |
| `AB_cube_inj.fits` | FITS `[432,432,nf]` | injected cube, **written then re-read** by the same call when `cache=0` (L1430 → L1527/1638/1641) | — |
| `AB_cube_refstar_clean.fits` | FITS `[≥nx, ≥ny, nref]` | only if `use_rdi` (L1860) — off in production | — |

Global (not per night):

| File | Used by |
|---|---|
| `<root>/psflib/n4_psf_cube_EEnorm.fits` (fallbacks `/Volumes/RAID36TB/NEAR2/psflib/…`, `/Users/kevinwagner/idl/n4_psf_cube_EEnorm.fits`) — FITS float `[41, 41, 14]`, header keys `PLATESCL` (arcsec/px, default 0.0453) and `LAMBDAD` (mas, default 283) | `near2_n4_load` (`near2_n4psf.pro` L124–180) |
| `<root>/psflib/throughput_merged.csv` (same fallback chain) — CSV, header row, columns `separation_arcsec, throughput`, 0–5″ at 0.01″ | `near2_n4_throughput` (`near2_n4psf.pro` L72–115) |
| `<root>/near2_routines.sav`, `<root>/parflags/…` | only for `nbridges>1` worker dispatch (§6) — do not port |

`root` for the psflib is taken from the `near2_texpcfg` common (`texp_root`) if the optimizer set it, else the
hard-coded RAID path (`near2_n4psf.pro` L74–80; L129–131 uses the `root` argument first).

**Minimal per-night test dataset for the optimizer path:**
```
n<seq>/AB_cube_skysub_cen_clean.fits
n<seq>/AB_parang_clean.sav          (angles)
n<seq>/AB_frametags.sav             (corrs, noises, coronoise)   [optional: no frame selection without it]
n<seq>/PSF_ACenB.fits
n<seq>/texp.sav                     (texp)                        [optional: weight 1 / TEXP=-1 without it]
psflib/n4_psf_cube_EEnorm.fits      (global)
psflib/throughput_merged.csv        (global)  [optional: analytic fallback T(r)=1-exp(-(r/0.742)^1.73)]
```
`info.sav` (`angles`, `exptimes`, L559) is **not** read on the optimizer path — only by `do_precise_center`
(L585) and `do_clean` (L960), i.e. `init=1`.

### 1.2 Legacy / init-only inputs (not needed by the reducer port)

* Raw ESO `*.fits` (multi-extension chop cubes) — `do_cuber` (L432–567): header keys `EXPTIME`,
  `HIERARCH ESO DET SEQ1 DIT`, `HIERARCH ESO DET NDIT`, `HIERARCH ESO TEL CHOP FREQ` (via `near2_hpar`),
  `ESO TEL PARANG START/END` (via `esopar`). Produces `AB_cube.fits`, `info.sav`, `texp.sav`.
* `AB_cube.fits` → `do_precise_center` (KS5 field-star centring, L573–946) → `AB_cube_skysub_cen.fits`,
  `AB_median_pupil.fits`, `ks5_*.fits`, `derotref_for_center.fits`, `cenref.fits`.
* `AB_cube_skysub_cen.fits` → `do_clean` (L953–1121) → `AB_cube_skysub_cen_clean.fits`, `AB_parang_clean.sav`,
  `AB_frametags.sav`, `AB_pupil.fits`, `AB_mean_pupil_hpf.fits`, `coronoises.fits`, `noise.pdf`.
* `do_rotate_initial` (L1132–1184) → `AB_cube_derotnf.fits`, `AB_median_derot_no_filter.fits`;
  `do_make_psfs` (L1187–1290) cuts B's PSF from `AB_cube_derotnf.fits` at `[169-8:169+7, 423-8:423+7]` (L1191),
  rejects asymmetric frames, recentres to (7.5,7.5) and medians → `PSF_ACenB.fits`.
* `do_rotate` (cADI branch, L1518–1620) is hard-off (`do_rotate=0`, L199).
* Everything gated by `hyp`, `klip_fraction`, `remove_crosstalk`, `block_burn`, `block_airy`, `identify`,
  `comb_near_2` (needs `/nocomb` off) — legacy, all off on the optimizer path.

---

## 2. Keywords of `reduce_near_2` (signature L133–141)

| Keyword | Default | Meaning / where used |
|---|---|---|
| `rho` | (none → no injection) | array of injection separations, **arcsec** (L380). Setting it turns on `do_inject` and `use_inj` (L377–378). |
| `theta` | — | injection PA, **degrees E of N** (the header comment at L152 says radians but L379 applies `!DTOR` to `theta-truenorth-270`, and the optimizer passes degrees, opt L4694). |
| `contrast` | — | linear contrast **w.r.t. α Cen A**; internally rescaled by `129/57` because the template is B (L381). |
| `k_klip` | 4 (L220) | KL modes retained (max mode when `klip_scan`). |
| `klip_scan` | 0 (L243) | 1 → compute residuals for 1..k_klip modes in one pass; output gains an nPC dimension (`[nx,ny,k_klip]` image cube). |
| `bin` | 15 (L208) | max frames per temporal bin (mean-combined, `bin_type='mean'` L211); `bin<=1` → no binning (L1693). |
| `inrad` | 5 (L229) | KLIP annulus inner radius, px in the **cropped 150×150 frame**; clamped to `[0, outrad-1]` (L232). |
| `outrad` | `shsize`=75 (L230) | outer radius, px; **capped at 70** (L231). |
| `n_ang` | 1 (L225) | number of azimuthal sectors per annulus. |
| `filter` | 15 (L201) | boxcar high-pass width in px; applied only if `filter > 1` (L1670, 2060, 2078); so `filter=0` **and** `filter=1` both mean "no filter". |
| `fast` | 1 (L235) | 1 = single KL basis per zone from all frames (angsep/anglemax ignored); 0 = per-target basis with angular selection. Optimizer sets it from its own `/fast` flag, **default 0 (per-target) in production** (opt L4536, L4898). |
| `spat_mean` | 0 (L239) | passed to `multiklip` → `get_klip_basis_new`: subtract each frame's own spatial mean (within the zone) before building the basis. |
| `temp_mean` | 0 (L240) | subtract the per-pixel temporal mean of the reference set before building the basis. |
| `do_destripe` | 1 (L202) | row/column destriping (`destripe(img,90°)` then `destripe(img,0°)`, `clip_level=0`, L1672–1677). **Optimizer passes 0** (`def_do_destripe=0`, opt L4838). |
| `angsep` | 1.0 (L221) | minimum parallactic separation between target and reference, in units of **FWHM (λ/D) of arc at the annulus mid-radius** — converted to degrees via `arcdist` (§3.8). Only honoured on the slow path. |
| `anglemax` | 360 (L222) | maximum |Δparang| (degrees) for a frame to be a reference. Slow path only. |
| `corr_thresh` | 0.95 (L346) | frame selection: keep `corrs >= corr_thresh`. |
| `noise_max` | 2.0 (L349, as `noise_thresh`) | keep `noises <= noise_max * mean(noises)` (`noise_clean=1` hard-on, L348). |
| `coronoise_max` | 2.0 (L351) | keep `coronoise <= coronoise_max * mean(coronoise)` (`coronoise_clean=1`, L350). |
| `dthmax` | (formula, §3.3) | override for the max within-bin PA span in degrees (L1702). |
| `seq_range` | `[1,6]` | contiguous night range. |
| `seq_values` | from `seq_range` | explicit night list; takes precedence. |
| `out_suffix` | `''` | appended to `AB_median_klip<suffix>.fits` and `AB_median_fm<suffix>.fits` (L399, 2143, 2190) to isolate concurrent same-night jobs. Optimizer uses `_pn`, `_pncs`, `_csi`, `_csc`, `_ccln`, `_c0`, `_c1`, `_injfm`. |
| `nocomb` | 0 | 1 → skip `comb_near_2` multi-night combine (L2376) and the `old_new.fits` diagnostic (L2394). Always set by the optimizer. |
| `fm_rho`, `fm_theta`, `fm_contrast` | — | KLIP-FM (Pueyo 2016) test sources: same conventions as `rho/theta/contrast` but the model cube is kept separate (`AB_cube_fm.fits`, L1441–1494) and propagated through the projection (`fmcube=`, `fm_out=`). Not part of the score; skip for v1. |
| `use_near2_throughput` | 1 (via `near2_n4switch` common or default, L171–173) | 1 = measured AGPM-N4 PSF library + measured throughput; 0 = legacy on-axis B template × Maire curve (`near2_agpm_trans`). |
| `use_rdi`, `rdi_mode` | 0, `'ardi'` (L272–275) | reference-star basis. Off in production. |
| `n_min_ref` | 10 (L283–284) | slow path: targets with fewer qualifying references are dropped (NaN) — §3.8. |
| `nbridges` | 1 (L182) | >1 → parallel per-night jobs via persistent workers (§6). Forces `cache=0` (L183). |
| `lean` | 0 | skip diagnostic cube writes (`AB_input-cube_klip.fits` L1830, `AB_cube_klip.fits` L2061, `AB_cube_klip_derot.fits` L2146). |
| `cache` | 0 | in-process cache of clean cube/angles/injected cube/frame tags (`near2_cache` common, L188–198). Injected cube keyed by `injkey = join(rho,theta,contrast)` (L1301). |
| `nthreads` | — | `CPU, TPOOL_NTHREADS=` (L184). |
| `nowait`, `jobflags` | — | parallel dispatch plumbing (§6). |
| `block_burn`, `aa,bb,ba,ab`, `block_airy` | 0 | NaN-mask persistence/off-axis-PSF regions in the final image (L2147–2172). Legacy, off. |

Hard-coded switches relevant to behaviour: `comb_type='nwadi'` (L203); `truenorth=36.5` deg (L204);
`bin_type='mean'` (L211); `annmode=1`, `hyp=0` (L216, 227) so the **annular path is always taken**;
`nrings=6`, `wr=33` are passed to `multiklip` but unused in annmode (L223–224).

---

## 3. Processing chain (do_inject → do_klip), in execution order

### 3.0 Overview
```
clean cube [432,432,nf] + angles[nf]
  └─(do_inject, if rho)──► inject PSFs into EVERY frame (no frame selection here) → AB_cube_inj.fits / cache
do_klip:
  load (injected or clean) cube + angles
  near2_framesel ............................ §3.1  (cube, angles culled)
  crop to [150,150] (hsize±shsize) .......... §3.2
  high-pass #1: frame - smooth(frame, filter)  §3.4  (if filter>1; no /nan)
  destripe 90°, 0° (if do_destripe) ......... §3.6
  angle-aware temporal binning .............. §3.3
  drop all-zero frames ...................... §3.3
  [auto-fast check; RDI] .................... §4
  KLIP (multiklip, annular zones) ........... §3.7–3.10  → klip_cube [150,150,nb] or [150,150,nb,k]
  high-pass #2 on klip_cube (/nan) .......... §3.4
  reference image (nosub or cADI) ........... §5
  derotate each frame by -(angle+truenorth)  §3.11
  nw_ang_comb per k ......................... §3.12
  write AB_median_klip<suffix>.fits ......... §5
```

### 3.1 Frame selection — `near2_framesel` (L86–131)
Applied at `do_klip` (L1649), `do_fm` (L1444) and `do_rotate` (L1529); **not** at injection or PSF-template
time (L80–82). Inputs: `corrs, noises, coronoise` from cache or `AB_frametags.sav`.
```
if nf<=0 or tags missing or len(corrs) != nf: return unchanged
keep = corrs >= corr_thresh
if noise_clean     and len(noises)==nf:    keep &= (noises    <= noise_max     * nanmean(noises))    | ~isfinite(noises)
if coronoise_clean and len(coronoise)==nf: keep &= (coronoise <= coronoise_max * nanmean(coronoise)) | ~isfinite(coronoise)
nw = count(keep)
minkeep = max( round(max(bin,1)) * round(max(k_klip,1)), 3 )          # L121-123
if nw < minkeep: return unchanged   (print "selection skipped, full cube used")   # L124-128
if nw == nf:     return unchanged
cube = cube[keep]; angles = angles[keep]
```
Note the min-keep rule uses `k_klip*bin` frames even when `klip_scan` (k_klip = max mode). The optimizer's
census replica (`near2_ref_binangles`, opt L4238–4255) implements the identical rule.

### 3.2 Crop (L1656)
`cube = cube[hsize-shsize : hsize+shsize-1, same, *]` = `[216-75 : 216+74]` = indices 141..290 → **150×150**.
Star at full-frame 215.5 → cropped **74.5 = (150-1)/2** (between-pixel centre; see §7).

### 3.3 Angle-aware temporal binning (L1693–1738)
Only if `bin > 1`. Groups consecutive surviving frames:
```
fwhm_px    = 1.028 * (11.25e-6/8.2) * 206265 / 0.0456          # = 6.38 px   (L1699)
orcap      = max(float(outrad), 1)  (outrad is the already-capped keyword value)   # L1700
bin_dthmax = 0.5 * fwhm_px / orcap * (180/pi)   [deg]           # L1701 ; e.g. outrad=46 → 3.97°
if dthmax given: bin_dthmax = dthmax
g=0; cnt=0
for ii in range(nfb):
    if cnt == 0: amn = amx = angles[ii]
    else:
        tmn=min(amn,angles[ii]); tmx=max(amx,angles[ii])
        if cnt >= bin or (tmx-tmn) > bin_dthmax:  g+=1; cnt=0; amn=amx=angles[ii]
        else: amn,amx = tmn,tmx
    grp[ii]=g; cnt+=1
```
Bin frame = arithmetic mean of members (`total(...)/n`, L1728; `bin_type='mean'`); bin angle = mean of member
angles (L1731). Then (L1741–1761) any frame whose `total()` is exactly 0 is removed together with its angle.

### 3.4 High-pass filter (L1670, L2060, L2078, L2128)
`frame = frame - smooth(frame, filter)` — **boxcar unsharp mask**, width `filter` px (IDL rounds even widths
up by one; border band of `filter/2` px is left untouched by `smooth`, so the high-passed border = 0).
Applied **twice**: (1) on the cropped data cube before binning/KLIP (L1670, NaNs propagate); (2) on the KLIP
residual cube before derotation (L2060, `/nan`). Gate: `filter > 1`. The optimizer searches `filter ∈ [0,25]`
integer, 0 = off (opt L4776 comment).

### 3.5 Spatial / temporal mean subtraction
`spat_mean`, `temp_mean` are forwarded to `multiklip` → `get_klip_basis_new` (L238–240, 1891–1912). Both 0 in
production and **not searched** (opt L4740). **[INFERRED]** legacy `KLIP.pro` L67 always subtracts the target
zone's mean; whether `get_klip_basis_new` subtracts anything when both flags are 0 must be checked in the
missing source. For the port, expose both as booleans, default False.

### 3.6 Destriping (L1672–1677)
`destripe(img, 90., clip_level=0.0, /nodisp)` then `destripe(img, 0., ...)` per frame — external routine (not
in upload; presumably subtracts a robust per-row/column offset along the given angle). Default keyword is 1 but
the optimizer always passes 0 — **skip in v1**, keep a hook.

### 3.7 Injection of synthetic companions (`do_inject`, L1299–1434)
Runs on the **full 432×432 uncropped, un-selected** clean cube, every frame.

*PSF flux scale (L1342–1347):*
```
pxscale = 0.0456
planet_theta[i]    = (theta[i] - truenorth - 270) * pi/180        # L379  (rad, detector azimuth at parang=0)
planet_r[i]        = rho[i]                                          # arcsec
planet_contrast[i] = contrast[i] * (129/57)                          # L381  (B template → A-referenced contrast)
n4_rap_d = 0.283/pxscale = 6.206 px                                  # 1 λ/D in data px
ref  = PSF_ACenB.fits [16,16]; rcx=rcy=7.5
ee_b = sum(ref[ dist((7.5,7.5)) <= 6.206 ])                          # template EE within 1 λ/D (ADU)
n4_refpa = -93.9 deg                                                 # library distortion-axis PA (near2_n4psf.pro L58-60)
```
*Per source i (L1356–1394):*
```
thru_i = near2_n4_throughput(rho_i)   if use_near2_throughput else near2_agpm_trans(rho_i)      # §8
nstamp, n4rap, ok = near2_n4_psf(rho_i)         # [40,40] unit-EE(1λ/D) library slice, star at (19.5,19.5); ok=0 outside 0.150–1.050"
use_lib = use_near2_throughput and ok
if use_lib:  base = nstamp * planet_contrast_i * thru_i * ee_b ;  lstx0 = lsty0 = int(hsize - 20) = 196
else:        base = stamp  * planet_contrast_i * thru_i          # stamp = 20x20 cut of the template embedded at
                                                                  #   big_ref[hsize-8+xx, hsize-8+yy] (L1325) → template centre at 215.5
                                                                  #   stamp origin stx0 = int(hsize-8)-2 = 206 (2-px zero margin, L1332-1334)
for each frame j:
    az     = planet_theta_i - angles[j]*pi/180          # detector azimuth from +x, CCW   (L1371)
    xshift = (rho_i/pxscale) * cos(az) ; yshift = (rho_i/pxscale) * sin(az)
    if use_lib: rstamp = rot(base, n4_refpa - az_deg, 1.0, /interp, missing=0)   # IDL rot is CW ⇒ CCW by (az - refpa); about stamp centre = star
    s, intx, inty = near2_psfstamp(rstamp or base, xshift, yshift)              # floor split + bilinear fractional shift (L62-72)
    place s at full-frame origin (x0,y0) = (lstx0+intx, lsty0+inty)  [lib]  or (stx0+intx, sty0+inty) [template]
    clip to the frame and ADD into cube[..., j]   (L1385-1392)
```
So in both branches the injected source has **encircled energy within 1 λ/D = contrast·(129/57)·T(ρ)·EE_B**,
centred at full-frame `(215.5 + xshift, 215.5 + yshift)`; the library changes only the PSF *shape* (per-frame
rotated so its vortex-distortion axis points radially). The sub-pixel shift is bilinear (identical to `fshift`).

*Sign convention check:* derotation is `rot(frame, -(parang+truenorth))` = CCW by `parang+truenorth`, so a
source at detector azimuth `az = theta-truenorth-270-parang` lands at sky azimuth `theta-270 ≡ theta+90` deg
from +x — exactly where the metric looks: `x = cx + (rho/pxscale) cos((theta+90)°)`, `y = cy + … sin(…)`
(opt L77–79, L6032). **N is +y, E is −x** (N-up/E-left) in the derotated image.

Also written (diagnostic, L1396–1426): `_injection_model.fits` — the star-free injected-PSF cube derotated and
median-combined, with `SRCRHOn/SRCPAn/SRCCONn/SRCTHRn`, `REFPA`, `PXSCALE` header cards. Then the injected cube
is cached (`cache=1`) or written to `AB_cube_inj.fits` (L1428–1430).

### 3.8 Annular zone geometry and reference selection
**Zones** (`getzone.pro` L12–38, called by `multiklip` **[INFERRED]** with `annulus=[inrad,outrad]` sub-ranges and
`n_ang` sectors, exactly as `adklip.pro` L66–73 does): for a `[nx,ny]` frame,
`dist_circle(mask, [nx,ny], nx/2., ny/2.)` and `angmask(mask, nx/2., ny/2.)`; zone pixels are
`ang ∈ [a0, a1]` (rad, inclusive both ends) **and** `r ∈ [annulus[0], annulus[1]]` (inclusive both ends).
Note `getzone` centres on `nx/2. = 75.0` for the 150 px crop (its comment L24–30 refers to a 320-px pipeline
whose star sits at `sc/2`), i.e. **0.5 px off the NEAR star centre of 74.5** — this only shifts the zone
boundary mask, not the pixel values (zone is mapped back in place), but a faithful port should replicate it (or
knowingly fix it and document). Sectors: `[si*360/n_ang, (si+1)*360/n_ang]` degrees **[INFERRED]** from
`adklip.pro` L67; whether `multiklip` pads/overlaps sectors is unknown. **Annulus padding** is done by the
caller: the optimizer passes `inrad = max(ir-2, 0)`, `outrad = min(or+2, 70)` (opt L695–696, 798), then
`reduce_near_2` re-clamps (L231–232).

**Reference selection per target (slow path only)** — window in |Δparang| (degrees):
```
ann_mid  = inrad + |outrad-inrad|/2                                       # L1931 (opt L4328 floors ann_mid at 1)
arcdist  = 360 * ((11.25e-6/8.2)*206265/0.0456) / (2π ann_mid)            # deg of rotation per FWHM of arc at ann_mid   (L1932)
         = 360 * 6.206 / (2π ann_mid)          (NB: no 1.028 factor here)
refs(ii) = { j ≠ ii :  |pa_j - pa_ii| >= angsep*|arcdist|  AND  |pa_j - pa_ii| <= anglemax }   # L1942, 1963
```
(angles here are the **binned** angles.) Pre-census (L1939–1948): `nstarv` = #targets with `< n_min_ref` refs;
`do_refdrop = (n - nstarv) >= max(n_min_ref, ceil(0.25 n))`. If `do_refdrop`, each starved target's KLIP frame is
set to NaN and skipped (L1961–1968) — it then gets zero weight in `nw_ang_comb` (coverage) and is excluded from
the exposure accounting. If not (`config globally infeasible`), the drop is disabled and `multiklip` is left to
its own fallback, which per L1928/1938 is **the all-frames basis** **[INFERRED from comments]**. What `multiklip`
does for 1 ≤ nref < k_klip is unknown (presumably truncates k to the number of available modes: opt L4575 says
`get_klip_basis_new` keeps all eigenmodes above 1e-6, so k is effectively `min(k, rank)`).

### 3.9 KLIP projection — `multiklip` **[INFERRED — obtain the source]**
Call (annular ADI, fast): `multiklip(cube, k_klip, style='adi', /fast, klip_scan=, spat_mean=, temp_mean=,
posang=angles, wl=11.25, diam=8.2, pixelscale=0.0456, angsep=, anglemax=, n_ang=, wr=, annmode_inout=[inrad,outrad]
[, refcube=, fmcube=, fmrefcube=, fm_out=])` → `[nx,ny,nb]` or `[nx,ny,nb,k_klip]` (L1905–1914).
Slow: same with `target=ii` (no `/fast`) → `[nx,ny]` or `[nx,ny,k_klip]` per call (L1984–1990).

Per zone (Soummer et al. 2012, as in `KLIP.pro` L55–106):
```
T  = target zone pixels (1×N, finite only)                     ; mean-subtracted in KLIP.pro L67 (see §3.5 caveat)
R  = reference zone pixels (M×N)                                ; fast: M = nb (all binned frames, target included)
                                                                ; slow: M = |refs(ii)| (§3.8)
Z  = get_klip_basis_new(R, k)   → K×N orthonormal KL modes:  C = R Rᵀ (M×M) → eigen (λ_m, v_m) descending
                                   Z_m = (1/√λ_m) Σ_i v_m[i] R_i  ; modes with λ ≤ 1e-6·(scale) dropped (opt L4575)
residual(k) = T − Σ_{m≤k} (T·Z_m) Z_m                            ; KLIP.pro L102-104:  T - T # (Zᵀ Z)
klip_scan:  residual(k) for k = 1..k_klip from the same basis (cumulative sums) → 4th axis, index k-1
final_arr[zone_indices] = residual ; pixels outside all zones = NaN (KLIP.pro L29)
```
`wl, diam, pixelscale` are only used inside `multiklip` to compute `arcdist` (same formula as L1932; opt L4326
"matching multiklip"). In `/fast` mode angsep/anglemax are **ignored** (opt L4788–4792, L4537).

### 3.10 What happens with the FM / RDI arguments
Not on the scoring path. `fmcube` is projected with the same basis (`fm_out` = model residual), later combined with
`nw_ang_comb_ref` using the science weights (L2127–2143). Skip for v1.

### 3.11 Derotation (L2105–2118)
`frame = rot(frame, -(angles[ii] + truenorth), /interp)`, i.e. rotate **counter-clockwise by
`parang + 36.5°`** about `((nx-1)/2,(ny-1)/2) = (74.5, 74.5)` with bilinear interpolation, 0 outside.
For the scan case each k-slice cube is derotated separately. Explicit port (numpy `img[y,x]`):
```
c = (n-1)/2 ; a = (parang+truenorth) in rad          # CCW on-sky by a
for output (x,y):  dx=x-c, dy=y-c
   xs = c + cos(a)*dx + sin(a)*dy ;  ys = c - sin(a)*dx + cos(a)*dy    # source coords (inverse map), bilinear, cval=0
```
(verify sign once against `rot` on a test image: IDL `rot(img, +a)` moves a feature at +x toward −y, i.e. CW when
+y is up.)

### 3.12 Combination — `nw_ang_comb(in_cube, angles)` (`nw_ang_comb.pro` L15–63; `comb_type='nwadi'`)
Input: **already derotated** cube `[nx,ny,nb]` and the binned angles (degrees).
```
A = copy(in_cube)
for i: A[i] = rot(A[i], +angles[i], /interp)                 # back toward pupil frame (CW by parang; NOT by truenorth — fine, constant offset)
A[~finite(A)] = 0
var = variance(A, dim=3, /double)                            # per-pixel temporal sample variance (÷(n-1)), float64
vfloor = median(var[var>0]) * 1e-6  (or 1e-30 if none)     ; var = max(var, vfloor)
V[i] = var  for all i
for i: A[i] = rot(A[i], -angles[i], /interp) ; V[i] = rot(V[i], -angles[i], /interp)     # derotate both (edges → 0)
V = max(V, vfloor)
cov = finite(in_cube)                                        # coverage from the ORIGINAL input (NaN frames → 0 weight)
inv = cov / V ; den = Σ_i inv_i
out = Σ_i A_i * inv_i / max(den, 1e-30) ;  out[den <= 0] = NaN
```
Consequences to reproduce: (a) the data that is averaged has been **rotated twice more** (bilinear smoothing);
(b) the weights are inverse temporal variance computed in the pupil frame; (c) NaN target frames (dropped
starved targets) contribute zero weight; (d) pixels never covered → NaN. `mean`/`median` branches exist
(L2109–2110, 2119–2120) but are dead code under `comb_type='nwadi'`.

### 3.13 Output normalisation / NaN masking
There is **no photometric normalisation** of the output image; units are input ADU after high-pass and KLIP.
NaNs arise from: pixels outside `[inrad,outrad]` (KLIP fills NaN), `nw_ang_comb` zero-coverage pixels, and the
optional `block_burn`/`block_airy` masks (off). No explicit inner/outer NaN mask is applied beyond the zone.

---

## 4. `fast` vs slow mode

| | `fast=1` | `fast=0` (per-target) |
|---|---|---|
| Basis | one KL basis per zone from **all** binned frames (target included ⇒ some self-subtraction) (`reduce_nircam_coro.pro` L2176–2177) | per target frame, basis from `refs(ii)` (§3.8) |
| `angsep`/`anglemax` | ignored | honoured |
| Starved-target drop (`n_min_ref`) | n/a | yes (L1929–1969) |
| Output | `multiklip` returns the whole cube in one call (L1905–1914); `klip_cube` pre-copy skipped when `lean` (L1824–1825) | filled frame by frame (L1990) |
| Auto switch | — | if `angsep <= 0` and `anglemax >= (max−min of binned angles)` the slow path is degenerate and `fast=1` is forced (L1839–1845). RDI also forces fast (L1883). |
| KLIP-FM | supported | supported |
| Production | `run_setup.txt` "KLIP_mode: per-target" — the optimizer default is `fast=0` (opt L4536; `def_fast=1` at L4838 is only a stored default) | ✔ (the best-params note `klip_best_params_run_20260620_082516.md` says "per-target KLIP mode") |

Everything else (selection, crop, filter, binning, derotation, combine, outputs) is identical.

---

## 5. Outputs written per night, and what the optimizer reads back

Per call, in `root/n<seq>/`:

| File | When | Content |
|---|---|---|
| `AB_median_klip<suffix>.fits` | always (L2190) | **the product**: `[150,150]` (single k) or `[150,150,k_klip]` (scan; slice k-1 = k modes). Header (`near2_provhdr` L33–52 + L2175–2189): `PROG, DATE, METHOD, NIGHT, TEXP, NINJ, INJCON, HISTORY INJ rho/theta/contrast, KKLIP, KLIPSCAN, ANGSEP, ANGLMAX, BIN, INRAD, OUTRAD, NANG, FILTER, FAST, SPATMEAN, TEMPMEAN, DESTRIPE, NREFMIN, NFRDROP`. |
| `AB_median_nosub.fits` | clean pass (`use_inj=0`, L2087–2096) | derotated + median stack of the filtered, selected, binned cube (no PSF subtraction). |
| `AB_median_cadi_inj.fits` | injected pass (L2072–2086) | classical ADI (2nd high-pass, median-PSF subtract in detector frame, derotate, median). |
| `AB_median_fm<suffix>.fits` | if `fm_rho` (L2143) | KLIP-FM response image. |
| `_injection_model.fits` | injected pass (L1423) | star-free injected-PSF model, sky frame. |
| `AB_cube_inj.fits` | injected pass, `cache=0` (L1430) | full injected cube (input for do_klip). |
| `AB_cube_fm.fits` | if `fm_rho` (L1492) | FM model cube. |
| `AB_input-cube_klip.fits`, `AB_cube_klip.fits`, `AB_cube_klip_derot.fits` | `~lean` (L1830, 2061, 2146) | diagnostics (large). |

Optimizer read-back (`near2m_reduce`, opt L714–717): `files = root+'n'+seq+'/AB_median_klip'+sfx+'.fits'` for
each night in `seqv`, then
* `near2m_avg(files)` (opt L450–470): NaN-aware **weighted mean across nights**, weight `w = sqrt(texp)` from
  `n<seq>/texp.sav` (1 if absent): `comb = Σ w·img_finite / Σ w·finite`, NaN where no coverage. Works on 2-D or
  3-D (scan) files alike.
* `near2m_nightstack(files)` (opt L479–496): stacks per-night images into `[nx,ny,nnight]` (or
  `[nx,ny,nnight,k]`), NaN slices for missing nights.
Per-night mode uses suffixes `_pn` (injected) and `_pncs` (clean) (opt L851–854); the non-pernight clean-subtract
path uses `_csi`/`_csc` (opt L732–733).

---

## 6. Parallel dispatch (semantics only — do not port)

`nbridges > 1` (L2221–2372): for every night a command string
`reduce_near_2, seq_range=[s,s], nbridges=1, /nocomb, cache=1, root='…' + parstr` is built (L2237–2275; `parstr`
serialises every reduction keyword incl. `rho/theta/contrast`, `out_suffix`, `lean`, `klip_scan`, `nthreads`,
`use_near2_throughput`, RDI, `n_min_ref`). It is written atomically as `parflags/[<subdir>/]job_<wname>_<id>.txt`
(line 1 = done-flag path, line 2 = command) for a persistent worker `n<seq>w<slot>` (`near2_worker.pro`): the
worker polls, claims the job by renaming to `.run`, `execute()`s the command (so the night's cube stays cached in
that IDL process), writes the done flag containing `ok`/`error`, and refreshes `alive_<wname>.txt` (heartbeat)
and `busy_<wname>.flag` (touched every ~5 s from the KLIP loop, L1953–1958). The dispatcher launches a worker if
its heartbeat is >30 s stale and no busy flag <1800 s old (L2283–2312); one-shot `idl` batch jobs are the fallback
(L2321–2332). `nowait=1` returns immediately with `jobflags` (L2334–2335); otherwise it polls the flags with a
7200 s timeout (L2353–2366). All of this is replaced by `concurrent.futures` in Python; the only semantic to keep is
**one night = one independent job, outputs isolated by folder + `out_suffix`**.

---

## 7. Constants and geometry

| Quantity | Value | Source |
|---|---|---|
| λ | 11.25 µm (`wl=11.25`; also `11.25e-6` in FWHM formulas) | L1590, 1699, 1892, 1932 (optimizer uses 11.0 µm for its own FWHM, opt L4675 — a known inconsistency) |
| D | 8.2 m | same |
| pixel scale | 0.0456 ″/px (`pxscale`, L371); library file 0.0453 (not resampled) | |
| λ/D | 0.283″ = 6.206 px (`n4_rap_d`, L1342); with 1.028 factor 6.38 px (`fwhm_px`, L1699) | |
| `truenorth` | 36.5° (added to parang in every derotation) | L204 |
| raw crop | `xcen=250, ycen=284`, `hsize = 500 − 284 = 216` → full frame **432×432** (`[xcen-216 : xcen+215]`) | L161–162, 509 |
| KLIP crop | `shsize=75` → **150×150**, indices 141..290 of the full frame | L161, 1656 |
| star centre | full frame **(215.5, 215.5)** = `hsize − 0.5`; cropped **(74.5, 74.5)** = `(n−1)/2` (L1367 comment; template centred at 7.5 of 16 placed at `hsize−8`, L1283/1325) | |
| rotation centre (`rot`) | `((n−1)/2, (n−1)/2)` = 74.5 — consistent with star | IDL `rot` default |
| `getzone` centre | `n/2 = 75.0` — 0.5 px off (see §3.8) | `getzone.pro` L28–29 |
| `outrad` cap | 70 px; `inrad ∈ [0, outrad−1]` | L231–232 |
| default production annulus | `ann_edges=[11,44]` px → padded `[9,46]` | opt L4796, 695–696 |
| B→A contrast factor | 129/57 | L381 |
| `n_min_ref` | 10; `ref_frac` 0.95 (optimizer projection) | L284, opt L4543–4544 |

---

## 8. Injection model — `near2_n4psf.pro` and `near2_agpm_trans.pro`

### 8.1 PSF library files (`/Volumes/RAID36TB/NEAR2/psflib/`)
Only two files are read by code: `n4_psf_cube_EEnorm.fits` and `throughput_merged.csv` (+ a `README_injection.md`
referenced in comments, L13, L70). No index CSV is read.

`n4_psf_cube_EEnorm.fits`: `[41, 41, 14]` float; header `PLATESCL` (0.0453 ″/px) and `LAMBDAD` (283 mas). Each
slice = measured off-axis AGPM-N4 PSF from an Altair field-selector scan, EE-normalised (unit EE within r=10 px),
star at pixel (20,20) (odd/on-pixel). Slice k is at separation `(k+2)·0.075″` (L175); slices 12 and 13 share the
final offset and are **averaged** → 13 slices at 0.150…1.050″ (L170–174).

`near2_n4_load` (L124–180), cached in `near2_n4cfg`:
```
rap = (LAMBDAD/1000)/PLATESCL = 6.25 library px
for each slice: sl = fshift(slice, -0.5, -0.5)[0:39, 0:39]        # star (20,20) → (19.5,19.5) of a 40×40 stamp
                sl /= sum(sl[dist((19.5,19.5)) <= rap])            # unit EE within 1 λ/D
n4_sep = (arange(13)+2)*0.075
```
`near2_n4_psf(rho_as)` (L190–216): `ok=0` if `rho` non-finite or outside `[0.150, 1.050]`; else
`j = clip(value_locate(n4_sep, rho), 0, 11)`, `f = (rho−sep_j)/(sep_{j+1}−sep_j)`,
`st = (1−f)·S_j + f·S_{j+1}`, re-normalised to unit EE within `rap` about (19.5,19.5). Returns `st [40,40]`,
`rap`, `ok=1`. **The stamp is used at the data pixel scale without resampling** (0.0453 vs 0.0456, 0.7 %).

`near2_n4_refpa() = −93.9°` (L58–60): PA (deg CCW from +x on the detector) of the scan displacement, i.e. the
direction in which the library PSFs are distorted; each slice is rotated by `rot(base, refpa − az_deg)` (L1378)
so the distortion axis points along the source's instantaneous detector azimuth.

### 8.2 Throughput `near2_n4_throughput(rho_as)` (L72–115)
Reads `throughput_merged.csv` once (skips non-numeric header; columns `separation_arcsec, throughput`, 0–5″ step
0.01″; measured/interpolated inside 0.5″, analytic outside), then `interpol` (linear) in ρ; `ρ > s_max` → 1.0;
clipped to [0,1]. Fallback if the CSV is missing: `T(ρ) = 1 − exp(−(ρ/0.742)^1.73)` clipped to [0,1] (L113–114).
Defined as encircled energy within 1 λ/D relative to full off-vortex transmission.

### 8.3 Legacy curve `near2_agpm_trans(rho_as)` (`near2_agpm_trans.pro` L21–27)
Digitised Maire et al. 2020 Fig. 3 (AGPM-N4 EE within 1 λ/D), linear `interpol`, clipped [0,1]:
```
rr = [0.0, 0.1, 0.2, 0.3,   0.4,   0.5,   0.6,   0.8,   1.0,   1.25, 1.5,   2.0,  3.0,  6.0 ]
tt = [0.00,0.05,0.18,0.337, 0.438, 0.513, 0.599, 0.734, 0.810, 0.86, 0.898, 0.94, 0.97, 1.00]
```
Used **only** when `use_near2_throughput=0`. With the switch on (production), a `ρ` outside the library range
(0.150–1.050″) still takes its throughput from `near2_n4_throughput` (table, ramping to 1) and only the PSF *shape*
falls back to the on-axis B template (L1359–1364).

### 8.4 Python `InjectionModel` interface implied
`psf(rho_arcsec) -> (stamp[40,40] unit-EE, center=(19.5,19.5), ok)`, `throughput(rho_arcsec) -> T`,
`refpa_deg = -93.9`, plus the fallback template (`PSF_ACenB.fits`, EE_B within 6.206 px) and the `129/57` scale.
Injected EE(1λ/D) = `contrast · (129/57) · T(ρ) · EE_B`.

---

## 9. `near2_texp` and `texp.sav`

`texp.sav` holds one float scalar `texp` = the night's total on-source integration in **seconds**.

Written by `do_cuber` (L541–561): per raw file `fexp_h × nframes` where
`fexp_h = DIT·NDIT` if both `HIERARCH ESO DET SEQ1 DIT` and `HIERARCH ESO DET NDIT` > 0, else `1/(2·CHOP FREQ)`
if `HIERARCH ESO TEL CHOP FREQ` > 0, else `EXPTIME` (L470–476); `texp = Σ_files`.

Rebuilt by `near2_texp.pro` (L47–117) for nights initialised before that code existed: same per-frame rule
(L72–77), frame count = Σ over all HDUs of `NAXIS3` (3-D) or 1 (2-D) (L80–87), skipping pipeline products
(`AB*` and names containing `.fits` twice, L64–65); `texp = Σ f_exp·nfr` counting **both chop halves** (L104;
the header comment's `ceil(NAXIS3/2)` is not what the code does). Consumers: `near2_provhdr` (TEXP card),
`near2m_avg` (`sqrt(texp)` night weights), `near2m_texp` (displayed exposure, scaled by the frame-selection keep
fraction, opt L5306–5330).

`near2_hpar(h, key)` (`near2_hpar.pro`): finds a header line starting with `HIERARCH <key>` followed by space or
`=`, returns the float before any `/`, else −1. In Python: `astropy` exposes these as `hdr['HIERARCH ESO DET NDIT']`
/ `hdr['ESO DET NDIT']` directly.

---

## 10. Port checklist / open items

1. **Get `multiklip.pro` and `get_klip_basis_new.pro`** (and `destripe.pro` if destriping is ever needed) — they
   determine: mean-subtraction behaviour when `spat_mean=temp_mean=0`, sector boundaries/overlap, eigenvalue cut,
   handling of `nref < k`, the all-frames fallback, and the exact klip_scan accumulation.
2. Decide whether to keep the `getzone` 0.5-px centre offset (75.0 vs 74.5) for bit-faithfulness.
3. The IDL pipeline's `theta` is degrees despite the docstring; `rho` arcsec; contrast relative to A.
4. The high-pass is applied twice (pre- and post-KLIP); the second uses NaN-aware smoothing.
5. `nw_ang_comb` double-rotates the data — reproduce, don't "optimise away".
6. Frame selection is applied after injection and must not alter the injected frames' order; the min-keep rule is
   `max(k_klip·bin, 3)`.
7. `outrad` capped at 70 and `inrad` clamped inside `reduce_near_2` **after** the optimizer's ±2 px padding.
8. Production runs use `fast=0` (per-target) even though `reduce_near_2`'s own default is `fast=1`; in `fast=1`
   angsep/anglemax have no effect.
