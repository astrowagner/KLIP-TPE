# C. Post-optimization VERIFICATION stack — implementation spec

Source files (all under `/mnt/user-data/uploads/idl/`): `near2_verify.pro` (792 lines),
`near2_candidates.pro` (435), `near2_param_verify.pro` (394), `near2_bench*.pro`, plus
the S/N-map helpers in `optimize_near_2_tpe.pro` (2353–2551) and the call sites in the
optimizer main loop (7781–7935, 7990–8020, 8371–8396). Line numbers below refer to those
files. The brief (§7) puts this whole layer *out of scope for v1*; this note is so it can
be ported faithfully later, or so v1 can expose the right hooks (per-night cube, per-eval
injected images, run log) for it.

Conventions shared by every routine:

* Star centre `cx=(nx-1)/2, cy=(ny-1)/2` (verify 529, candidates 202, snrmap 2363).
  **Exception**: `near2m_paramverify` and `near2m_pvstitch` use `cx=nx/2.` (param_verify
  147, 381) — a latent 0.5 px inconsistency (brief §8.4); the port should use `(n-1)/2`
  everywhere and note the change.
* Position convention: `phi=(PA+90)°`, `x=cx+(rho/pxscale)·cos(phi)`, `y=cy+(rho/pxscale)·sin(phi)`
  (verify 128–129). Inverse: `PA=(atan2(dy,dx)/DTOR − 90) mod 360` (candidates 259).
* Instrument constants hard-coded: `lambda=11.0e-6, D=8.2, pxscale=0.0456"`,
  `fwhm=1.028·(lambda/D)·206265/pxscale ≈ 6.07 px` (verify 517, candidates 189).
* `radprof(img)` — radial-profile flattening applied to every image before S/N work.
  **Not among the uploaded sources** (external library routine); confirm with Kevin
  whether it subtracts or divides the azimuthal median profile. Every S/N estimate here
  assumes its output.
* Matched-filter kernel `near2v_mfkern(fwhm)` (verify 53–59): unit-sum Gaussian,
  `sigma=fwhm/2.3548`, size `kd=2·ceil(1.5·fwhm)+1`. Identical to the Gaussian fallback of
  `near2m_mfkern` (opt 223–252); the optimizer's version can substitute the measured
  AGPM-N4 radial PSF when `use_n4_global=1` and `rho` is passed — verify/candidates never do.
* `near2v_mfconv(img,fwhm)` (verify 117–120): NaN→0, then `convol(..., /edge_truncate)`.
* `near2m_med(a)` (opt 542–546): NaN-dropping median with `/even` (mean of the two
  middle values).
* "weights" per night: `w_j = sqrt(texp_j)` from `<root>/n<night>/texp.sav` (variable
  `texp`), else 1 (verify 532–538).

---

## C.1 Statistical primitives (`near2_verify.pro`)

### C.1.1 Mawet (2014) small-sample FAP — `near2v_tfap(snr, n)` (66–72)

```
df = max(n-2, 1)
if snr <= 0: return 0.5
x  = df / (df + snr^2)
FAP = 0.5 * I_x(df/2, 1/2)          # regularised incomplete beta = Student-t survival P(T>snr)
```
Python: `scipy.stats.t.sf(snr, df)`. `n` is the number of noise apertures actually used (`nv`),
not the nominal `floor(2πr/fwhm)`.

### C.1.2 Inverse threshold — `near2v_tthresh(alpha, n)` (80–91)

Bisection (61 iterations) on `near2v_tfap(tau,n)=alpha`, `tau∈[0,50]` for `alpha<0.5`,
`[-50,0]` for `alpha≥0.5`; `alpha≤0 → 1e30`. Python: `t.isf(alpha, max(n-2,1))`.

### C.1.3 Contrast limit from injections — `near2v_climit(alpha, beta, ci, si, napi, n)` (99–107)

For each injection k with `si[k]>0, ci[k]>0`:
```
na    = max(round(napi[k]), 3)
tau   = tthresh(alpha,   na)         # detection threshold at per-resel FAP alpha
dlt   = tthresh(1-beta,  na)         # completeness margin at TAP beta
lim_k = ci[k] * (tau + dlt) / si[k]  # linear flux<->contrast scaling (Jensen-Clem 2018)
```
Return the plain mean of `lim_k` over valid k (NaN if none).

`near2v_climg(alpha_img, beta, ...)` (352–360) is the same with a per-IMAGE FAP converted to
per-resel first: `ar = min(alpha_img/na, 0.49)`.

### C.1.4 Matched-filter S/N at a source — `near2v_snrfap(img,cx,cy,rho,pa,pxscale,fwhm,exr,srad, exclude=, mfmap=)` (122–154)

Returns `[snr, fap, nap_used, mf_peak]`.

1. `A = mfmap` if supplied with matching size, else `near2v_mfconv(img,fwhm)` (127).
2. Nominal position `(xs,ys)` from (rho,PA). **Peak search**: over the disc `dx²+dy²≤srad²`
   around `(round xs, round ys)`, pick the pixel with max `A` where `img` is finite (131–136).
   Result `(px,py)`; if none beats −1e30, stays at the rounded nominal.
3. `r = |(px,py)−(cx,cy)|`, `nap = floor(2πr/fwhm)`. If `r<fwhm` or `nap<5` → `[NaN,NaN,nap,pkv]` (137–138).
4. Ring samples: `th0=atan2(py−cy, px−cx)`; for `q=1..nap−1`: `th=th0+q·2π/nap`,
   `(xk,yk)=round(cx+r cos th, cy+r sin th)`. Skip if `dist((xk,yk),(px,py)) < exr`
   (radial exclusion — so q=1 and q=nap−1 typically drop when exr=1.5 fwhm), or within `exr`
   of any `exclude` coordinate, or out of bounds, or `img[xk,yk]` non-finite. Collect `A[xk,yk]` (139–148).
5. `nv<3 → NaN`. `sd=stddev(use)` (sample, N−1); `sd≤0 → NaN`.
   `snr = (A[px,py] − mean(use)) / (sd·sqrt(1+1/nv))` (150–152).
6. `fap = tfap(snr, nv)` (153).

Note the optimizer's `near2_snrpk` / `near2m_snrmap` differ: they (a) `near2m_declip` the
image first (6σ 3×3-median spike removal, opt 267–280), (b) skip q=0..1 and q=nap−1 by
index instead of radial exclusion, and (c) floor `sd` at a radial robust-sigma profile
(2382–2385, 2408). So verify/candidate S/N ≠ optimizer score, per brief §8.5.

### C.1.5 Night combine — `near2v_combine(cube, w, idx)` (160–170)

Weighted mean over the selected slices, NaN-aware: `comb = Σ w_j·img_j·fin_j / Σ w_j·fin_j`;
pixels with zero weight → NaN. Duplicate indices (bootstrap) are counted with multiplicity.

### C.1.6 Night-STIM map — `near2v_stim(cube, w)` (177–196)

```
mn   = Σ_j w_j img_j fin_j / Σ_j w_j fin_j
sdv  = sqrt( Σ_j w_j (img_j − mn)² fin_j / Σ_j w_j fin_j )      # weighted, BIASED (no N−1)
sfloor = median(sdv[finite & >0])
stim = mn / max(sdv, 0.3·sfloor, 1e-12)                          # regularised
```
NaN where total weight ≤0; all-NaN map if `nn<2`. (Pairet 2019 STIM analog at NIGHT level.)

### C.1.7 Bootstrap over nights — `near2v_boot(cube,w,cx,cy,rho,pa,pxscale,fwhm,srad,nboot,seed)` (203–214)

For `b=0..nboot−1`: draw `idx = floor(U(0,1)·nn)` for nn slots (resample nights **with
replacement**); `cb = combine(cube,w,idx)` (no radprof!); `s = snrfap(cb,…, exr=srad, srad)`;
record `vals[b]=s[3]` (the matched-filter **peak value**, not the S/N). Result
`bsig = mean(vals)/stddev(vals)` over finite draws (NaN if <3 finite or std=0).
Note `exr` is passed as `srad` (=drift_px=1.5 px), so the ring exclusion is tiny here; the
port should decide whether to keep that quirk (the peak value is what is used, so exclusion
only affects the NaN guard). Defaults: `nboot=500`, `seed=42`.

### C.1.8 Injection aggregation — `near2v_collect(files, pxscale, fwhm, exr, drift)` (414–440)

For each per-eval injected FITS (`eval*_score_inj.fits.gz`, fallback `eval*_honest_inj.fits.gz`):
4-D `[nx,ny,night,k]` → take central k slice; 3-D → NaN-aware unweighted mean over nights;
`radprof`; header keys `INJNSRC`, `INJRHO<m>`, `INJPA<m>`, `INJCON<m>` (m=1..); measure
`snrfap(img,…,exr,drift)`; keep entries with finite S/N>0. Returns `{n, r, c, snr, nap}`.

---

## C.2 `near2_verify` — multi-night source verification (493–746)

**Inputs**
* `rundir/annulus*/<clean_cube>` or `rundir/<clean_cube>`; default `final_nights.fits.gz`;
  3-D `[nx,ny,nnight]`; header `NIGHT<j>` (1-based) gives night ids (521–539).
* Optional `inj_cube` (e.g. `final_nights_kscan_inj.fits.gz`), 3-D or 4-D (central k
  taken); header `INJNSRC`, `INJRHO<m>/INJPA<m>/INJCON<m>` (621–628).
* `<root>/n<night>/texp.sav` for weights.

**Keywords / defaults** (499–518): `tag=''` (output prefix), `root='/Volumes/RAID36TB/NEAR2/'`,
`rundir` = newest `run_*` (504), `cand_rho[]`, `cand_pa[]` (arcsec, deg E of N),
`drift_px=1.5` (peak-search radius = orbital-drift tolerance), `snr_excl=1.5` (×fwhm ring
exclusion, `exr`), `nboot=500`, `fap_op=2.87e-7` (Gaussian 5σ equivalent),
`seed=42`, `fap_list=[fap_op,1.35e-3,1e-2,0.10]`, `tap_list=[0.50,0.90,0.95]`,
`rbin_as` (default 1 fwhm in arcsec ≈0.277"), `/lite` (skip bootstrap + PDFs),
`ovr_*/fm_*` (plot overlays only).

**Algorithm**
1. Night subsets (542–552): `all`, `first = 0..nn/2−1`, `second = nn/2..`, `even = j%2==0`,
   `odd = j%2==1`. Each subset image = `radprof(combine(cube,w,idx))`. `sub_names =
   ['combined','first-half','second-half','even','odd']`.
2. `stim = near2v_stim(cube,w)` → write `<tag>verify_nightstim.fits` (555–558).
3. Write `<tag>verify_subsets.fits` = `[nx,ny,5]` cube, header `SUBSET1..5` (561–565).
4. Per candidate c (585–618):
   * **Path A**: `sa = snrfap(sub[0], …, exr, drift_px)` → S/N, FAP, nap. Decision
     `DETECT` iff `fap < fap_op` (610).
   * Subset S/N `ssub[s]`, s=0..4 (591). `persist = nanmin(ssub[1:5])` (606).
   * **Path B STIM**: `stim_val = stim[round(sx),round(sy)]`; null set = pixels with
     `|r − rho/pxscale| ≤ fwhm`, finite, and farther than `1.5·fwhm` from the source;
     `fapB = fraction(stim[ann] ≥ stim_val)` (594–602).
   * **Path B bootstrap**: `bsig = near2v_boot(cube, w, …, drift_px, nboot, seed)` unless `/lite` (604).
5. Injection calibration (621–735), if `inj_cube` given: same 5 subsets + STIM on the
   injected cube (weights `w[0:nni−1]`), write `<tag>verify_nightstim_inj.fits`,
   `<tag>verify_subsets_inj.fits`; per injected source m: Path-A S/N/FAP, subset S/N,
   injected-STIM value and its annulus FAP (**no** keep-out around the source here, 660).
   Keep `(snr, nap, contrast, rho)` for sources with finite S/N>0 and contrast>0.
6. Detection-limit tables (674–718): bin the injections by rho (`rbw` bins from `min(ri)`);
   per bin: `sep=mean(ri)`, `n_inj`, `S/N_med=median(si)`, `tau=tthresh(fap_list[0], round(mean(nap))>3)`,
   then **TABLE 1** `c@TAP<pp>` = `climit(fap_list[0], tap, ci, si, napi)` for each tap;
   **TABLE 2** (TAP closest to 0.90) `climit(fap_a, tap90, …)` for each `fap_list` entry plus a
   last column at per-image 10%: `aimg = 0.10/mean(napi)`.

**Outputs** (all in `rundir`, prefixed by `tag`)
| file | content |
|---|---|
| `verify_report.txt` | header (run, cube, nights, fap_op, drift); per candidate: `[A] combined S/N, FAP, nap`; `[A] decision`; `[B] night-STIM, empirical FAP`; `[B] bootstrap significance`; `[B] subset S/N comb/1st/2nd/even/odd`; `[B] persistence`. Then injection block: per source `rho PA c | S/N FAP | STIM STIM-FAP` + subset line; then TABLE 1 / TABLE 2 as above. |
| `verify_nightstim.fits`, `verify_nightstim_inj.fits` | STIM maps |
| `verify_subsets.fits`, `verify_subsets_inj.fits` | `[nx,ny,5]` subset images |
| `verify_subsets.pdf`, `verify_subsets_inj.pdf`, `verify_limits.pdf` | **plotting only** (`near2v_fig` 260–345, `near2v_climfig` 368–407, helpers 228–257) — skip |

### C.2.1 `near2_verify_annulus` (756–792) — per-annulus limit-curve aggregation

Called by the optimizer after each annulus completes (opt 7933) when cubes are saved.
Globs `anndir/eval*_score_inj.fits.gz` (fallback `*_honest_inj*`), runs `near2v_collect`,
then `near2v_curvetxt` (447–488; same TABLE 1/2 content, `# ` comment headers, one line
per rho bin) to `anndir/verify_curve.txt`, appends to run-level accumulators
`va_r/va_c/va_snr/va_nap` and rewrites `rundir/verify_curve.txt`. `verify_limits.pdf`
at both levels is plotting only.

---

## C.3 `near2_candidates` — blind peak search + tiering (150–435)

**Inputs** (185–211): `rundir/<snrfile>` (default `klip_stitched_snr.fits`, Mawet S/N map
from `near2m_snrmap`), `rundir/<imgfile>` (`klip_stitched.fits`; falls back to the S/N map),
`rundir/<cubefile>` (`klip_stitched_nights.fits.gz`, `[nx,ny,nn]`, header `NIGHT<j>`,
optional `INJNSRC/INJRHO<m>/INJPA<m>`), `rundir/paramverify_stim_stitched.fits` (optional),
common block `near2_srcmask` (`msk_rho, msk_pa`) = known real sources.

**Keywords / defaults**: `outsub='cand'`, `label=''` (or `'inj_blind_'`), `pxscale`, `fwhm`,
`iwa_px=1.2·fwhm`, `owa_px=min(nx,ny)/2 − fwhm`, `snrmin=2.0`, `npage=25` (top-N given
full verify pages), `batch=6` (candidates per `near2_verify` call), `cache` (pointer to
struct array for cross-annulus reuse), `r_freeze_px=−1`, `/quick` (table + map only).
Optimizer call sites: final pass opt 8388–8395 (`iwa=max(ann_edges[0],1.2 fwhm)`,
`owa=min(ann_edges[nann], min(nx,ny)/2−2)`); running per-annulus pass opt 8010–8013
(`/quick`, `r_freeze_px=ann_edges[ia]`, on `klip_stitched_running_cand_snr.fits`).

**Step 1 — peak detection `near2c_detect`** (32–51): mask non-finite and `r<iwa`, `r>owa`
to −1e30; iterate up to `nmax=400`: take global max; stop if `< snrmin`; record; suppress a
disc of radius `sep = max(0.9·fwhm, 2)` px (216). Returns `{n,px,py,snr}` (integer pixels).

**Step 2 — per-candidate metrics** (222–307). Precompute: `slices[j] = radprof(cube[:,:,j])`,
`comb = radprof(nanmean over nights)` (unweighted), even/odd subset images `s1,s2`
(unweighted nanmeans, 238–242), matched-filter maps of each via `near2v_mfconv`
(246–247). Constants: `exr=1.5·fwhm`, `srad=1.5` px, `ap=max(0.5·fwhm,1.5)` (235–236),
`nresel = π(owa²−iwa²)/fwhm²` (234).
For each peak (skipping cached ones inside `r_freeze` within `0.6·fwhm`, 262–272):
* `pn[j]` = `snrfap(slices[j], …, exr, srad, mfmap=…)[0]` per night;
  `Nnt = count(pn ≥ 2.0)`, `pnmin/pnmed/pnmax` (275–277).
* **STIM (aperture version)**: `av[j]` = mean of `slices[j]` over the disc `≤ap` px around the
  peak; `stim = mean(av)/stddev(av)` over finite nights (needs ≥2) (279–284).
* `sub1 = snrfap(s1,…)`, `sub2 = snrfap(s2,…)` (286–287).
* `psfc = near2c_psfcorr(comb, px, py, fwhm)` (57–72): stamp half-box `max(round(1.5 fwhm),3)`,
  subtract stamp median, zero-mean Gaussian (`sigma=fwhm/2.3548`) centred on the peak,
  normalised cross-correlation over finite pixels ∈[−1,1].
* Combined `cf = snrfap(comb,…)`; `csnr = cf[0]` if finite (replaces the map value);
  look-elsewhere FAP `fap = min(1 − (1−cf[1])^nresel, 1)` (292–294).
* Flags: `'KNOWN'` if within `0.8·fwhm` of a `msk_*` source; `'INJ'` (or `'KNOWN/INJ'`) if
  within `0.8·fwhm` of a header injection (296–306).

**Step 3 — score & tier** (336–364):
```
f_nt  = Nnt / nn
f_psf = clip(psfc, 0, 1)            (NaN → 0)
f_stm = stim / (|stim| + 1.5)       (NaN or stim ≤ 0 → 0)
f_pv  = pv / (|pv| + 1.5)           (pv = max of 3×3 box of paramverify_stim_stitched at the peak;
                                     1.0 if the file is absent; NaN or pv ≤ 0 → 0)
score = csnr · (0.25+0.75 f_nt) · (0.25+0.75 f_psf) · (0.40+0.60 f_stm) · (0.40+0.60 f_pv)

tier A : csnr ≥ 5.0  and Nnt ≥ ceil(nn/2)  and psfc ≥ 0.5
tier B : csnr ≥ 3.5  and (Nnt ≥ 2 or psfc ≥ 0.4)
tier C : otherwise
```
Rank by descending score.

**Step 4 — outputs** in `rundir/cand/`:
* `<label>candidates.txt` (369–393): header lines (`#`), then fixed-width columns
  `rank x y rho" PA S/N Nnt pn_md STIM sub1 sub2 PSF FAP score tier flag`
  (formats `I4 I5 I5 F7.3 F7.1 F7.2 I4 F7.2 F7.2 F6.2 F6.2 F6.2 E10.2 F8.2 A A`).
* `<label>map.pdf` (396–416) — plotting only.
* `cache_log.txt` (313–319) — reuse diagnostics; `README.txt` (`near2c_readme`, 78–145).
* Unless `/quick`: for batches b of `batch` among the top `min(nc,npage)` ranked peaks,
  call `near2_verify(rundir, tag='cand/<label>g<bb>_', cand_rho, cand_pa,
  clean_cube=cubefile, inj_cube='klip_stitched_nights_inj.fits.gz' if present)` (421–433)
  → `gNN_verify_*` products of §C.2.
* Cache struct per peak `{px,py,rho,pa,csnr,nnt,pnmin,pnmed,pnmax,stim,sub1,sub2,psfc,fap,flag}` (324–329).

---

## C.4 `near2_param_verify` — persistence across top-N configurations

### C.4.1 Config selection (optimizer, opt 7795–7817)
* Pool = TPE-phase evals only: `it ≥ n_init`, finite score `Y>−9000`, `Kbest ≥ 1`;
  fall back to all evals if empty (7796–7797).
* Sort by `Y` descending; greedily keep a config if its mean normalised L1 distance to
  every already-kept config, over the base (non-night) dims,
  `d = (1/ndim_base) Σ_d |x_d − x'_d| / (hi_d − lo_d)`, is `≥ pv_divmin` (default 0.05);
  stop at `n_pv=20` (defaults opt 4494–4495). Need ≥2 kept.
* One **shared fixed injection set** for the whole ensemble: `near2m_randpos(nsrc_a, rlo, rhi, …)`
  (deterministic radial ladder `rlo+(i+0.5)(rhi−rlo)/n`, random common azimuth anchor,
  opt 4141+), contrast `ccal` = the annulus's calibrated S/N=5 contrast.
* Each config is re-reduced clean (`cclean[:,:,i]`) and injected (`cinj[:,:,i]`) with its own
  decoded per-night parameters and night selection; weights `wts[i] = max(Y[ep],0)` (7850).

### C.4.2 `near2m_paramverify(anndir, ia, cclean, cinj, wts, fixr, fixt, ccal, pxscale, fwhm, rlo, rhi)` (135–277)
1. `sncl[i] = near2m_snrmap(radprof(cclean_i), fwhm, −1, −1)`;
   `snij[i] = near2m_snrmap(radprof(cinj_i), fwhm, sxp, syp)` (fixed sources excluded from
   noise rings) (33–38, 153–154). `near2m_snrmap` (opt 2353–2417): declip, matched filter,
   per-pixel ring of `nap=floor(2πr/fwhm)` samples at indices q=2..nap−2, exclusion radius
   `exr=snr_exclfwhm·fwhm` (default 1.5), `sd = max(stddev(ring), sigprof[round r])` where
   `sigprof` = 1.4826·MAD of `A` in a `±0.6 fwhm` radial band (≥8 px), `snr=(A−mean)/(sd·sqrt(1+1/nv))`;
   NaN for `r<fwhm` or `nap<5`.
2. `near2m_pv_meanstd` (44–56): per-pixel NaN-aware mean, **unbiased (N−1)** two-pass std, count over configs.
   **param-STIM** `stim = mn / max(sd, 0.3·median(sd), 1e-6)`, NaN where count<2 (157–163).
3. **detfrac** `= fraction of configs with clean S/N > 3.0` per pixel; **recovery** = same on `snij` (166–172).
4. Injection-calibrated reliability (176–191): `stimj` from `snij` the same way; `truev[s]` =
   max of 3×3 box of `stimj` at each fixed source; null = `stim` (clean) values in `rlo≤r≤rhi`
   more than `1.5·fwhm` from every fixed source.
5. Meta-combine (194–217): `Ki[i]` = median over sources of the 3×3 peak of `snij[i]`;
   `Kmed=median(Ki)`; `we_i = wts_i·max(Ki,0)`; `comb = Σ we_i (Kmed/Ki) cclean_i / Σ we_i`
   (same for `combj`); `combsnr = snrmap(radprof(comb))`.
6. Contrast curve (221–229): `ps = near2_snrpk(radprof(combj), …fixr, fixt…)`;
   `reval = rlo + (rhi−rlo)(i+0.5)/12`, i=0..11; `sig_r = radrms(radprof(comb), reval)`
   (`near2m_pv_radrms`, 62–73: stddev in `[r−1, r+1)` annuli); `Kcal_s = ps_s·sig(r_s)/ccal`;
   `Kc = median(Kcal>0)`; `c5 = 5·sig_r/Kc`.

**Outputs** (`anndir` = `rundir/param_verify/`, `tag=NN`, 232–249): `paramverify_stim_NN.fits`,
`paramverify_detfrac_NN.fits`, `paramverify_recovery_NN.fits`, `paramverify_combine_NN.fits`,
`paramverify_combsnr_NN.fits`, `paramverify_calib_NN.txt` (`n_configs, detfrac_thr,
null_median, null_p90, null_p95, null_p99, true_median, true_min`, then rows
`CC r_px r_as c5`), `README.txt`. `paramverify_NN.pdf`, `paramverify_paramsubsets_NN.pdf`
(`near2m_pvsubsets`, 287–360) and `paramverify_config_NN_MM.png` — **plotting only**.

### C.4.3 `near2m_pvstitch(rundir, ann_dirs, ann_tags, ann_rlo, ann_rhi, …)` (367–394)
For each kind in `['stim','detfrac','recovery','combine','combsnr']`: sum each annulus's map
over its zone `rlo−1 ≤ r ≤ rhi+1` (finite pixels), divide by the per-pixel count
(2-px seams averaged), NaN elsewhere → `rundir/paramverify_<kind>_stitched.fits`.
`paramverify_stim_stitched.fits` is the `f_pv` input of §C.3 (candidates 344).

---

## C.5 Benchmark harness (`near2_bench*.pro`)

### C.5.1 Setup — `near2_bench` (near2_bench.pro 24–115)
Runs `optimize_near_2_tpe` on **one annulus** with identical metric/calibration/validation,
varying only `search_mode`:
* Defaults (47–72): `ann_edges=[0,20]` px (production annulus 1, ~0.3–0.9"),
  `n_iter=500`, `nseed=5`, `n_init=50`, `k_klip_max=100`, `nbridges=6`, `opt_nights=0`
  (nights fixed — grid can't search per-night dims), `cmax_cal=6e-5`,
  `use_near2_throughput=1`, `clean_subtract=1`, `use_contrast=[6e-5]` (forced
  sub-threshold injection, ~0.5× the 5σ limit — brief §8.1), no known-source mask.
* Batch tag `btag='bench_YYYYMMDDHHMMSS'` written to `<root>comb/opt/bench_tag.txt` (76–79).
* Launch order (96–108): grid (`seed=101`), random (`seed=202`), TPE seeds `1000+s`,
  all with `bench_tag=btag, /no_cubes`.

### C.5.2 Baselines inside the optimizer
* **grid** (opt 5757–5781): 3-D coarse grid over `(bin, filter, n_ang)` only; all other dims
  at defaults (`inrad, outrad, angsep, anglemax, corr_thresh, noise_max, coronoise_max`,
  `width` = mid-range, k_klip default, all nights in). `gpts = max(round((n_iter−1)^(1/3)), 2)`
  cells per axis, visited in sequence: `gi=(it−1) mod gpts³`, `i1=gi mod gpts`,
  `i2=(gi/gpts) mod gpts`, `i3=(gi/gpts²) mod gpts`, `x = lo + (hi−lo)·i/(gpts−1)`.
  (For n_iter=500: gpts=8, 512 cells; wraps after 512.)
* **random** (opt 5782–5794): uniform in `[lo,hi]` per dim; k_klip dims drawn from the
  non-uniform `kgrid`; night dims by `night_pinc`. Same generator as TPE warm-up.
* Eval 0 in every mode is the seeded default config (bench_plot 65, conv 87).

### C.5.3 Run bookkeeping written by the optimizer
* `run_<stamp>/runinfo.txt` (opt 4735–4736, rewritten 8315): one line
  `search_mode  n_iter[,…]  n_init[,…]  nann  bench_tag` (format `A-8,2x,A,2x,A,2x,I4,2x,A`).
* `run_<stamp>/optimize_tpe_results.txt` (opt 609): per-eval rows
  `annulus iter <ndim opt_names…> drop1 drop2 k_klip medSNR`; `iter` 1-based; eval 0 not logged.
* `<root>comb/opt/bench_summary.txt` (opt 7893–7896), **appended** one row per annulus per run:
  `mode  ann  nann  n_iter  n_init  seeded_default_score  validated_best  stamp`
  (format `A-8 I3 I3 I5 I4 F9.3 F9.3 A`). `seeded_default = Y[0]`, `validated_best = best_snr`
  (the validated winner's median S/N, opt 7495).
* `klip_stitched.fits` existence = run fully finished; `checkpoint.sav` = resumable.

### C.5.4 Convergence curves — `near2_bench_conv` (conv 22–146)
Select runs whose `runinfo.txt` field 5 equals the batch tag (default from `bench_tag.txt`),
mode ∈ {grid,random,tpe}, and `klip_stitched.fits` present (46–56). From
`optimize_tpe_results.txt` keep annulus-1 rows, `(iter, last column = medSNR)`; sort by iter;
**running best** `rb[j]=max(rb[j−1], sn[j])` (58–72). Pad shorter runs with their last value
to `nev_max` (79–84). `def_lvl = mean(pad[:,0])` (88). Grid/random: latest matching run's
curve; TPE: per-eval mean ± std across seeds (109–121). Prints final running-best per mode
(135–141). Figure `fig_bench_conv.pdf` — plotting only.
Note: the curve is the **search score** (upward-biased; brief §8.6), not validated S/N.

### C.5.5 Final-value comparison — `near2_bench_plot` (plot 22–109)
Reads `bench_summary.txt`, keeps rows whose `run_<stamp>/runinfo.txt` tag matches (or
`tag='all'`), `default = median(seeded_default)`, per-mode `mean ± std` of `validated_best`
over seeds/annuli (65–81); prints numbers; bar chart `fig_bench.pdf` — plotting only.

### C.5.6 Restart — `near2_bench_restart` (restart 163–297)
Same defaults as `near2_bench`; reads the current `bench_tag.txt`. Tallies runs whose
`runinfo.txt` has `n_iter` == target and matching tag: finished (has `klip_stitched.fits`)
counted per mode; unfinished with `checkpoint.sav` recorded as resumable (196–225). Needs
`1−ng` grid, `1−nr` random, `nseed−nt` TPE. First **resumes** partial runs via
`optimize_near_2_tpe, …, resume=<dir>` (241–267), then launches fresh runs with seeds
`101+s`, `202+s`, and TPE `9000+s` (kept clear of the 1000-range) (271–291).
`near2_bench_batch.pro` is the non-interactive wrapper (compile stack → `near2_bench_restart,
nbridges=12` → `exit`, CATCH → exit status 2 for a shell relaunch loop);
`near2_bench_finish.pro` regenerates the two figures.

---

## C.6 Port notes / hooks v1 should expose
* A per-night (per-partition) image cube with weights and the night ids, plus injected
  twins with `INJ*` header metadata — everything in C.2/C.3 depends on it.
* Per-eval injected images (for `near2_verify_annulus`) and the per-eval log with
  `(iter, score)` and run descriptor (mode, budget, batch tag) for the benchmark.
* A Mawet S/N-map function with source exclusion (`near2m_snrmap`) alongside the scalar metric.
* Statistical core is tiny: `scipy.stats.t.sf/isf`, weighted mean/std, bootstrap; the
  one genuinely fragile piece is `radprof` (external, unread).
* Inconsistencies to fix rather than port: `nx/2` centres in param_verify; biased weighted
  std in `near2v_stim` vs unbiased in param-STIM; injected-STIM FAP lacking the source
  keep-out (verify 660 vs 601); bootstrap passing `srad` as `exr`.
