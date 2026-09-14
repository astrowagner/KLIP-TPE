# `optimize_near_2_tpe` — implementation-ready specification of the main procedure

Source: `/mnt/user-data/uploads/idl/optimize_near_2_tpe.pro`, `pro optimize_near_2_tpe` lines 4334–8399.
All line numbers below refer to that file. Helpers re-read for this spec: `near2m_nightsel/tag/injrows` (2141–2212),
`near2m_stitch_inj/wfits_crop` (2052–2129), `near2m_snrmap/noiseprof/kfit` (2353–2551), `near2m_write_setup`,
`near2m_med`, `near2m_readcalcon`, `near2m_load_annhist` (504–675), `near2m_randpos` (4141–4198),
`near2_ref_*` (4208–4331), plus `near2m_reduce/reduce_ics/reduce_pn` (678–856), `near2m_pnweights/pncombine/pnkpick`
(1311–1368), `near2m_kgrid/ksnap` (1476–1495), `near2m_clip0` (288), `near2m_thru` (865), `near2m_avg/nightstack` (450–496).

Conventions used here: "eval slot" = 0-based index `it` into `X[ndim,n_iter]`, `Y[n_iter]`; the log's `iter` column is `it+1`.
"Score" = `Y[it]` = `msnr` (higher is better). Sentinel for failure = `-9999.` (tests use `Y gt -9000.`).

---------------------------------------------------------------------------------------------------

## 0. Top-level control flow (orientation)

```
resume? -> restore checkpoint.sav (cfg + state) FIRST (4361-4382)
defaults for every keyword (4400-4569)
k_klip_max auto, kgrid (4577-4594); rebuild routine .sav for workers (4600-4624)
run dir / stamp (4626-4673); constants (4675-4676); amax_hi from PA files (4681-4690)
injected-source defaults (4693-4708); annulus edges (4718-4731); runinfo.txt (4735-4737)
search space opt_names/lo/hi/int (4741-4833)
fixed defaults def_* (4837-4840); per-annulus record arrays (4843-4856)
open log optimize_tpe_results.txt (4875-4884); run_setup.txt + scripts archive (4889-4982)
cfg snapshot struct (5014-5033); resume validation (5035-5061)
WHILE ia < nann:                                                   (5062)
   extend-mode decision (5068-5088); per-annulus n_init/n_iter (5093-5099)
   annulus geometry + injection band + nsrc_a (5102-5138)
   [resume_skip -> goto RESUME_ANN]  (5160-5177)
   CALIBRATION (trial 0): contrast targeting loop (5222-5410)
   calib verify products (5418-5443)
   recal params (5457-5459)
RECAL_RESTART:                                                     (5460)
   reset X,Y,Kbest,... seed best-record from calibration (5464-5535); contrast preview (5539-5562)
   trial 0 <- calibration (5568-5592)
RESUME_ANN:  restore search state from ckpt (5599-5676); extend splice (5682-5724)
   FOR it = it_start .. n_iter-1:                                  (5726)
       propose xnew (5731-5843); clamp/round/link (5844-5850); decode nights (5853-5858)
       map to scalars (5862-5886); pernight decode + census projection (5891-5939)
       global census projection (5947-5970); fresh positions (5975-5982)
       reduce + score (5995-6208); clean-subtract fallback (6216-6243); Y[it]=msnr (6245)
       re-cal revisit (6255-6276); cube products (6283-6342); tag (6345-6354)
       best-update (6368-6567) [+ contrast preview + KLIP-FM preview]
       display block (6603-7097, plotting)  ; checkpoint (7103-7140) ; LOG LINE (7147-7154)
   telemetry (7160-7169)
   VALIDATION (7174-7712) -> winner replaces best_*; contrast curve; KLIP-FM; final_setup.txt
   post-validation display (7720-7779, plotting); param_verify (7785-7880)
   bench_summary.txt row (7893-7899); ann_* records (7902-7913)
   opt_width edge commit (7919-7923); verify_annulus / running candidate search (7928-8018)
   ia++
STITCH (8031-8274); final contrast curve (8277-8301); copies + runinfo (8309-8318)
final verify / param_verify stitch / candidate search (8351-8397)
```

Pure-plotting blocks you can skip when porting (they do not change state used by the search, except where noted):
- 5349–5379 (calib display), 6603–7097 (per-eval display; NOTE it also writes `klip_stitched_running*.fits` and the
  `resume_bestfm` forward-model pass, and it builds `Xcorner/ycorner/warm/evph` which ARE checkpointed), 7339–7427
  (validation display), 7720–7779, 7882–7886, the `near2m_savepanels/pn*` calls.

---------------------------------------------------------------------------------------------------

## 1. Keywords / parameters (with defaults and meaning)

Class: **S** = search hyper-parameter, **P** = run protocol, **N** = NEAR-specific / infrastructure, **D** = display only.

| keyword | default (line) | class | meaning |
|---|---|---|---|
| `do_noinj` | 0 (4400); forced 1 if `use_noinj_noise` | P | legacy flag; only meaningful with the difference metric |
| `use_noinj_noise` | 0 (4405) | P | 1 = legacy "inj − clean difference" metric (`near2_snr_inj`); 0 = raw Mawet aperture S/N on `radprof(img)` via `near2_snrpk` (production) |
| `savepng` / `savepdf` | 1 / 1 (4409-4410) | D | PNG snapshots / PDF + FITS products. `savepdf` also gates creation of per-annulus dirs `annulusNN/` (5140) |
| `nbridges` | 6 (4411) | N | parallel night jobs (capped at #nights) |
| `cache` / `lean` / `nthreads` | 1 / 1 / undefined (4412-4413) | N | reduce backend options |
| `n_init` | 40 (4414); may be per-annulus vector | S | random warm-up evals (slot count, includes slot 0) |
| `n_iter` | 200 (4466); may be per-annulus vector | S | total eval slots per annulus (slot 0 = calibration) |
| `pernight` | 0 (4415) | S | per-night parameter blocks (production = 1) |
| `warmstart` | `= pernight` (4420) | S | warm-up draws are global-tied (`near2m_pntie` over ALL bases) |
| `link_params` | `['bin','filter']` (4427) | S | bases whose per-night slots are tied on EVERY proposal (after rounding). `['']` disables |
| `pdf_every` | 10 (4435), ≥1 | D | heavy PDF cadence |
| `use_near2_throughput` | 1 (4442) | N | measured AGPM-N4 PSF/throughput model (`near2_n4switch` common) |
| `use_rdi` / `rdi_mode` | 0 / `'ardi'` (4449-4450) | N | reference-star basis switch |
| `scan_mode` | 0 (4454); `search_k = ~scan_mode` | S | 1 = legacy k-scan (k not a searched dim; `Kbest` from argmax over scan); 0 = k_klip searched |
| `gamma` | 0.25 (4472) | S | TPE good/bad quantile |
| `ncand` | 48 (4473) | S | TPE candidates per proposal |
| `sbest_frac` (variable `seed_best_frac`) | `pernight ? 0.5 : 0.0` (4477) | S | fraction of TPE candidates seeded from the running best (`pbest=` of `near2m_propose`) |
| `explore_frac` | 0.15 (4478) | S | ε-exploration fraction (see §5 for the *effective* rate) |
| `cmap` / `cmap_plots` | 'inferno' / =cmap | D | colormaps |
| `param_verify` / `n_pv` / `pv_divmin` | 1 / 20 / 0.05 (4493-4495) | P | parameter-ensemble reliability stage (pernight only; out of scope) |
| `p_local` / `n_elite` | 0.15 / 5 (4496-4497) | S | local-move fraction and elite pool size |
| `night_pinc` | 0.75 (4498) | S | two-slot: P(slot=0 i.e. "no drop") per slot in random draws; legacy: P(include) per night |
| `kfit_order` | 2 (4499) | P | polynomial order of the log-throughput fit `near2m_kfit` (0 = constant median) |
| `seed` | `long(systime(1))` (4500) | S | IDL RNG seed; after first `randomu` call it becomes the RNG state array; checkpointed |
| `seq_range` / `seq_values` | [1,6] / `indgen(...)+seq_range[0]` (4501-4510) | N | candidate night ids; `nnights = n_elements(seq_values)` |
| `opt_nights` | 1 (4512) | S | 1 = two-slot jackknife (`drop1`,`drop2`), 2 = legacy per-night binary flags, 0 = fixed nights |
| `opt_framesel` | 1 (4513) | S | search `corr_thresh/noise_max/coronoise_max` |
| `save_nightmap` | 1 | D | |
| `n_valid` | 8 (4516) | P | fresh-position validation trials per candidate (0 disables validation) |
| `n_top` | 3 (4517) | P | distinct top configs re-scored in validation |
| `cmax_cal` | undefined → `do_cmax=0` (4518) | P | ceiling on auto-calibrated contrast (only applied when passed) |
| `snr_excl` / `snr_srch` | 1.5 (×FWHM) / 1.5 (px) (4519-4520) | P | metric: noise-ring exclusion radius / peak-search radius (published via `near2_snrcfg` common) |
| `use_workers` / `workers_per_night` | 1 / 1 (4521-4522); wpn bumped to 2 when `clean_subtract` (4524) | N | worker pool |
| `search_mode` | 'tpe' (4527) | S | 'tpe' \| 'random' \| 'grid' |
| `bench_tag` | 'none' (4529) | P | benchmark batch label; `bench_tag ne 'none'` disables contrast previews/intro |
| `fast` | 0 (4536) | N/S | /fast = single KLIP basis; then `angsep/anglemax` are NOT searched and no census projection |
| `n_min_ref` / `ref_frac` | 10 / 0.95 (4543-4546); n_min_ref ≥1, ref_frac clipped [0,1] | S | reference-count guardrail |
| `clean_subtract` | 0 (4549) | S/P | search score = S/N_inj − max(S/N_clean,0) (production ON). Validation stays raw |
| `known_rho` / `known_pa` | none (4561-4567) | P | known real sources (arcsec, deg PA E of N); excluded from noise rings and injection placement |
| `k_klip_max` | auto (4568-4589): `((nfmax/5) < 100) > 10` from max NAXIS3 over nights' science cubes; fallback 25 | S | upper bound of k dims and of kgrid |
| `rho` / `theta` / `contrast` | [0.8,0.8,0.8] / [350,170,440] / 3e-5 (4693-4695) | P | explicit injection set (only used in NON-annular mode; annular mode always re-draws) |
| `random_position` | undefined (4700-4708) | P | ≥1 → that many sources; also overrides `nsrc_a` per annulus (5135) |
| `ann_edges` | [11., 44.] px (4720) | P | radial bin edges (px); `nann = n_elements-1`; `annular = n_elements ≥ 2` |
| `opt_width` / `width_range` | 0 / [15,30] px (4718-4719) | S | greedy auto-width annuli: 'width' becomes a searched dim; `rcap=70` px |
| `no_cubes` | 0 → `save_cubes=1` (4390) | D/P | FITS cube products |
| `legacy_stitch` | 0 (4398) | P | 1 = equal-weight stitch and no seam trimming of the contrast curve |
| `night_pinc`, `known_*` … | see above | | |
| `use_contrast` | undefined (5186) | P | per-annulus vector; entry `>0` forces that contrast for annulus `ia` and disables S/N targeting and re-cal |
| `resume` | undefined (4367) | P | run dir or bare run name; restores config + state; takes precedence over all other keywords |
| `extend_ann` | undefined (4355) | P | per-annulus new totals; requires `resume`; not with `opt_width` |
| `nightsel` variants, `save_nightmap` | | D | |

Production configuration (from brief + defaults): `pernight=1, warmstart=1, link_params=['bin','filter'], opt_nights=1, opt_framesel=1, search_k, clean_subtract=1, seed_best_frac=0.5, p_local=0.15, n_elite=5, explore_frac=0.15, gamma=0.25, ncand=48, n_valid=8, n_top=3, n_min_ref=10, ref_frac=0.95, ann_edges=[11,44]` (or a 3-annulus edge list), `n_init=50`/`n_iter=500…1500`.

Constants (4675-4676): `lambda=11.0e-6 m, diam=8.2 m, pxscale=0.0456 "/px, fwhm = 1.028*(lambda/diam)*206265/pxscale = 6.2378 px` (=0.2844").
NOTE: the census helpers use **11.25e-6** instead (4263, 4330): `fwhm_px(census)=6.3796 px`, `lambda/D = 6.2058 px` — keep this discrepancy if bit-matching the projection.

Globals published through IDL `common` blocks (a Python port should make them explicit context): `near2_snrcfg (snr_excl, snr_srch)`, `near2_runcfg (fast)`, `near2_refcfg (n_min_ref)`, `near2_n4switch`, `near2_rdicfg`, `near2_srcmask (known sources)`, `near2_scorecfg/near2_srcraw/near2_statcfg` (display of raw vs corrected S/N), `near2_evalstat (eval_failed)`, `near2_texpcfg (root)`, `near2_parcfg`.

---------------------------------------------------------------------------------------------------

## 2. Search-space construction (4741–4833)

### 2.1 Base block
`bn_names/opt_names` are built in this order (same for global and per-night):

| name | lo | hi | int | included when |
|---|---|---|---|---|
| `bin` | 5 | 30 | 1 | always |
| `n_ang` | 1 | 8 | 1 | always |
| `filter` | 0 | 25 | 1 | always |
| `width` | width_range[0] | width_range[1] | 1 | global + `opt_width` only (after filter, 4774-4779) |
| `inrad` | 0 | 5 | 1 | NON-annular global only (4782-4785) |
| `outrad` | 45 | 55 | 1 | NON-annular global only |
| `angsep` | 0.0 | 3.0 | 0 | `~fast` |
| `anglemax` | 20 | `amax_hi` | 1 | `~fast` |
| `corr_thresh` | 0.0 | 1.0 | 0 | `opt_framesel` |
| `noise_max` | 0.3 | 3.0 | 0 | `opt_framesel` |
| `coronoise_max` | 0.3 | 3.0 | 0 | `opt_framesel` |
| `k_klip` | 1 | `k_klip_max` | 1 | `search_k` (i.e. not scan_mode) |

`amax_hi` (4681-4690): `pa_span = max over nights of (max(angles)-min(angles))` from `n<id>/AB_parang_clean.sav`;
`amax_hi = pa_span > 5 ? ceil(pa_span) : 360`. (Production table in the brief shows 154.)

### 2.2 Per-night replication (4741-4765) — when `pernight and annular and nnights > 1`
`nblk = n_elements(bn_names)` (9 in production). `opt_names[jn*nblk : (jn+1)*nblk-1] = bn_names + '_n' + strtrim(fix(seq_values[jn]),2)`
for `jn = 0..nnights-1` — i.e. **night-major ordering**: `bin_n1, n_ang_n1, …, k_klip_n1, bin_n2, …`. Bounds/int flags replicated.
`ndim_base = nblk*nnights` (54). Otherwise (global) `ndim_base = n_elements(opt_names)` (9 in production-global).

### 2.3 Night-selection dims (4818-4832) appended after `ndim_base` when `opt_nights and nnights > 1`
- `opt_nights=1` (two-slot): `drop1`, `drop2`, lo 0, hi `float(nnights)`, int. `ndim = ndim_base+2` (56).
- `opt_nights=2` (legacy): one dim per night named `'n'+id`, lo 0, hi 1, int.
- `opt_nights=0`: no dims (`ndim = ndim_base`).

**Decoding** (`near2m_nightsel`, 2141-2154): returns `sel[nnights]` bytes, 1=use.
- two-slot: for each slot d: `c = clip(round(x[d]), 0, nnights)`; `if c ≥ 1: sel[c-1] = 0`. Slot value `c` is the **1-based position in `seq_values`**, not the night id (they coincide for seq_values=1..6). Two equal non-zero slots drop one night. 0/1/2 nights dropped.
- legacy: `sel[d-ndim_base] = clip(round(x[d]),0,1)`.
- Both: `if total(sel) < 1: sel = all ones` (never empty).
`seqv_use = seq_values[where(sel eq 1)]` (5853-5858).

### 2.4 Integer handling
`opt_int` marks integer dims; after every proposal: `xnew = clip(xnew, opt_lo, opt_hi)` then `round()` on int dims (5844-5845). k dims additionally use the non-uniform grid `kgrid = near2m_kgrid(k_klip_max)` (1476-1486): `1..min(20,kmax)`, then `25,30,…,min(50,kmax)`, then `60,70,…,kmax` (only entries ≤ kmax). Random draws pick a grid entry uniformly; TPE/local proposals are snapped with `near2m_ksnap` (nearest entry; ties → lowest index via `min`).

### 2.5 Fixed defaults (4837-4839) — "eval 29 of run_20260620_082516"
`def_k_klip=6, def_angsep=1.923, def_anglemax=26, def_bin=29, def_inrad=5, def_outrad=75, def_n_ang=2, def_filter=12, def_spat_mean=0, def_temp_mean=0, def_do_destripe=0, def_corr_thresh=0.907, def_noise_max=1.451, def_coronoise_max=1.471`. `cdef = inj_contrast[0]` (3e-5 unless `contrast=`). In annular mode `def_inrad/def_outrad` are overwritten per annulus (5107-5117).

---------------------------------------------------------------------------------------------------

## 3. Annulus definition and injection geometry

### 3.1 Edges (4718-4731, 5102-5127)
- Fixed mode: `ann_in = ann_edges[ia]`, `ann_out = ann_edges[ia+1]` (px); `def_inrad = ann_in`, `def_outrad = ann_out`. Warning if `ann_out-ann_in < 2*fwhm`.
- Injection radial band (px): `rlo = ann_in + fwhm`, `rhi = ann_out - fwhm`; if `rhi ≤ rlo` → both = `0.5*(ann_in+ann_out)`.
- `opt_width` mode: `def_inrad = ann_in`, `def_outrad = min(ann_in + round(0.5*(w0+w1)), rcap)`, `rlo=rhi=rmid=0.5*(def_inrad+def_outrad)`; in the eval loop each eval's `orad = min(ir + width, rcap)` (5872) and sources are re-centred at `rmid_e = 0.5*(ir+orad)` (5977).
- The reduction itself pads the zone: `near2m_reduce` passes `inrad = max(ir-2,0)`, `outrad = min(orad+2,70)` to `reduce_near_2` (695-696, 798) so adjacent tiles overlap by ±2 px.

### 3.2 Number of sources per trial (5132-5135)
`nsrc_a = (ia == 0) ? 4 : 6`; `if def_outrad*pxscale < 2.0 → 3`; `if def_outrad*pxscale < 1.0 → min(nsrc_a,2)`; `random_position ≥ 1` overrides with that count. (Default `[11,44]`: 44 px = 2.006" → **4** sources.) After calibration `nsrc_a = n_elements(inj_rho)` (5411).

### 3.3 Position draw `near2m_randpos(n, rlo_as, rhi_as, fwhm, pxscale, seed, rho, theta)` (4141-4198)
- `rlo_as = max(rlo_as, 1.5*fwhm*pxscale)`; `rhi_as = max(rhi_as, rlo_as)`; `rspan = rhi-rlo`.
- **Deterministic radii**: `rho[i] = rlo + (i+0.5)*rspan/n` (written as `rlo + ((r0-rlo) + i*rspan/n) mod rspan`, `r0 = rlo + 0.5*rspan/n`); if `rspan == 0` all `rho = rlo`.
- **Random azimuth**: one `randomu(seed)` → `th0 = U*360`; `theta[i] = (th0 + i*360/n) mod 360` (deg, PA E of N).
- Known-source avoidance: redraw `th0` (another `randomu`) up to 30 times while any source is within `exr = snr_excl*fwhm` px of a known source.
- Warns (only) if any pair closer than `2*fwhm` px.
- **Every call consumes ≥1 uniform from the shared RNG stream** — important for replay.

### 3.4 Pixel convention
Star centre `cx = (nx-1)/2., cy = (ny-1)/2.` everywhere. Source pixel: `x = cx + (rho/pxscale)*cos((theta+90)*!DTOR)`, `y = cy + (rho/pxscale)*sin((theta+90)*!DTOR)` (e.g. 6086, 6133) — i.e. PA measured from +y (north up) towards −x (east left).

### 3.5 `rho/theta/contrast` keyword override
Only affects the NON-annular path (and the calibration's non-annular branch `randpos(nsrc_a, 0.40, 1.70, …)`). In annular mode `inj_rho/inj_theta` are always regenerated (5136) and `inj_contrast = replicate(cdef, n)`; only `contrast[0]` survives as `cdef`. With `random_position`, the initial set is drawn once at 4703 in the band 0.40–1.70".

---------------------------------------------------------------------------------------------------

## 4. Calibration (trial 0) — 5179–5412

Goal: choose per-annulus injection contrast `ccal` so the *default config* scores median S/N in [4,6].

```
force_c = (n_elements(use_contrast) ≥ ia+1) ? float(use_contrast[ia]) : 0
if force_c > 0: inj_contrast[*] = force_c
img_nc = -1  (only computed for use_noinj_noise: clean k-scan at default config)
ical = 0, caldone = 0, kbest_c = -1
while ical < 8 and not caldone:
   if kbest_c < 0:                                    # ONCE per annulus
      img_ic = near2m_reduce(inject=1, kklip=k_klip_max, kscan=1, def_inrad, def_outrad,
                             def_bin, def_n_ang, def_filter, 0,0,0, def_angsep, def_anglemax,
                             inj_rho, inj_theta, inj_contrast, ...)      # 3-D cube [nx,ny,k]
      snrc[k] = near2m_med(near2_snrpk(radprof(img_ic[*,*,k]), cx, cy, inj_rho, inj_theta, pxscale, fwhm))
      kbest_c = argmax_k snrc (0-based, NaN-aware)
   if kbest_c ≥ 0:
      draw TWO fresh position sets: (crho0,ctheta0), (crho1,ctheta1)   # 2 randpos calls (band rlo..rhi)
      cinj = replicate(inj_contrast[0], n)
      if clean_subtract: img_iccln = near2m_reduce(0, kbest_c+1, 0, defaults..., crho0, ctheta0, cinj, out_suffix='_ccln')
      for each trial t in {0,1}:  img_icf = near2m_reduce(1, kbest_c+1, 0, defaults..., crho_t, ctheta_t, cinj)
          csnr_src = near2_snrpk(radprof(img_icf), cx, cy, crho_t, ctheta_t, ...)
          csnr_orig = med(csnr_src)
          if clean_subtract: csnr_raw = csnr_src;
                             csnr_src -= near2m_clip0(near2_snrpk(radprof(img_iccln), cx, cy, crho_t, ctheta_t, ...))
          srev[t] = near2m_med(csnr_src)
      msnr_c = near2m_med(srev)                      # median of the two trials (= their mean)
   write calibNNNN_setup.txt (phase 'calib', step ical+1, k = kbest_c+1, positions crd/ctd, contrast, msnr_c)
   if force_c > 0: caldone
   elif not finite(msnr_c): caldone (warn; keep contrast)
   elif 4 ≤ msnr_c ≤ 6: caldone
   else: fac = clip(5/max(msnr_c,0.5), 0.1, 10); inj_contrast *= fac
   if force_c ≤ 0 and do_cmax: inj_contrast = min(inj_contrast, cmax_cal)
   ical++
ccal = inj_contrast[0]
```
Important details:
- The calibration reductions (5243, 5299, 5303, 5329) do **not** pass `corr_thresh/noise_max/coronoise_max` → `near2m_reduce` defaults `corr_thresh=0.95` and leaves the others undefined (reduce_near_2 defaults, historically 2.0). Only the later clean seed reduction (5492) and the verify clean pass (5424) use `def_corr_thresh=0.907` etc. The "default config" of trial 0 is therefore *not* frame-selected identically to the `x0` vector recorded in `X[*,0]`.
- The k-scan (`kbest_c`) is done once at the first contrast; the contrast loop re-measures only at fixed `kbest_c+1`.
- Metric in the scan and re-measure: the same `near2_snrpk(radprof(img))` used in the search (raw), minus clip0(clean) if `clean_subtract`.
- `nrev = 2`; with `nbridges > 1` both re-measures are dispatched concurrently with suffixes `_c0`/`_c1` and read back with `near2m_avg` (sqrt(texp)-weighted night average, 450-470).
- Up to **8** trials; failure to converge only prints a warning.

### 4.1 Seeded baseline (5464-5535, 5568-5592)
After `RECAL_RESTART:` (history reset) the running best is seeded from the calibration:
- `best_snr = double(near2m_med(csnr_src))` — median over the **last** re-measure's per-source vector (trial `_c1`), NOT `msnr_c`. Fallback `msnr_c`, else `-1d99`.
- `best_img = img_icf` (last re-measure), `best_noinj = sc_f` (only defined in the difference metric) else a fresh clean reduction at `k = kbest_c+1` with `def_corr_thresh/def_noise_max/def_coronoise_max` (5492-5496).
- `besti = -1` ("best is the calibration seed"), `best_k = kbest_c+1`, `best_injr/injt = crd/ctd` (last re-measure positions), `best_injc = ccal`.
- `xcal` / `x0` (5501-5520, 5570-5589): every dim gets its base default by stripping `_n<id>`; `'k_klip' → kbest_c+1` (or def_k_klip if scan failed); `'width' → mid`; night dims → `0` (two-slot) or `1` (legacy).
- **Trial 0 = calibration**: if `finite(msnr_c) and msnr_c > -9000 and recal_changed == 0`: `X[*,0] = x0`, `Y[0] = best_snr`, `Kbest[0] = kbest_c+1`, `expl_hist[0]=0`, and `it_start = 1`. **Slot 0 is never written to the log** in this case (log lines are written inside the loop). Otherwise `it_start = 0` and slot 0 is re-run as an explicit eval with the default vector (5733-5756), and *is* logged as iter 1.
- `stat_lo_g = (nsrc_a == 2) ? 'mean' : 'median'` — display word only; the aggregate is always `near2m_med` = `median(finite, /even)` (mean of the two middle values for even N; for N=2 this equals the mean).

### 4.2 Re-calibration revisit (5457-5459, 6255-6276)
`recal_check=10, recal_ntop=5, recal_budget=4` (0 if forced contrast; also 0 on `resume_skip`/extend).
At `it == 10` (after scoring slot 10): `ywu = Y[1:10]` finite & > −9000; need ≥3; `mwu = median(top-5 of sorted ywu)`; if `4 ≤ mwu ≤ 6` → `recal_budget=0`; elif `mwu < 4 and do_cmax and ccal ≥ 0.999*cmax_cal` → stop checking; else `fac = clip(5/max(mwu,0.1), 0.1, 10)`, `inj_contrast *= fac` (capped by cmax_cal), `ccal = cdef = inj_contrast[0]`, `recal_changed=1`, `budget--`, `n_recal++`, **`goto RECAL_RESTART`**: X/Y/Kbest/etc. are reset, the seed best-record is rebuilt from the (stale, old-contrast) calibration vectors, and since `recal_changed=1` slot 0 is now a real eval of the default config (`it_start=0`). The log already contains the aborted evals (same annulus, iter 1..11) → **duplicate (annulus, iter) rows are possible; a parser must take the last occurrence** (`near2m_load_annhist` does exactly that by overwriting in file order, 666-673).

---------------------------------------------------------------------------------------------------

## 5. Evaluation loop (5726–7156), per slot `it`

### 5.1 Proposal — exact order of branches (5731-5843)
```
1. extend_reseed and it == it_start        -> xnew = x_bestprior (prior best from spliced history)
2. it == 0                                  -> default vector (as x0; k_klip = kbest_c+1)
3. search_mode == 'grid'                    -> grid cell (see 5.7)
4. search_mode == 'random' or it < n_init   -> WARM-UP uniform draw (5.2)
5. randomu(seed) < p_local                  -> LOCAL move (5.3)          [local_hist[it]=1]
6. randomu(seed) < explore_frac             -> EXPLORE uniform draw (5.2 rules, no pntie) [expl_hist[it]=1]
7. else                                     -> TPE proposal (5.4)
```
Branches 5 and 6 each consume one uniform; effective per-eval probabilities in the guided phase are `P(local)=0.15`, `P(explore)=0.85*0.15=0.1275`, `P(TPE)=0.7225`.

Then, for ALL branches (5844-5850):
```
xnew = clip(xnew, opt_lo, opt_hi)
xnew[int dims] = round(xnew[int dims])
if pernight and n_elements(link_params) ≥ 1: xnew = near2m_pntie(xnew, opt_names, ndim_base, link_params)
```
`near2m_pntie(x, names, ndim_base, bases)` (1001-1015): for each base b, `w = where(strpos(names[0:ndim_base-1], b+'_n') eq 0)`; if ≥2 matches, `x[w] = x[w[0]]` (all slots take the **first** night's value). Empty-string bases skipped.

### 5.2 Warm-up / random / explore draw (5782-5794, 5820-5831)
```
for d in 0..ndim-1: xnew[d] = lo[d] + U*(hi[d]-lo[d])                # ndim uniforms (incl. night dims)
for d with strpos(name,'k_klip')==0: xnew[d] = kgrid[min(floor(U*nkg), nkg-1)]   # one more U per k dim
[warm-up only] if pernight and warmstart: xnew = near2m_pntie(xnew, opt_names, ndim_base)   # all 9 bases tied
if ndim > ndim_base:
   two-slot: for each slot: xnew[d] = (U < night_pinc) ? 0 : float(1 + floor(U*nnights))   # 1 or 2 uniforms (IDL ?: is lazy)
   legacy:   xnew[d] = (U < night_pinc) ? 1 : 0
```
Explore draws (branch 6) are identical **except no warm-start tie** (link_params tie still applies afterwards). In `search_mode='random'`, branch 4 is taken for every eval (warm-start tie applied to all if pernight+warmstart).

### 5.3 Local move (5795-5819)
```
yv2 = Y[0:it-1]; ggd = where(finite(yv2) and yv2 > -9000)
if none: uniform draw (dims + kgrid; NO night-dim special draw -> night dims uniform in [0,nnights] then rounded)
else:
   os   = ggd[reverse(sort(yv2[ggd]))]          # best first
   neu  = clip(round(n_elite), 1, nggd)
   base = os[min(floor(U*neu), neu-1)]
   xnew = X[*, base]                            # full copy incl. night dims
   ng2  = clip(round(gamma*nggd), 1, max(nggd-1,1)); gd2 = os[0:ng2-1]
   dsel = min(floor(U*ndim_base), ndim_base-1)  # ONE reduction dim, never a night dim
   bw   = near2m_bw(X[dsel, gd2], lo[dsel], hi[dsel])
   xnew[dsel] += bw * randomn(seed)
   if k dim: xnew[dsel] = near2m_ksnap(xnew[dsel], kgrid)
```

### 5.4 TPE proposal (5832-5843)
```
i0k  = (it ≥ 2) ? 1 : 0          # exclude slot 0 (unprojected calibration default) from the KDE training set
xnew = near2m_propose(X[*, i0k:it-1], Y[i0k:it-1], opt_lo, opt_hi, gamma, ncand, seed, pbest=seed_best_frac)
for k dims: xnew[d] = near2m_ksnap(xnew[d], kgrid)
```
Night dims are proposed by TPE as continuous values and rounded by the generic int rounding.

### 5.5 Decode, scalar mapping, per-night table (5853-5896)
- `seqv_use` from `near2m_nightsel` (§2.3); `wsel = where(sel eq 1)`.
- `X[*,it] = xnew` (first write, 5859).
- Scalars start at defaults and are overwritten by exact-name matches (`case opt_names[d]`): `bin→bn, inrad→ir, outrad→orad, width→orad=min(ir+w,rcap), n_ang→na, filter→fl, angsep→asep, anglemax→amax, corr_thresh→cthr, noise_max→nmax, coronoise_max→cnmax, k_klip→kprop=max(round,1)`. In pernight mode none of the suffixed names match → scalars remain defaults until replaced by per-night medians (6011-6012).
- pernight: `pn = near2m_pndecode(xnew, …)` → per-night arrays over ALL `seq_values` (unsearched → defaults; `kk` default 10).

### 5.6 Feasibility projection (reference-count guardrail)
Helpers: `near2_ref_getnight(seq, root)` loads/caches `AB_parang_clean.sav` (angles) and `AB_frametags.sav` (corrs, noises, coronoise).
`near2_ref_binangles(night, cthr, nmax, cnmax, bin, k, outrad, pxscale)` (4238-4285): replicate frame selection
(`keep = cor ≥ cthr and (noi ≤ nmax*mean(noi) or NaN) and (cor2 ≤ cnmax*mean(cor2) or NaN)`; applied only if
`minkeep=max(round(k)*round(bin),3) ≤ nkeep < nf`), then angle-aware binning: consecutive frames grouped into bins of ≤`bin`
frames, closing early if PA span would exceed `bin_dthmax = 0.5*fwhm_px(11.25µm)/max(outrad,1) rad → deg`; binned angle = mean.
`near2_ref_arcdist(ir, orad, pxscale) = 360*((11.25e-6/8.2)*206265/pxscale) / (2π*max(ann_mid,1))`, `ann_mid = ir + |orad-ir|/2` (deg of rotation per unit angsep, angsep in λ/D at the annulus mid-radius).
`near2_ref_fraction(bang, angsep, anglemax, arcdist, nmin)`: fraction of binned frames t with `count(|Δpa| ≥ angsep*arcdist and |Δpa| ≤ anglemax, excluding t) ≥ nmin`.
`near2_ref_maxangsep(…, aslo, ashi)`: largest feasible angsep in [aslo,ashi] by 41-step bisection (returns `aslo` if even `aslo` infeasible, `ashi` if `ashi` feasible).
`near2_ref_minanglemax(…, amlo, amhi)`: smallest feasible anglemax in [amlo, amhi] (returns `amhi` if even that infeasible).

**Per-night path** (5906-5938; requires `pernight and ~fast`; runs for **all** `nnights`, including dropped nights):
```
israndom = (it < n_init) or expl_hist[it]
arc_e = arcdist(ir, orad)
for jn in 0..nnights-1:
   nite = getnight(seq_values[jn]); skip if !ok
   kk_n = (pn_kk[jn] ≥ 1) ? pn_kk[jn] : k_klip_max
   bang = binangles(nite, pn_cthr[jn], pn_nmax[jn], pn_cnmx[jn], pn_bn[jn], kk_n, orad, pxscale); skip if ≤1
   as_j = pn_asep[jn]; am_j = pn_amax[jn]
   if fraction(bang, 0, am_j, arc_e, n_min_ref) < ref_frac:
        am_j = minanglemax(bang, 0, arc_e, n_min_ref, ref_frac, am_j, amax_hi); as_j = 0      # anglemax only raised
   else:
        asfeas = maxangsep(bang, am_j, arc_e, n_min_ref, ref_frac, 0, max(as_j,0))
        if as_j > asfeas: as_j = israndom ? U*asfeas : asfeas                                # random: re-draw interior; guided: project to boundary
   pn_asep[jn]=as_j; pn_amax[jn]=am_j
write back into xnew for dims named 'angsep_n<id>' / 'anglemax_n<id>';  X[*,it] = xnew
```
**Global path** (5947-5970; `~pernight and ~fast`; any search_mode): loop over **included** nights `seqv_use` only,
`kk_ref = kprop ≥ 1 ? kprop : k_klip_max`, `bang = binangles(nite, cthr, nmax, cnmax, bn, kk_ref, orad)`; per night compute
`(as_jn, am_jn)` with the same two-case rule but **no random re-draw** (always projects to the boundary); combine as
`amax = max over nights`, `asep = min over nights`; write back into `xnew['angsep'], xnew['anglemax']`; `X[*,it] = xnew`.
Note the projected values are recorded in `X` (hence in the log and KDE history), but the projected `angsep` is a float even though
it is later rounded nowhere (angsep is non-int); `anglemax` after projection may be non-integer (bisection result) despite `opt_int=1`.

### 5.7 Grid mode (5757-5781)
All dims default-valued via a `case` on the *exact* name (`inrad, outrad, angsep, anglemax, corr_thresh, noise_max, coronoise_max, width`, else `two_slot?0:1`), then
`gpts = max(round((n_iter-1)^(1/3)), 2)`, `gi = (it-1) mod gpts^3`, `i1 = gi mod gpts` → `bin`, `i2 = (gi/gpts) mod gpts` → `filter`,
`i3 = (gi/gpts²) mod gpts` → `n_ang`, each `lo + (hi-lo)*i/(gpts-1)` (then rounded). Consequences worth knowing:
- `k_klip` is **not** in the case → set to 0 (two-slot) or 1 → clamped to `opt_lo=1` → **grid mode always evaluates k_klip = 1 when `search_k`** (only `scan_mode` gives grid a real k choice).
- In pernight mode no suffixed name matches → grid degenerates to all-lower-bounds. Grid is only meaningful global.
- Cells are visited in sequence and wrap after `gpts^3` cells; angsep/anglemax then go through the global census projection.

### 5.8 Fresh injection positions per eval (5975-5982)
Annular: `near2m_randpos(nsrc_a, rlo*pxscale, rhi*pxscale, …) → inj_rho, inj_theta` (opt_width: `rmid_e` both bounds); `inj_contrast = replicate(ccal, n)`. **Then** the scoring paths draw a second set `hrho/htheta` (6060-6062, 6105-6107, 6177-6179) which is what is actually scored; the first set only feeds the legacy scan (`scan_rho`) and the clean tile of best-updates. Two `randpos` calls per eval → 2+ uniforms consumed.

### 5.9 Reduction + scoring paths
Common: `eval_failed = 0` (5989); `img_noinj` computed only for `use_noinj_noise` (k-scan clean).

**(a) pernight + search_k (production; 6056-6098)**
```
wsel2 = wsel (included nights) ; bnL..cnmxL = pn_*[wsel2]
representative scalars: bn=round(median(bnL)), na, fl, amax=round(median), asep=median, cthr/nmax/cnmax=median
kpnL = max(round(pn_kk[wsel2]),1); kk_pn = kpnL
draw hrho/htheta (randpos); hinj = replicate(ccal, n)
dh = near2m_reduce_pn(1, kpnL, 0, ir, orad, bnL, naL, flL, asepL, amaxL, hrho, htheta, hinj, ..., cthrL, nmaxL, cnmxL,
                      nightcube=ncube_h, also_clean=clean_subtract, cleancube=ncube_hc)
if ncube_h valid:
   wgt_pn = near2m_pnweights(ncube_h)  = 1/N uniform (1311-1320)
   img_h  = near2m_pncombine(ncube_h[...,1 k], zeros, wgt_pn)   # NaN-aware weighted mean over nights
   snr_src = near2_snrpk(radprof(img_h), cx, cy, hrho, htheta, pxscale, fwhm); msnr = near2m_med(snr_src)
   if clean_subtract and ncube_hc valid:
       img_hc = pncombine(ncube_hc, ..., wgt_pn); snr_cs = near2_snrpk(radprof(img_hc), ..., hrho, htheta)
       cur_orig_g = med(snr_src); snr_src = snr_src - near2m_clip0(snr_cs); msnr = med(snr_src) (NaN -> -9999)
       cleansub_done = 1
   bestk = max(round(median(kpnL)),1); Kbest[it] = bestk
   per night j: ynight_hist[night, it] = med(near2_snrpk(radprof(ncube_h[*,*,j]), ..., hrho, htheta))   # RAW per-night S/N
                knight_hist[night, it] = kpnL[j]
```
If `ncube_h` is missing, `Kbest[it]` stays 0, `msnr=-9999`, `okev=0`.

**(b) pernight + scan_mode (legacy; 6017-6055)**: 4-D scan `near2m_reduce_pn(1,[k_klip_max],1,…)` → `near2m_pnkpick` (greedy coordinate ascent on combined S/N, uniform weights, 2 passes) → `kk_pn`; `bestk=round(median(kk_pn))`; `Sscan[0:nk-1,it] = snrk` (combined S/N vs common k); honest re-score at fresh positions with `near2m_reduce_pn(1, kk_pn, 0, …)`; `msnr = med(snr_src)`. Clean-subtraction here happens in the fallback block (5.10).

**(c) global + search_k (6100-6135)**: `kk_use = kprop ≥ 1 ? kprop : max(round(0.5*k_klip_max),1)`; `Kbest[it] = kk_use` (always set, even if the reduction fails); draw `hrho/htheta`;
`clean_subtract ? near2m_reduce_ics(kk_use, …, cleanimg=img_hc, nightcube=ncube_h) : near2m_reduce(1, kk_use, 0, …, nightcube=ncube_h)`;
`snr_src = near2_snrpk(radprof(img_h), …)` (or `near2_snr_inj(img_h, sl_noinj, …)` for the difference metric, with an extra clean reduce);
if `clean_subtract and img_hc valid`: `snr_src -= clip0(near2_snrpk(radprof(img_hc), …))`; `msnr = med(snr_src)`.
Night averaging in `near2m_reduce` is `near2m_avg` = per-pixel NaN-aware mean **weighted by sqrt(texp)** per night (450-470) — unlike the uniform per-night combine of the pernight path.

**(d) global + scan_mode (6137-6202)**: k-scan cube at `inj_rho`; `snrk[k]`; `bestk = argmax+1`; `Sscan[…]`; then honest re-score at fresh positions at `bestk` (`reduce_ics` if clean_subtract); subtraction as in (c).

### 5.10 Failure handling and clean-subtract fallback (6207-6243)
- `if eval_failed: okev=0; msnr=-9999` (set by `near2m_reduce_pn` when a night still fails after one retry, 840-844). `if ~finite(msnr): msnr=-9999`.
- Fallback clean-subtract block runs when `clean_subtract and ~keyword_set(cleansub_done) and okev and msnr > -9000 and snr_src[0] > -9000`: one clean reduction at the eval's config (pernight: `reduce_pn(0, kk_cs, …)` at `inj_rho` (= `hrho` by now), combined with `wgt_pn`; global: `near2m_reduce(0, max(round(bestk),1), …)`), then `snr_src -= clip0(near2_snrpk(radprof(img_cs), …, inj_rho, inj_theta))`, `msnr = med(snr_src)`.
- **Apparent bug (do not port):** `cleansub_done` is only ever assigned inside the pernight branch (6015, 6083). In the **global** paths (c)/(d) it is undefined → `keyword_set` = 0 → the fallback block runs **after** the in-path subtraction already happened → the clean S/N is subtracted **twice** in global+clean_subtract mode (and `cur_orig_g` records the once-subtracted aggregate). Production (pernight) is unaffected. A port should subtract exactly once on every path.
- `Y[it] = msnr` (6245). No duplicate-proposal detection exists in the search loop (identical configs are simply re-evaluated); dedup happens only when selecting validation candidates.

### 5.11 Score storage
`X[ndim,n_iter]` (projected/tied values), `Y` (-9999 = failed, NaN = never run), `Kbest[n_iter]` (long; representative k; 0 = unset), `expl_hist`, `local_hist` (bytes), `Sscan[k_klip_max,n_iter]` (scan modes only), `ynight_hist[nnights,n_iter]` (raw per-night median S/N, pernight), `knight_hist[nnights,n_iter]` (per-night k), plus the best-record scalars (`besti, best_snr, best_k, best_img, best_noinj, best_bn/na/fl/as/am/ct/nm/cnm/or/seqv, best_injr/injt/injc, best_srcx/srcy, best_snrsrc, best_kpn, best_pn_*`).

### 5.12 Best update (6368-6567)
Condition: `msnr > best_snr and bestk ≥ 1` (strict; ties keep the earlier best; the calibration seed counts). Sets `best_snr=msnr, besti=it, best_img=sl_inj`, then (raw metric) one clean reduction at the same config/k for the tile (`reduce_pn(0, kk_pn, …)` combined with `wgt_pn` in pernight; `near2m_reduce(0, bestk, …)` global) → `best_noinj`; records all `best_*` incl. `best_injr/injt = inj_rho/inj_theta` (= scored positions), `best_kpn = kk_pn`, `best_pn_wgt`; writes `klip_best_eval_inj.fits` / `klip_best_eval.fits` (rundir) with the header keys listed at 6456-6486; then (if `bench_tag eq 'none'`) the live contrast preview and a KLIP-FM preview pass (§9.3) — these run extra reductions but affect no search state except `pcr/pcc/pfmr/pfmc` (display) and `fm_best_full/inj_best_full`.

---------------------------------------------------------------------------------------------------

## 6. Logging / text products

### 6.1 `optimize_tpe_results.txt`
Opened at 4875-4884. Location: **shared** file `<root>comb/opt/optimize_tpe_results.txt` (truncated on a fresh run; on resume/extend the run copy is copied over it and it is opened `/append`), and after every log line it is copied to `<rundir>/optimize_tpe_results.txt` (7153-7154). Concurrent runs would clobber the shared copy.

Header (one line): `'# annulus iter  ' + strjoin(opt_names,' ') + '   k_klip   medSNR'`
→ tokens after `#`: `annulus iter <opt_names…> k_klip medSNR` (so `ndim = ntok - 4`; `near2m_load_annhist` derives columns from this header).

Row (7149-7151), written **after** the checkpoint save, once per completed loop iteration:
```
string(ia+1, it+1, format='(I3,1x,I4)') + '  ' + strjoin(string(X[*,it], format='(F8.3)'), ' ')
  + string(bestk, format='("   ",I5)') + string(msnr, format='("   ",F8.3)') + kpn_str
kpn_str = pernight and kk_pn[0] ≥ 1 ? '   kpn=[' + strjoin(strtrim(fix(kk_pn),2), ',') + ']' : ''
```
Parsing notes: (i) whitespace-separated tokens: `annulus, iter, ndim floats, k_klip, medSNR[, 'kpn=[k1,k2,...]']` where the kpn list covers the **included** nights in `seqv_use` order; (ii) `msnr=-9999.` does not fit `F8.3` → IDL prints `********`; treat asterisks in `medSNR` as failed (−9999); (iii) `X` values are the projected/tied/rounded vectors actually run; (iv) duplicate `(annulus, iter)` rows can appear after a re-cal restart — last wins; (v) slot 0 (calibration) is normally absent (iter starts at 2) unless the eval-0 path ran; (vi) `bestk` is the median per-night k in pernight mode, `kk_use` in global, and may be 0 on a failed pernight eval; (vii) validation trials are **not** logged here.

### 6.2 `run_setup.txt` (4895-4947), in rundir
Lines (label padded to 19 chars + `: `): `run_dir, KLIP_mode, ref_guardrail, clean_subtract, metric, injection model, search_mode, annular / opt_width, ann_edges (px) [F0.2 list], nann, n_iter / n_init, k_klip_max, n_valid / n_top, gamma/ncand/explore, seed_best_frac, p_local / n_elite, cmax_cal (ceiling), use_contrast, cmap (display), opt_nights, night_pinc, opt_framesel, pernight, warmstart, nights (seq_values), seed / bench_tag, snr_excl / snr_srch, known_src (mask), kfit_order, nbridges, use_workers / wpn, cache / lean, save_cubes, legacy_stitch, pxscale / fwhm(px)`; blank; `# searched parameters  (name: [lo, hi]  integer?)` then one line per dim `'  ' + A-14 name + ': [' + F0.3 lo + ', ' + F0.3 hi + ']  int=' + 0/1`; blank; `# fixed defaults / seed configuration` + two lines of `def_*` + `  injection contrast (anchor) = E10.3`.
Also `scripts/` copy of sources.

### 6.3 Per-step setup files `near2m_write_setup` (504-536)
`calibNNNN_setup.txt` (phase 'calib', step = trial index), `evalNNNN_setup.txt` (phase 'eval', step = it+1; only when `save_cubes and okev and bestk ≥ 1`), `valid_candNN_setup.txt` (phase 'valid', step = source eval index e2+1), `final_setup.txt` (phase 'final'). Content: `phase, annulus, step, [bin, n_ang, filter, angsep(F0.3), anglemax(F0.1), corr_thresh(F0.3), noise_max(F0.3), coronoise_max(F0.3), k_klip — omitted when pernight=1 is passed (only final_setup)], inrad/outrad (F0.1 px), nights, median_SNR (F0.3, if finite)`, then `# injected sources:  rho_arcsec   PA_deg     contrast` + rows `F10.3 F10.2 E12.3` (or `# injected sources: none …`). pernight appends `per-night k*  : n1=k n2=k …` to valid files and a `# PER-NIGHT winner block` table + `per-night combine weights` line to `final_setup.txt` (7633-7648). `near2m_readcalcon` reads the contrast = 3rd token of the line after "injected sources".

### 6.4 Other text products
- `runinfo.txt` (4735, rewritten 8315): `search_mode(A-8)  n_iter-list  n_init-list  nann(I4)  bench_tag`.
- `bench_summary.txt` (shared, append; 7894-7896): `mode(A-8) ia+1(I3) nann(I3) n_iter(I5) n_init(I4) Y[0](F9.3) best_snr(F9.3) stamp` — `Y[0]` = seeded default score, `best_snr` = **validated** winner score (if validation ran).
- `contrast_curve.txt` (rundir after each annulus 7683-7695; `comb/opt/` + copy at end 8285-8312): header lines, then `F10.3 3x E12.4` (sep arcsec, 5σ contrast) sorted by separation; optional KLIP-FM section with its own two `#` lines.
- `klip_stitched_params.txt` (8253-8270): one row per annulus (format at 8267) + nights.
- checkpoint: `checkpoint.sav` (§7).

---------------------------------------------------------------------------------------------------

## 7. Checkpoint / resume

### 7.1 Written (7103-7140) every eval, after display and **before** the log line; atomic (`.tmp` → move). Structure `ckpt`:
`ia, it, n_iter, n_init, ndim, wall (elapsed s), X, Y, Kbest, expl, loc, knh (knight_hist), Ss (Sscan), ccal, injc (inj_contrast), besti, bsnr, borig, bk, bnoinj, bimg, bsx, bsy, bssrc, Xc, yc, wm, ev (corner arrays), annedg, nann, anno (ann_noinj ptrs), aninjr, aninjt, absnr, akk, aninjc, abn, ana, afl, asm, atm, add, ccrall, cccall, csr, csc, fmrall, fmcall, btag, bor, bsnrk, bseqv, bbn, bna, bfl, bsm, btm, bdd, bas, bam, bct, bnm, bcnm, bkpn, binjr, binjt, binjc, ynh (ynight_hist), seed (RNG state), cfg`.
`cfg` (5014-5033) = struct `cfgver=1` + one pointer field per keyword name in the `kwn` list (all keywords except `resume`/`extend_ann`); null pointer = keyword not passed (re-defaulted identically on resume). Snapshot taken once, after defaults, before the annulus loop (so `n_init/n_iter` are the original possibly-vector values and `seed` is whatever the RNG variable holds at that time).

### 7.2 Restore (4361-4382, 5035-5061, 5160-5177, 5599-5676)
1. `resume=` resolves `<dir>/checkpoint.sav` or `<root>comb/opt/<basename>/checkpoint.sav`; `restore`; for every valid pointer in `cfg` the keyword variable is (re)created via `scope_varfetch` → all later `if n_elements(kw) eq 0` defaults are skipped. Keywords passed alongside `resume` are overwritten (config lives in the checkpoint).
2. Run identity: `rundir = saved dir`, `stamp` from its name.
3. If `ndim eq ckpt.ndim`: `ann_edges, nann, seed, ann_noinj, ann_injr/injt/injc, ann_bsnr, ann_kk/bn/na/fl/sm/tm/dd`, the cumulative contrast arrays, `ia_resume=ckpt.ia`, `it_resume=ckpt.it` restored; else "RESUME ABORTED" → fresh run in the same dir. NOTE: `ann_as/am/ct/nm/cnm/ir/or/ann_inj/ann_seqv/ann_nightcube*` are **not** checkpointed → the final stitch of a resumed multi-annulus run lacks earlier annuli's night cubes and uses zeros for those fields in `klip_stitched_params.txt`.
4. Annulus loop starts at `ia_resume`. For that annulus: if `ckpt` has tag `BTAG` and `annulusNN/calib0001_setup.txt` exists → `resume_skip` → `goto RESUME_ANN` (calibration skipped entirely; `recal_budget=0`, `recal_changed=1`, `kbest_c = ckpt.bk-1`). If no calib on disk → fresh calibration+search for this annulus, checkpoint state discarded from here on. Old checkpoints without `BTAG` re-run calibration then fall through to `RESUME_ANN`.
5. `RESUME_ANN`: `X,Y,Kbest,expl_hist,local_hist` (padded to `n_iter`), `knight_hist, Sscan, ccal, inj_contrast, besti, best_snr, best_k, best_noinj, best_img, best_srcx/y, best_snrsrc, Xcorner…, seed, t_start -= wall, it_start = ckpt.it+1`, full best-record if `BTAG`. Sticky `n_iter`: `if ckpt.n_iter > n_iter: n_iter = ckpt.n_iter` (5098-5099).
6. Log: the run copy is copied back to the shared log and appended (4875-4880), so history survives; the checkpointed eval `ckpt.it` is never re-run (its row may be missing from the log if the crash hit between checkpoint and log write — X/Y in the checkpoint are authoritative).
7. `resuming` cleared after the entry annulus; later annuli run fresh. Display/FM panels are recomputed at the first eval via `resume_bestfm` (a clean FM pass + an injected pass at the best config; no score impact).

### 7.3 Extend (`extend_ann`, 5064-5088, 5192-5218, 5682-5724)
Iterates all annuli from 0; per annulus target `et_ann`: ≤0 untouched; ≤ existing → validate-only (`n_iter = nprev_ext`, empty loop); > existing → continue search to `et_ann`. Contrast frozen to the original (`ann_injc[ia]` or `calib*_setup.txt`). History spliced from the checkpoint (entry annulus) or from the log (`near2m_load_annhist`, requires header ndim match; slot 0 NaN from log). The prior best `x_bestprior = X[*, argmax Y]` is re-scored as the first continued eval. Old contrast-curve samples of that annulus's band are dropped before re-validation.

---------------------------------------------------------------------------------------------------

## 8. Validation (7174-7712), runs when `n_valid ≥ 1`

1. Pool: `pool = where(finite(Y) and Y > -9000 and Kbest ≥ 1)` over ALL slots (calibration seed, warm-up, TPE). `ord = reverse(sort(Y[pool]))` (descending; IDL `sort` not stable).
2. Candidates: walk `ord`; accept `e2` unless `total(abs(X[*,e2]-X[*,cand])) < 1e-4` for an already-accepted candidate (full-vector L1, includes night dims); stop at `ntop_eff = min(n_top, npool)`.
3. Per candidate `c` (source eval `e2`):
   - decode scalars from `X[0:ndim_base-1, e2]` by exact name (global) ; `seqv_c` via `near2m_nightsel`; `kc = max(Kbest[e2],1)`.
   - pernight: `pnv = pndecode(X[*,e2])`, subset to `seqv_c` (`wcv`), representative medians; **one `randpos` draw** (positions unused for the clean image, but consumes RNG); `kpn_v = max(round(pnv.kk[wcv]),1)` (search_k) or from a scan + `pnkpick` (scan_mode); `kc = max(round(median(kpn_v)),1)`; clean per-night cube `reduce_pn(0, kpn_v, …)` → `wgt_v = pnweights` (uniform) → `img_vc = pncombine`.
   - global: `img_vc = near2m_reduce(0, kc, 0, ir, orad, bn, na, fl, sm, tm, dd, asep, amax, …, seqv_c, corr_thresh=cthr, noise_max=nmax, coronoise_max=cnmax)` (uses `X`'s already-projected angsep/anglemax; no re-projection).
   - If `img_vc` valid: for `t in 0..n_valid-1`: fresh `randpos` (band `rlo..rhi`), `inj_contrast = ccal`; injected reduction at the same config/k (`reduce_pn(1, kpn_v, …)` combined with `wgt_v`, or `near2m_reduce(1, kc, …)`); **raw** metric `snr_vec = near2_snrpk(radprof(img_vi), cxv, cyv, inj_rho, inj_theta, …)` (or `near2_snr_inj` for the difference metric) — **no clean subtraction regardless of `clean_subtract`**; `vs[t] = near2m_med(snr_vec)`; samples `(inj_rho, snr_vec)` appended to `vsmp_r[c], vsmp_s[c]` for the contrast curve.
   - Committed trial = the one with `vs` closest to `vmed = near2m_med(vs)` (`min(abs(vs-vmed))`, first minimum) → its image/positions/`snr_vec` become the candidate's products.
   - `vscore = near2m_med(vs)` (median over trials, `/even`); `valid_candNN_setup.txt` written with `vscore`.
   - Election: `if finite(vscore) and vscore > vbest` (strict → ties keep the earlier = higher-search-score candidate; `vbest` starts at `-1d99`).
4. Winner replaces the best-record: `best_snr = vbest, besti = vwin, best_k = vw_k, best_noinj = vw_noinj (clean image), best_img = vw_inj (median trial), best_* params, best_seqv, best_srcx/y, best_snrsrc, best_kpn/pn_*`, `best_tag` gets "(validated)". `Y` is **not** modified. `bench_summary` and `ann_bsnr[ia]` record `vbest`.
5. If no candidate produced a finite score (`vwin < 0`) the search best stays.

---------------------------------------------------------------------------------------------------

## 9. Products

### 9.1 Per-annulus (after validation)
- **Injection-calibrated 5σ contrast curve** (7524-7564), from the winner's pooled validation samples `(smp_r [arcsec], smp_s [S/N])`, needs ≥3 samples with `smp_s > 0`:
  ```
  rin_c = max(ann_edges[ia], fwhm) px ; rout_c = best_or (px)
  sigp = near2m_noiseprof(vw_noinj, fwhm, rin_c, rout_c, rprof)      # tophat aperture (radius fwhm/2) sum image;
        # per integer radius r in [rin,rout]: nap=floor(2πr/fwhm) apertures at angles p*2π/nap; sig = stddev(vals)*sqrt(1+1/nv) (nv≥3, nap≥4)
  sig_at = interpol(sigp, rprof, smp_r/pxscale)
  Ks   = smp_s * sig_at / ccal                 # measured throughput incl. coronagraph T
  Kalg = Ks / max(near2m_thru(smp_r), 1e-3)    # T(r) factored out (near2_n4_throughput or near2_agpm_trans)
  r_as = rprof*pxscale
  Kfit = near2m_kfit(smp_r, Kalg, r_as, kfit_order) * near2m_thru(r_as)   # robust poly fit of log K vs r, order auto-reduced, 3σ clip, clamped to sampled range; <3 pts -> median
  c5   = 5*sigp/max(Kfit,1e-12)  ;  samp_c5 = 5*ccal/smp_s
  seam trim (annular, ~legacy_stitch, nann>1): trim_as = max(round(fwhm/3),1)*pxscale (=2 px);
        keep r_as ≥ ann_edges[ia]*px + trim (ia>0) and r_as ≤ ann_edges[ia+1]*px - trim (ia<nann-1)
  append to ccr_all/ccc_all (curve) and csmp_r/csmp_c (samples)
  ```
- **KLIP-FM cross-check** (7566-7677): one clean forward-model pass at the winner config with `nfm = clip(floor((rout_c-1-rin_c)/1.5)+1, 5, 24)` test radii `fmrad = rin_c + (rout_c-1-rin_c)*i/(nfm-1)` px on a golden-angle spiral `fmth = (i*137.508) mod 360`, contrast `ccal`; `fmimg = near2m_avg(AB_median_fm.fits)`; `resp = near2_fmresp(radprof(fmimg), …)` = matched-filter peak (per-source measured kernel at each radius) within `snr_srch` px; `Kfm = resp/ccal`; `sigmf = near2m_noiseprof(radprof(vw_noinj), fwhm, rin_c, rout_c, /mfkern, rho=median(fmrad*pxscale))` (unit-sum matched-filter kernel); `c5fm = 5*interpol(sigmf, rprofm, fmrad)/Kfm` for `Kfm > 0` (≥3 needed); appended to `fmr_all/fmc_all`. Note the injection curve uses the un-flattened `vw_noinj`, the FM curve the `radprof`-flattened one.
- pernight final per-night clean cube at the winner's per-night block/k (`reduce_pn(0, vw_kpn, …)`, 7590) and injected cube (7609) → `ann_nightcube/ann_nightcubi`; `final_setup.txt`; `contrast_curve.txt`; records `ann_kk/bn/na/fl/sm/tm/dd/ir/or/bsnr/as/am/ct/nm/cnm`, `ann_noinj = best_noinj`, `ann_inj = best_img`, `ann_vwinjr/t`, `ann_seqv = best_seqv` (7902-7913).
- Running products (every eval, display block): `klip_stitched_running[_inj|_snr|_snr_inj].fits` — equal-weight NaN-aware mean over tiles `ann_edges[ja]-2 ≤ r ≤ ann_edges[ja+1]+2` using earlier annuli's `ann_noinj/ann_inj` and the current running best (6705-6741); running per-night cube stitch + candidate search (7945-8018, uses the inverse-variance weights below).

### 9.2 Final stitch (8031-8122)
```
awgt[ia] = 1 (legacy or nann==1) else:
   sa = near2m_noiseprof(ann_noinj[ia], fwhm, ann_ir[ia], ann_or[ia], rpa)
   gi = rpa in [ann_ir+fwhm, ann_or-fwhm] with finite sa>0 (fallback: all finite>0)
   sg = median(sa[gi]) (1 if invalid) ; awgt[ia] = 1/sg^2 ; awgt /= max(awgt)
stitched = Σ_ia awgt[ia]*img_ia  / Σ_ia awgt[ia]   over finite pixels of each tile, tile mask = ann_edges[ia]-2 ≤ r ≤ ann_edges[ia+1]+2
seams (pixels in [edge0, edge_nann] still NaN) patched by a padded re-run of that annulus's best config with inrad-3/outrad+3
stitched_f = radprof(stitched)  -> klip_stitched.fits  (+ header history of per-annulus best)
```
Same weights for the per-night clean/injected cube stitches (`klip_stitched_nights[_inj].fits.gz`, union of night ids; NaN where an annulus did not use a night) and for the validated injected stitch `klip_stitched_inj.fits`; S/N maps via `near2m_snrmap` (matched-filter Mawet map, per-radius MAD floor, source exclusion). Text: `klip_stitched_params.txt`.

### 9.3 Live contrast preview at each new best (6500-6566) — display only, but it consumes reductions
Same math as 9.1 using `snr_src` and the eval's source radii `sqrt((srcx-cx)²+(srcy-cy)²)*pxscale`, requires ≥2 positive sources, plus a clean FM spiral pass and a second injected-FM pass (`out_suffix='_injfm'`) at the best config. Skipped when `bench_tag ne 'none'`.

### 9.4 texp weighting
- `near2m_avg` (global night average, calibration re-measures, FM images): per-night weight `sqrt(texp)` from `n<id>/texp.sav` (1 if absent).
- pernight combine (`near2m_pnweights/pncombine`, `pnkpick`): **uniform 1/N** (inverse-variance explicitly rejected, 1312-1317).
- Display `t_exp` string sums `texp` of included nights (6613-6622); `near2m_texp` scales by the frame-selection kept fraction (927-969) for labels only.

---------------------------------------------------------------------------------------------------

## 10. Other non-obvious points

- **RNG**: single IDL stream `seed` (long → state array after first use). Consumers in order per eval: proposal uniforms (§5.1-5.4; `randomn` for the local jitter), census re-draws in random evals (pernight only, one per infeasible night), `randpos` #1 (5980), `randpos` #2 (scoring positions), best-update passes consume none. Calibration: 2 `randpos` per trial. Validation: pernight 1 `randpos` per candidate + `n_valid` per candidate; global `n_valid` per candidate. param_verify: 1. `seed` is checkpointed so a resume continues the stream. Bit-exact replay of positions requires reproducing IDL's `randomu` (Mersenne Twister in IDL ≥ 8) — probably not worth it; replaying logged `X` through the metric does not need it, but note the *positions* per eval are not logged (only in `evalNNNN_setup.txt` / FITS headers `INJRHOn/INJPAn`).
- **Sorting**: "best first" = `reverse(sort(Y))`; ties are unordered. Best-update and election use strict `>`.
- **Units**: `rho` arcsec, `theta` deg PA (E of N); annulus edges/`inrad/outrad/filter/width` px; `angsep` in λ/D units (× `arcdist` deg for the census; the FITS comment "x FWHM" is loose); `anglemax` deg; `bin` frames; contrast = companion/star flux ratio (AGPM off-axis transmission applied at injection).
- **Trial 0 exclusion from KDE** (5838): the calibration default's angsep/anglemax were never projected, so it is dropped from the TPE training set when `it ≥ 2` (still eligible as best/validation candidate).
- **Per-night representative scalars** (`bn=round(median(bnL))` etc.) are what appear in `evalNNNN_setup.txt`, FITS headers and `klip_stitched_params.txt` for pernight runs; the true per-night block is only in `X` (log) and `final_setup.txt`.
- **`Kbest` column** in the log is `bestk` (pernight: rounded median of per-night k; global: proposed k), i.e. redundant with the `k_klip[_n]` dims of `X` except for the median collapse.
- **k for the eval-0 default** is `kbest_c+1` (calibration scan argmax), not `def_k_klip=6`.
- **anglemax integer flag**: `opt_int=1` but the census projection writes back a bisection float → logged X can hold non-integer anglemax (and the KDE trains on it).
- **`nsel` night dims in local moves** are copied from the elite; in TPE they are proposed continuously then rounded — no `night_pinc` bias applies outside random draws.
- **Effective explore rate** 0.1275 (branch order), and warm-up slots include slot 0 (`it < n_init` test on the slot index), so with `n_init=50` there are 49 real random draws after the calibration seed.
- **`wall` clock, ETA, telemetry** (7160-7169): realized local/explore fractions computed over slots `n_init..n_iter-1`.
- **Shared files in `comb/opt/`** (`optimize_tpe_results.txt`, `bench_summary.txt`, `contrast_curve.txt`, `klip_stitched_params.txt`) are cross-run; rundir copies are the per-run truth.
- Frame-selection thresholds `noise_max/coronoise_max` are multiples of the night mean of the tag; `corr_thresh` is absolute (`near2_ref_binangles`, 4248-4250).
