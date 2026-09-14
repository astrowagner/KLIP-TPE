# D. The four IDL optimizer variants — what differs, what is core, and what the Python seams must support

Source files (all under `/mnt/user-data/uploads/idl/`):

| Variant | Optimizer | Reducer | Lines |
|---|---|---|---|
| NEAR (reference, brief §3–§7) | `optimize_near_2_tpe.pro` | `reduce_near_2.pro` | 8399 / 2401 |
| MWC (JWST/NIRCam F430M, MWC 758c) | `optimize_mwc_tpe.pro` | `reduce_nircam_coro.pro` | 8686 / 2722 |
| LMIR (LBT/LMIRCam L', RXJ0534) | `optimize_lmircam_tpe.pro` | `reduce_lmircam.pro` | 3346 / 6562 |
| HPB (LBTI Fizeau M-band, HP Boo) | `optimize_hpboo_tpe.pro` | `reduce_hpboo.pro` | 2428 / 1046 |

Line numbers below are `file:line`. `mwc` = optimize_mwc_tpe.pro, `lmir` = optimize_lmircam_tpe.pro, `hpb` = optimize_hpboo_tpe.pro, `near` = optimize_near_2_tpe.pro, `rnc` = reduce_nircam_coro.pro, `rlm` = reduce_lmircam.pro, `rhp` = reduce_hpboo.pro.

---

## 1. Per-variant descriptions

### 1.0 NEAR (reference; summarized only where needed for comparison)

- Constants: `near:4675` `lambda=11.0E-6, diam=8.2, pxscale=0.0456`, `fwhm = 1.028*(lambda/diam)*206265/pxscale` (≈6.2 px).
- Defaults: `n_init=40` (4414), `n_iter=200` (4466), `gamma=0.25`, `ncand=48`, `explore_frac=0.15` (4478), `seed_best_frac = pernight?0.5:0` (4477), `p_local=0.15`, `n_elite=5` (4496–4497), `opt_nights=1`, `opt_framesel=1` (4512–4513), `n_valid=8`, `n_top=3` (4516–4517), `snr_excl=1.5` (4519), `link_params=['bin','filter']` (4427), `cmax_cal` off unless passed (4518).
- Base space `['bin','n_ang','filter']` (+`inrad,outrad` when non-annular, 4770/4782) + `angsep,anglemax` (slow mode) + framesel triple + `k_klip` on the kgrid; per-night replication + `drop1/drop2` two-slot jackknife.
- Reference-count guardrail `near2_ref_*` (4208–4330) — **present only in NEAR**; none of the other three have any analogue (`grep mwc_ref_` → 0 hits).

### 1.1 MWC — JWST/NIRCam coronagraphy, single dataset, ARDI

**Instrument constants** — `mwc:5014-5015`: `lambda=4.30E-6, diam=6.5, pxscale=0.063`, `fwhm=1.028*(lambda/diam)*206265/pxscale ≈ 2.2 px`. Matched filter is a **fixed 2.0-px circular Gaussian** regardless of fwhm (`mwcm_mfkern`, 362–396, `mf_fwhm=2.0`; the changelog A1 explains why: high-pass+KLIP sharpen the reduced core, the per-separation WebbPSF kernel over-smoothed).

**Data layout / inputs** — root `/Users/kevinwagner/Data/JWST/MWC758_Cycle2/` (4876); reducer writes `reduced/F430M_ARDIKLIP[_inj|_cs|_fm].fits` (848–911). A symlink shim (4884–4892) exposes the single reduction under NEAR-style `n1/AB_median_klip[_inj].fits` so inherited multi-night code paths resolve. Two rolls of the science target + a reference-star (HD 36575) library: full RDI library is 25 alt-roll + 25 PSF-ref frames = 50 → `k_klip_max=50` (4903–4906). Frame quality tags are not used (`opt_framesel=0`, 4832).

**Searched parameters** (`mwc:5128-5165`, the non-annular branch actually taken because `annular=0; nann=1` is forced at 5078):

| name | lo | hi | int | notes |
|---|---|---|---|---|
| `combtype` | 0 | 2 | 1 | **categorical**: index into `['mean','median','nwadi']` |
| `klipfilter` | 5 | 9 | 1 | high-pass width px (README says 0–10; code now 5–9) |
| `nkalt` | 0 | 25 | 1 | # alt-roll refs kept per target (25 = all) |
| `nkpsf` | 1 | 25 | 1 | # PSF-ref frames kept (floor 1 so it is always ARDI, never pure ADI — comment 5134) |
| `k_klip` | 1 | 50 | 1 | appended by `search_k` (5163–5164), sampled on `mwcm_kgrid` |

`inrad=2, outrad=15, n_ang=1` are **fixed** (5203; the README's "8 knobs" incl. inrad/outrad/n_ang is stale). `fast=1` default (4833) so `angsep/anglemax` are not searched — and in fact `mwcm_reduce` ignores bn/fl/asep/amax entirely (comment 866–868).

**Categorical encoding**: the categorical lives in the continuous TPE vector as a float in [0,2]; decode is `ct=(['mean','median','nwadi'])[(round(x)>0)<2]` at `mwc:6215` (eval loop) and `7444` (validation). Bandwidth floor `0.08*(hi-lo)=0.16` and `round()` after clamping (6176) mean TPE effectively does a KDE over category *indices* with implicit ordinal adjacency (mean↔median↔nwadi). This is a known crudeness to fix in Python (see §3). Categorical value is published to the reducer through `common mwc_knobs` (`kn_ct,kn_kf,kn_nka,kn_nkp`, 6223) — the source of changelog bugs A3/A4.

**Reduce-wrapper signature** — `mwcm_reduce(inject, kklip, kscan, ir, orad, bn, na, fl, sm, tm, dd, asep, amax, inj_rho, inj_theta, inj_contrast, root, seqv, nbridges, lean, cache, nthreads, fm_rho=, fm_theta=, fm_contrast=, corr_thresh=, noise_max=, coronoise_max=, out_suffix=, nowait=, jobflags=, nightcube=, combtype=, klipfilter=, nkalt=, nkpsf=, reddir=, rfilt=, rstyle=)` (848–855) — the NEAR positional signature kept verbatim with NIRCam knobs bolted on as keywords. It calls
`reduce_nircam_coro, obj='MWC758', filter='F430M', /ardiklip, /fixrad, [/fast], rho=, theta=(PA+90)*!DTOR, contrast=, kklip=, combtype=, klipfilter=, nang=, optinrad=, optoutrad=, refsel=0|1, nkalt=, nkpsf=, nodiag=1, out_suffix=` (889–904). `fullib = (nkalt>=25 and nkpsf>=25)` selects the `/fast` per-roll RDI path, otherwise the slow per-target CC-based reference selection (`rnc:1977-1982, 2202-2270, 2407-2425`). Reducer-side keyword plumbing: `rnc:1-4` signature, `rnc:263-270` overrides (`kklip→k_klip, combtype→comb_type, klipfilter→klip_filter, nang→n_ang, optinrad/optoutrad→inrad/outrad`), `rnc:285` `out_suffix→suffix`, `rnc:170` `nopsf` skips synth-PSF build on the clean pass.

`mwcm_reduce_ics` (931–989) is the **clean_subtract** driver: it spawns injected and clean reductions as two background `idl` processes (batch files + done-flags, 954–976), with targeted fallback to run only the missing pass in-process (981–986). Outputs isolated by `out_suffix='_cs'`.

**Metric** — three objective functions coexist:
- `mwc_snrpk` (460–565) — **default search and validation metric**: declip (`mwcm_declip`, 413–424, 6σ MAD 3×3), 2-px Gaussian matched filter, peak located within `snr_srch=1.5 px`, per-FWHM ring excluding injected sources (1.5 FWHM), known real sources (`common mwc_srcmask`, default c at ρ=0.607″ PA=213.8°, 4864–4865) and a **hand-drawn disk/contaminant pixel mask** (`mwcm_diskmask`, 434–447, 320×320 absolute pixel boxes, `disk_on=1b` at 4897–4899); **radial-band σ floor** (|r′−r|≤0.6 FWHM band, 519–552) so a starved ring can't deflate the denominator; **strict-Mawet penalty on retained aperture count** `nret=nclean` (553–560, changelog A2). Returns `(A_pk − bg)/(σ·sqrt(1+1/nret))`, can be negative. Always fed `radprof(img)` (azimuthal-median-flattened).
- `mwc_snr_inj` (128–217) — injection-difference metric (`use_noinj_noise=1`, off by default 4729): signal = aperture flux on (inj−clean), noise = clean ring σ with half-FWHM ring resampling when starved (192–206), penalty on `nret` (212–213).
- `mwc_snr` (67–118) — legacy plain aperture Mawet; unused by the loop.
- `clean_subtract=1` **default** (4838): search score `= snr_inj − mwc_clip0(snrpk(radprof(clean)))` (6379; `mwc_clip0` 455–457; also applied during calibration 5664/5679). Validation is raw `mwc_snrpk` (7549–7551). Aggregate = `mwcm_med` (NaN-robust median, 719).
- Extra k clamp: `nref_eff=(nkalt<25)+(nkpsf<25)`, `kk_use = kprop < nref_eff` (6357–6358) — a **feasibility projection** (k ≤ available modes) that is applied to the reducer call but *not written back into X* (X stored at 6190 before the clamp). Python should record the projected vector.
- Per-eval telemetry `mwc_cbright` (229–252): c's brightness estimated from the fakes — instrument/science-specific, not core.

**Injection** — PSF source is reducer-internal: `rnc:1544-1575` reads the WebbPSF/MASK335R off-axis PSF cube (`MWC758_PSFs_MASK335R_F430M_2024-03-02.fits`, 0.1″-spaced slices), **linearly interpolates the slice at the injection separation** (1553–1556), congrids 641→320 px, normalizes to the on-axis synthetic PSF total, then adds `fshift(psf*contrast, ρcos(θ−parang), ρsin(θ−parang))` to each frame (1611–1623). Coronagraph throughput is thus baked into the off-axis PSF; the optimizer's `mwcm_thru` returns unity (1081–1086). PA convention: optimizer works in PA E-of-N; reducer wants radians `(PA+90)*!DTOR` (880, 890). `mwcm_randpos` (4607–4656): **fixed radius** 0.607″ (c's separation), random PA per source rejecting ±25° of the N/S axis (`ns_hw`), known sources (1.5 FWHM), and <3 FWHM from each other; deterministic radial ladder when a band is given. 3 sources per eval (6248–6249), fresh every eval (anti speckle-gardening).

**Single vs multi night** — single dataset: `seq_range=[1,1]`, `nnights=1`, `opt_nights=0`, `pernight` inert (4820–4831). All NEAR multi-night code (`mwcm_reduce_pn`, `pndecode`, `pntie`, `nightsel`) is still present but dead. Two *rolls* exist inside the reducer, combined internally; they are not exposed as partitions.

**Annuli** — `annular=0, nann=1` hard-forced (5078); `ann_edges` default `[0,14]` (5066) is not used for a sweep. Zone is the fixed `[inrad=2,outrad=15]` px annulus spanning c at ~9.6 px (0.607″/0.063).

**Calibration** (5529–5753) — contrast targeted so the *default* config (def_k_klip=25, median, klipfilter 5, 25/25) gives median clean-subtracted S/N in [4,6]: up to 8 trials, `fac=5/S/N` clipped ×0.1..×10 (5732–5734), `cmax_cal` ceiling only if passed (5737), `use_contrast[ia]>0` bypasses (5536, 5569). Best k is *not* scanned here (`kbest_c = def_k_klip−1`, 5610). Two fresh-position honest re-measures per trial (5633–5684), score = median of the two. Plus the **re-calibration revisit** at `it==recal_check=10` (6478–6499): median of top-5 warm-up scores outside [4,6] → rescale contrast toward 5 and `goto RECAL_RESTART` (up to `recal_budget=4` restarts; disabled for forced contrast). Eval 0 is seeded with the default config (6070–6093).

**Validation** (7398–7733) — pool = all finite trials incl. warm-up/seed (7404); top `n_top=3` *distinct* (L1 < 1e-4, 7412) re-scored on `n_valid=8` fresh random positions; one clean reduction per candidate at its fixed k (7497) + `n_valid` injected; **committed image = the median-scoring trial** (7674–7688) so image and reported score agree; winner = max median. Per-candidate decode republishes `kn_*` (7451, changelog A3). Winner knobs stored as `vw_*` (7720–7727) and republished before the final reduction (7810, changelog A4).

**TPE core** — `mwcm_bw/kde/propose/kgrid/ksnap` (1755–1858) are identical to NEAR incl. `pbest`. Loop at 6063–6181: seed eval 0 → `grid`/`random` benchmark modes (6094–6118: grid over bin/filter/n_ang only — meaningless for MWC's space, inherited) → warm-up uniform + kgrid draw → `p_local` local move (one dim of a top-`n_elite`, 6132–6156) → `explore_frac` → `mwcm_propose(..., pbest=seed_best_frac)` + ksnap. Defaults identical to NEAR: `n_init=40, n_iter=200, gamma=.25, ncand=48, explore=.15, p_local=.15, n_elite=5, seed_best_frac=0` (pernight off). `search_k=1` default (scan_mode off, 4773–4774). Checkpoint every eval (`ckpt` struct 7340–7371, atomic tmp+move) incl. the `cfg` struct of all keywords (5377–5395) so `resume=` takes no other keywords (4685–4706); `extend_ann` reopens annuli (4672–4683).

### 1.2 LMIR — LBT/LMIRCam L′, four nights × two LBT sides, ADI

**Instrument constants** — `lmir:1837-1842`: `platescale=0.010604` (SX dewarp scale), `lambda=3.8E-6`, D=8.4 (hard-coded in the fwhm formula), `fwhm=1.028*(lambda/8.4)*206265/platescale ≈ 9 px`. `snr_excl=1.0` (not 1.5) because 1.5-FWHM zones at a 9-px FWHM blanket the inner rings (1810–1814). Matched filter = FWHM Gaussian (`lmm_mfkern`, 908–915). Reducer side: `rlm:405 obs_wl=3.8`, `rlm:317 platescale=0.010604`.

**Data layout** — `nights=['RXJ0534','RXJ0534_2','RXJ0534_3','RXJ0534_4']`, `sides=['left','right']` → `nds=8` flat (night, side) **datasets** with labels `n1L…n4R` (1820–1835). Each reduce_lmircam call handles one dataset and internally averages two nod positions (`cube1/cube2`, `rlm:5661-5662, 6447`), writing `/Volumes/RAID36TB/LMIRCam/<night>/reduced/<night>_<side>_klip[_inj].fits` (`rlm:6471`), `_klip_nosub.fits` (clean only, 6472–6473), `_klip_fm.fits` (6479), and `texp_<side>.sav` (6457). The optimizer averages datasets itself with `lmm_avg` (512–541), **weighted by sqrt(t_exp)**, NaN-aware. The KLIP zone written back is NaN'd outside `annmode_inout` (`rlm:6464-6468`).

**Searched parameters** (`lmir:1926-1955`):

| name | lo | hi | int | notes |
|---|---|---|---|---|
| `bin` | 1 | 20 | 1 | |
| `n_ang` | 1 | 8 | 1 | |
| `klip_filter` | 3 | 40 | 1 | maps to both `klip_filter` and `second_klip_filter` (`rlm:2537-2540`) |
| `angsep` | 0.3 | 3.0 | 0 | only without `/fast` (default: searched → slow per-frame KLIP, 1972) |
| `anglemax` | 20 | 360 | 1 | idem |
| `width` | width_range | | 1 | only `/opt_width` adaptive-annulus mode |
| `n1L … n4R` | 0 | 1 | 1 | one **binary include flag per dataset** (legacy NEAR opt_nights=2 style); `opt_nights=1` default |

**k_klip is NOT searched**: every reduction runs `klip_scan` over 1..`k_klip_max=15` (1786) and the eval picks the argmax-k slice (2310–2330), then does an **honest re-score** at that fixed k on fresh positions (2337–2368) which becomes the eval's score. `wr` fixed (annmode ignores it, 1922–1925); `spat_mean/temp_mean/do_destripe` fixed (0/0/1, 1969). No kgrid, no categoricals (`comb_type` is fixed 'median' in the reducer, `o_comb_type` exists but unused).

**Reduce-wrapper** — `lmm_cmd(nn, sd, inject, kklip, kscan, bn, na, fl, wrr, sm, tm, dd, asep, amax, fst, annin, tpa, trho, tcon, fm_rho=, fm_theta=, fm_contrast=)` (413–441) builds a **command string** `"reduce_lmircam, '<night>', lbtside='<side>', /klip, /oneside, noverify=1, o_k_klip=, o_klip_scan=, o_bin=, o_n_ang=, o_klip_filter=, o_wr=, o_spat_mean=, o_temp_mean=, o_do_destripe=, o_angsep=, o_anglemax=, o_fast=, o_hyp=, o_annmode=, o_annio=[in,out], o_lean=1 [, inj_pa=[...], inj_rho=, inj_contrast=]"`; `lmm_reduce_eval(bridges, nbr, nn_list, sd_list, ..., which, img_noinj, img_inj)` (553–604) fans out dataset×{clean,inj} jobs either serially via `execute` or over an `IDL_IDLBridge` pool, deletes stale outputs first (578), and returns the averaged clean/injected images (or scan cubes). `which`: 0=both, 1=clean only, 2=inj only. Zone padded by 2 px each side (561). Reducer overrides at `rlm:2534-2554`; injection block `rlm:2558-2571` converts sky PA → `planet_theta=(PA+90−truenorth)*!DTOR` (truenorth resolved inside the reducer, so the optimizer passes plain PAs).

**Metric** — `lmir_snrpk` (926–987): FWHM-Gaussian matched filter, peak within `snr_srch=1.5`, ring excluding injected + known sources; fallback to full ring if known-source exclusion leaves <4; penalty `sqrt(1+1/n_elements(use))`; needs `nv>=3`. No declip, no radial-band floor, no disk mask. `lmir_snr_inj` (116–165) for `use_noinj_noise` (default 0). **No clean_subtract option** (the file predates it). Score = `lmm_med` of honest per-source S/N (2361). One documented quirk: at `it == n_init` the running best is **reset** so the "best" is tracked from the TPE phase only (2386–2394) — display-motivated, but validation pool also prefers TPE-phase evals (2775–2776 falls back to all only if none). Python should not reproduce the reset; use the full history and let validation choose.

**Injection** — reducer-internal, PSF template = the per-nod unsaturated median PSF `<night>_<side>_cube<j>_align_cen_clean_median_PSF.fits` (`rlm:5693-5700`), scaled by `refscale=1`, `fshift` per frame with `(θ − parang)` (5745–5763). No throughput model (direct imaging). `lmm_randpos` (452–505): random anchor (ρ,θ) then **even spreading** in both ρ (steps (rhi−rlo)/n) and θ (360/n), redraw anchor up to 30× to avoid known sources; floor `min_inj_fwhm=2.0` FWHM from the star (462–464). 3 sources in the innermost annulus, 6 outside (2082). Fresh positions every eval (2274–2281). `cal_bin=8` heavier binning **only during calibration** (1797, 2105/2122/2158).

**Single vs multi** — multi-dataset (8) with binary selection; **no per-dataset parameter blocks** (one global block). All-excluded proposals are forced to all-on and written back into X (2241–2243).

**Annuli** — full NEAR machinery: fixed `ann_edges` or `/opt_width` greedy build-out with searched `width` (1868–1908, 1939–1945); non-annular default zone `annmode_inout=[0, sep_px+2 FWHM]` (1845–1846). Sources inset by 1 FWHM from the edges (2077–2078). Guard that `ann_edges` is strictly increasing (1879–1895).

**Calibration** (2093–2208) — same 4–6 S/N loop as NEAR: clean+inj scan at `kcal=min(k_klip_max,12)` with `cal_bin`, pick kbest once (2139), two honest fresh-position re-measures per trial (2146–2166), `cmax_cal=5e-2` ceiling always on (1795, 2194), contrast **warm-started from the previous annulus** (`c_seed`, 1853, 2097, 2208). No recal-revisit, no `use_contrast` forcing.

**Validation** (2770–2960) — as NEAR/MWC: top-3 distinct, 8 fresh trials, raw `lmir_snrpk`, one clean reduction per candidate at its Kbest; winner = max median.

**TPE core** — `lmm_bw/kde/propose` (338–405): textbook version **without `pbest`**, and the loop (2226–2233) has **no `p_local`, no `explore_frac`, no kgrid, no seed-eval-0, no grid/random benchmark modes, no checkpoint/resume, no run_setup.txt** (only `runinfo.txt`, 2039–2051). Defaults `n_init=8, n_iter=14` (smoke-test values, 1781–1782), `gamma=.25, ncand=48, n_valid=8, n_top=3`. Results log `optimize_tpe_results.txt` is written to the shared root and **overwritten each run** (2035) — one of the "accumulating file" hazards.

### 1.3 HPB — LBTI Fizeau M-band, single night, ADI, batched parallel TPE

**Instrument constants** — `hpb:1390-1393, 1455`: `obs_wl=4.77 µm`, `diam=28.8 m` (Fizeau baseline, used for KLIP λ/D and the deconvolution kernel), `ap_diam=8.4 m` for the **detection aperture** FWHM `fwhm=1.028*(obs_wl e-6/ap_diam)*206265/platescale`, `platescale=0.0107`, `truenorth=0`. Two "FWHM"s therefore coexist: the metric aperture (8.4 m) and the reducer's physics (28.8 m; `rhp:318`). Matched filter = FWHM Gaussian (`hpb_mfkern`, 91–97); the earlier empirical asymmetric FM template was removed because `convol` flips the kernel and produced negative S/N for bright sources (comment 130–134, 1671–1677).

**Data layout** — one night; reducer products under `/Volumes/RAID36TB/LMIRCam/HPBoo/reduced/`: `HPBoo_cube_align_cen.fits` (+ `_lastread` master), `HPBoo_angles.sav`, `HPBoo_frametags.sav` (corrs, noises), `HPBoo_psf_inject.fits`; outputs `HPBoo_klip[_deconv][_inj][<out_suffix>].fits`. `reduce_hpboo` keeps the centred cube in `common hpboo_cache` (`rhp:344-367`) so `/reuse` calls skip I/O. `hpb_setcrop` (1206) auto-crops the working cube to the search radius per annulus.

**Searched parameters** (`hpb:1483-1521`):

| name | lo | hi | int | notes |
|---|---|---|---|---|
| `k_klip` | 1 | 20 | 1 | **uniform draw, no kgrid** |
| `bin` | 10 | 25 | 1 | |
| `n_ang` | 1 | 8 | 1 | |
| `filter` | max(ceil(1.5 fwhm),3) | 50 | 1 | |
| `inrad` | 0 | ρ_px−2 FWHM | 1 | pinned to annulus edge when `ann_edges` given (1630) |
| `outrad` | ρ_px+2 FWHM | min(data, ρ_hi+2 FWHM) | 1 | idem; `hpb_decode` guards `orad<=ir+2 → ir+2 fwhm` (604) |
| `anglemax` | 20 | field-rotation span | 1 | |
| `angsep` | 0 | 3 | 0 | only when `fast=0` (default slow, 1420) |
| `corr_thresh` | min(corrs) | 75th pct(corrs) | 0 | **data-derived bounds** from frametags (1508–1515); it is an ABSOLUTE CC cut, not 0–1 |
| `noise_max` | 0.3 | 3.0 | 0 | multiple of mean noise |

`opt_framesel=1` default. No categoricals (`comb_type='nwadi'` fixed in reducer, `rhp:303`). `usedeconv` selects Wiener-deconvolved products (`rhp:18-31`, 805–812) — a boolean run option, not searched.

**Reduce-wrapper** — `hpb_reduce(kk, bn, na, fl, ir, orad, asep, amax, fastf, cthr, nmax, irho, ith, ic, reddir, obs_wl, diam, platescale, truenorth, opt_framesel, usedeconv, dreg)` (253–271) → `reduce_hpboo, /reuse, adi=0, /klip, fast=, k_klip=, bin=, n_ang=, filt_klip=, annmode_inout=[ir,orad], angsep=, anglemax=, inj_rho=, inj_theta=, inj_contrast=, [corr_thresh=, noise_max=], deconv=, dreg=, outdir=, obs_wl=, diam=, platescale=, truenorth=, quiet=1`, reads `HPBoo_klip[_deconv]_inj.fits`. Siblings: `hpb_reduce_noinj` (278), `hpb_scan` (`kscan=1, kkmax=`, 300), `hpb_fm` (323). For the pool, `hpb_cmd` (409–427) serializes the same call to a string with `out_suffix='_jN'`. Reducer signature `rhp:175-191`; frame selection `rhp:662-674`; injection `rhp:694-711`.

**Metric** — `hpb_snrpk` (121–191): Gaussian MF, peak search, ring excluding all injected sources with a **fallback to excluding only this source** if `nv<3` (172–182), penalty `sqrt(1+1/nv)`, returns a per-source `reason` code (1 inner radius, 2 starved ring, 3 flat ring) used for diagnostics during calibration (1727–1730). **No known-source mask, no clean_subtract, no difference metric.** Angle convention: **math angle, no +90** (138–143) because reduce_hpboo injects with `cos(θ−parang)` directly (`rhp:704-705`). Score = `hpb_med` over `nsrc=4` sources (1848–1849). Image is `radprof(img*1.0)`.

**Injection** — reducer-internal; template = first-read (unsaturated) median PSF × exposure-time ratio (`rhp:601-608, 689-708`), `fshift` per frame. No throughput. `hpb_randpos` (62–85): NEAR-style spread in **both ρ and θ** across `rho_range=[0.5ρ,2ρ]` (1400) intersected with the annulus inset by 1 FWHM; θ anchor random unless `theta>=0` given. `nsrc` shrinks in tight annuli (1633–1637: 1 if outer <0.4″, 3 if <0.5″).

**Single vs multi** — single night; `opt_nights` accepted but ignored (1469).

**Annuli** — optional fixed `ann_edges`; when annular, `inrad/outrad` bounds are **collapsed to the annulus edges** (1630) rather than removed from the vector (so two dead dims stay in the TPE space — bandwidth floor `0.08*(hi−lo)=0` → `hpb_bw` returns 0 for n<2... fine, but a Python SearchSpace should drop pinned dims). Stitch by NaN-average (`hpb_stitch`, 895).

**Calibration** (1662–1734) — one reduction per trial at the default config with `cal_bin=64` (1403), median S/N target [4,6], `fac=5/S/N` ×0.1..×10, `cmax=5e-2` ceiling; **no honest re-measure, no k scan** (k fixed at `def_k=4`). NaN trials retry with fresh positions.

**Search loop — batched** (1789–1983): proposes `nb=min(nworkers, remaining)` candidates **from the pre-batch history** (1797–1835; the whole batch shares the same posterior — no fantasized/liar updates), dispatches them to a persistent `hpb_worker` pool via job files + done-flags (`hpb_pool_init/run_batch`, 480–579, with dead-process respawn and optional kill watchdog), then scores. Proposal rule (1817): random if `iter<n_init` **or `it<2`** or `randomu<explore_frac`; else `hpb_propose`. No `pbest`, no `p_local`, no kgrid, no local moves, eval 0 seeded with defaults (1800–1816). Defaults `n_init=20, n_iter=120, gamma=.25, ncand=48, explore=.15, n_valid=5, n_top=3`. **No checkpoint/resume.** Per-run `run_setup.txt` + script archive (1555–1621) like NEAR.

**Validation** (1990–2026) — top `n_top` by score (**no distinctness filter**), `n_valid=5` fresh position sets each, raw `hpb_snrpk` median, winner = max median; the winner is then re-reduced once more at fresh positions for display (2018–2024). No clean reduction in the loop (only for display at best-updates, 1862).

---

## 2. Cross-variant diff table

| Aspect | NEAR | MWC | LMIR | HPB | Core / specific |
|---|---|---|---|---|---|
| λ, D, pxscale | 11.0 µm, 8.2 m, 0.0456″ | 4.30 µm, 6.5 m, 0.063″ | 3.8 µm, 8.4 m, 0.010604″ | 4.77 µm, 28.8 m (KLIP) / 8.4 m (aperture), 0.0107″ | instrument config |
| FWHM (px) formula | `1.028 λ/D·206265/px` | same (≈2.2) | same (≈9) | same but with `ap_diam` (≈12) | core helper; allow override |
| Matched filter | FWHM Gaussian (or measured N4 PSF) | fixed 2.0-px Gaussian | FWHM Gaussian | FWHM Gaussian | `Metric` option: kernel = gaussian(fwhm_mf) or user kernel |
| Star centre | `(n−1)/2` | `(n−1)/2` (5599, 6370) | `(n−1)/2` (2317) | `(imnx−1)/2` (1667) | core convention |
| Angle convention into metric | PA E-of-N, `+90°` | PA, `+90°` | PA, `+90°` (truenorth inside reducer) | **math angle, no +90** | `Reducer`/`InjectionModel` declare the convention; metric consumes pixel (x,y) |
| Mode | ADI (multi-night) | **ARDI** (2 rolls + RDI library, per-target refsel) | ADI (per dataset) | ADI | mode is reducer-internal; optimizer only needs param names |
| Partitions | 6 nights, per-night blocks + drop1/drop2 | 1 dataset (2 rolls hidden) | 8 (night×side) datasets, binary flags, one global block | 1 | `SearchSpace` groups + selection dims |
| Categorical dims | none | `combtype` ∈ {mean, median, nwadi} as int index | none | none | `SearchSpace` categorical kind |
| Integer grid | kgrid for k_klip | kgrid | k not searched (scan) | uniform k | custom sampling grid per param |
| k_klip handling | searched (kgrid) or scan_mode | searched, clamped to `nref_eff` | **scan + argmax + honest re-score** | searched | two strategies: `search` vs `scan+rescore` |
| Feasibility projection | angsep/anglemax ref-count guardrail | k ≤ available refs | none | `outrad>inrad+2` guard | user hook `project(x)` |
| Frame-quality dims | corr_thresh, noise_max, coronoise_max | none | none | corr_thresh (data-derived bounds), noise_max | reducer-defined params |
| Metric | Mawet MF peak (`near2_snrpk`) | same + declip + band floor + disk/known mask + strict nret | same, known mask, no floor | same, no mask, exclude-self fallback, `reason` codes | one `MawetPeakSNR` with options: declip, sigma_floor, masks, penalty mode |
| clean_subtract | on (prod) | on (default) | absent | absent | `Metric.corrected` option with clamp |
| use_noinj_noise | opt | opt (off) | opt (off) | absent | `Metric` alt implementation |
| Aggregate | median (mean if N=2) | median | median | median | core |
| Injection PSF | measured AGPM N4 library + throughput | WebbPSF off-axis cube interpolated in ρ | unsaturated median PSF | first-read median PSF × texp ratio | `InjectionModel` inside reducer for all 4 — optimizer only passes (ρ,θ,c) |
| Throughput model | measured curve | unity | unity | unity | `InjectionModel.throughput(r)` default 1 |
| Position sampler | ρ-θ spread, per-eval fresh | fixed ρ, random θ with axis/known/mutual exclusions | ρ-θ even spread + anchor redraw | ρ-θ spread, θ anchor optional | `PositionSampler` strategy with exclusion zones |
| # sources / eval | 4 inner / 6 outer | 3 | 3 inner / 6 outer | 4 (1–3 in tight annuli) | per-annulus rule, configurable |
| Calibration | 4–6 S/N, scan-k once + 2 honest re-measures, forced `use_contrast`, `cmax_cal` opt, recal-revisit | same (k fixed at default) | same, `cal_bin`, `c_seed` warm-start, cmax always | single reduction/trial, `cal_bin=64`, cmax always, no re-measure | `Calibrator` with options |
| Warm-up | uniform + kgrid; pntie warmstart | same | uniform | uniform | core |
| TPE extras | pbest, p_local/n_elite, explore | same | none | explore only | core optimizer options (default on) |
| Batched proposals | no (serial + per-night parallel inside reducer) | no | no (parallel inside eval) | **yes**, nb per batch from shared history | `Runner`/`Optimizer.ask(n)` |
| Benchmark modes | tpe/grid/random | inherited | none | none | `RandomSearch`, `GridSearch` |
| Validation | top-3 distinct × 8 fresh, raw metric, median-trial commit | same | same (TPE-phase pool first) | top-3 (no distinct) × 5 | core `Validator` |
| Checkpoint/resume | every eval, cfg in ckpt, `extend_ann` | same | none | none | core `Runner` |
| Provenance | run_setup.txt + scripts copy | same | runinfo.txt only | run_setup.txt + scripts | core |
| Parallelism | spawned idl night-workers | spawned idl for clean+inj | IDL_IDLBridge per dataset | spawned worker pool + job files | leave to `Reducer`/executor |

**Core (identical in all four, port once):** `bw` (Scott/Silverman ×1.06, floor 0.08·range), `kde` (Gaussian Parzen + 0.25 uniform prior), `propose` (γ split, per-dim jitter from a random good obs, separable log l/g argmax), warm-up, per-source Mawet MF-peak S/N + median aggregate, contrast calibration toward S/N 5 with ×0.1..×10 steps, top-N re-validation on fresh injections with the winner taken from validated scores, per-annulus loop, `(n−1)/2` centre, `radprof` flattening before scoring.

---

## 3. Recommendations for the Python abstraction seams

### `SearchSpace`
- `Param(name, lo, hi, kind ∈ {float, int, categorical}, grid=None, choices=None, default=…)`. `kind='int'` → round after clamp; `grid` (e.g. `kgrid(kmax)`, `mwc:1839-1849`) → warm-up/explore draw uniformly over grid entries, TPE proposals snapped to the nearest entry (`mwcm_ksnap`, 1855). Grid should be a generic list, not the k-specific 1/5/10 stepping.
- **Categoricals**: encode each categorical as its own unit-simplex/one-hot block *or*, minimally, as an index dim whose KDE is replaced by a categorical Parzen (counts + Laplace/uniform prior) — i.e. `l_d(x)=(n_good(c)+α)/(n_good+Kα)`. This is what Optuna's TPE does and removes the spurious ordinal adjacency in MWC's `round(x)` index trick (`mwc:6215`). Keep a decode hook `to_reducer_kwargs(x) -> dict` so `combtype` becomes the string.
- **Groups / partitions**: `SearchSpace.replicate(block, partitions=['n1',…])` producing `name_<p>` dims plus `tie(names)` (= `mwcm_pntie`, 1239–1253) for linked params and for the global-tied warm start; `dims_of(partition)` (= `mwcm_pndims`, 1261). Partitions are generic: nights (NEAR), night×side datasets (LMIR), rolls/filters/detectors later. LMIR shows the global-block + selection-flags case must also work (block replicated with `partitions=None`).
- **Selection dims**: `SelectionScheme ∈ {none, binary_per_partition, two_slot_jackknife(max_drop=2)}` with `decode → mask` (`lmm_nightsel`, 1223–1236; `mwcm_nightsel`, 2524) and the "never drop everything" rule; `p_include` bias for random draws (`night_pinc`, `mwc:6129-6130`); write the sanitized mask back into X (LMIR does, `lmir:2242`).
- **Pinned dims**: when an annulus pins `inrad/outrad` (HPB 1630) drop them from the searched vector rather than setting lo=hi.
- **Data-derived bounds hook** (HPB corr_thresh percentiles, 1508–1515; `amax_hi` from field rotation in all three) → `Param.bounds_from(data_summary)` callback evaluated once at run start and recorded in `run_setup`.
- **Feasibility projection** `project(x) -> x'` applied on *every* proposal path *and* recorded into the history (MWC's k clamp at 6358 is not written back — don't copy that). Cover NEAR's ref-count guardrail, MWC's `k ≤ n_ref(nkalt,nkpsf)`, HPB's `outrad>inrad+2`.

### `Reducer`
- `reduce(params: dict, injections: list[(rho, theta, contrast)] | None, partition_ids: list | None, tag: str) -> Result(image, per_partition_images=None, kcube=None)`.
- Must declare: `angle_convention` ('pa_east_of_north' vs 'math'; HPB differs), `center` convention (default `(n−1)/2`), `supports_kscan` (LMIR/HPB/NEAR: returns `[nx,ny,k]` cube; MWC: no), `param_schema` (names → kwarg names + fixed defaults such as MWC `inrad=2,outrad=15,n_ang=1`, LMIR `wr, spat_mean…`), and optional `available_modes(params)` used by the projection (MWC `nref_eff`).
- Per-partition reduction + combine belongs in a `PartitionedReducer` wrapper: runs the same params (or per-partition param dicts) for each selected partition and combines with `weights ∈ {sqrt_texp (lmm_avg/mwcm_avg), inverse_variance (mwcm_pnweights), equal}`; returns the partition stack for per-partition diagnostics. This single wrapper covers NEAR nights, LMIR night×side, and future rolls.
- Output isolation `tag`/`out_suffix` so concurrent clean+inj or batched jobs never collide (every variant re-invented this: `mwc:906`, `rlm:2570`, `rhp:711`); the wrapper should delete stale outputs before dispatch (`lmir:578`, `hpb:514`).
- Parallel execution = executor injected into the Runner (`concurrent.futures`), with a per-job timeout/retry policy (HPB `job_timeout`, MWC per-night retry-once then score invalid, `mwc:1043-1067`). An invalid/failed evaluation scores `-inf`/sentinel, never a stale image.

### `InjectionModel` / `PositionSampler`
- In all four the PSF template and its per-frame placement live inside the reducer; the optimizer only supplies `(rho_arcsec, theta_deg, contrast)`. So `InjectionModel` in Python has two roles: (a) for the reference `KLIPReducer`, `psf_at(r) -> 2-D template` + `throughput(r)` (default 1; NEAR N4 and MWC ρ-interpolated WebbPSF cube are two implementations), (b) for wrapped external pipelines, a no-op that just forwards the triples.
- `PositionSampler.sample(n, r_lo, r_hi, rng, exclusions)` with strategies `fixed_radius_random_pa` (MWC) and `spread_rho_theta` (NEAR/LMIR/HPB), common exclusion rules: min radius (`1.5–2 FWHM`), known sources (r_excl), mutual separation (2–3 FWHM), forbidden PA sectors (MWC N/S ±25°). Fresh draw every evaluation, calibration trial and validation trial (all four do this).

### `Metric`
- `MawetPeakSNR(fwhm, kernel='gaussian'|array, mf_fwhm=None, search_px=1.5, excl_fwhm=1.5, declip_nsig=None|6, sigma_floor_band=None|0.6, penalty='retained'|'geometric', known_sources=[], pixel_mask=None, min_ring=3|6, self_only_fallback=False)` returns per-source S/N (NaN allowed, negatives allowed). The four `*_snrpk` are this one function with different flags: MWC = declip+band floor+retained penalty+masks; LMIR = known mask only; HPB = self-only fallback + reason codes; NEAR = band floor + masks.
- `InjectionDifferenceSNR` (= `*_snr_inj`) as the alternate implementation.
- `Objective(metric, mode ∈ {raw, clean_subtract}, aggregate ∈ {median, mean_if_2})` with `clean_subtract: s = s_inj − max(s_clean, 0)` and NaN propagation (`mwc_clip0`, 455). Flatten with `radprof` (azimuthal median) before scoring in all modes — make it an explicit `preprocess` step.
- Keep `score_search` and `score_validation` as separate entry points; validation is always raw.

### `Optimizer`
- `TPE(gamma=.25, ncand=48, n_init, prior_weight=.25, bw_floor=.08, pbest=0|.5, p_local=.15, n_elite=5, explore_frac=.15)` with `ask(n=1)` (HPB batching: all `n` from the same history) and `tell(x, y, flags)`. Flags `{warmup, explore, local, seed}` recorded per trial (`expl_hist/local_hist`, `mwc:5808-5809`).
- Eval 0 = seeded default config (NEAR/MWC/HPB); make `seed_points=[...]` a generic option.
- `RandomSearch`, `GridSearch(axes=…)` behind the same `ask/tell` (the MWC/NEAR grid is hard-wired to bin/filter/n_ang — generalize to a user-chosen axis subset at matched budget, `mwc:6094-6118`).
- Handle categorical dims per §SearchSpace; local move picks one *reduction* dim (not a selection dim, `mwc:6152`).

### `Runner`
- Protocol per annulus: `calibrate → (seed) → search → recal_revisit(optional) → validate → products`, each stage writing rows to one append-only `results.txt` with `run_tag`/`annulus`/`phase` columns and a JSON checkpoint **every evaluation** containing config + RNG state + history (NEAR/MWC only had this; LMIR/HPB have nothing — Python must have it everywhere). `resume(run_dir)` takes no other options.
- `Calibrator(target=(4,6), aim=5, max_trials=8, step_clip=(0.1,10), ceiling=None, forced=None, n_remeasure=2, cal_overrides={'bin': 64}, warm_start_from_previous_annulus=True, recal_check=10, recal_ntop=5, recal_budget=4)` — union of all four behaviours; the honest re-measure at fixed k and recal-revisit should default on.
- `Validator(n_top=3, n_valid=8, distinct_tol=1e-4, pool='all', commit='median_trial')`.
- Annulus handling: `AnnulusPlan(fixed_edges | adaptive(width_range, r_cap) | single_zone)`; per-annulus `n_init/n_iter` vectors (`mwc:4789`), n_sources rule by annulus radius, injection band inset by 1 FWHM, adaptive-width sources re-centred per eval (`lmir:2273-2277`).
- k strategy flag: `k_mode ∈ {'search', 'scan_rescore'}`. In `scan_rescore` the eval costs a scan + one honest fixed-k injected reduction (LMIR 2298–2368); in `search` k is a normal dim (MWC/HPB). Both must exist because reducers without a k-cube output (MWC) can only do `search`.
- ADI/RDI/ARDI: no optimizer-level flag is needed — MWC shows mode selection is a reducer parameter set (`refsel`, `nkalt`, `nkpsf`, `/ardiklip`). Expose it only as `Reducer.mode` metadata for provenance and let `SearchSpace` carry mode-specific params (e.g. reference-selection counts) with a projection tying `k ≤ n_refs`.
- Where per-night blocks generalize: NEAR per-night blocks + drop slots → `partitions` of any kind; LMIR shows partition selection without per-partition blocks; MWC's two rolls and any multi-filter/multi-detector case are new partition axes. The `PartitionedReducer` + `SearchSpace.replicate/tie/selection` pair covers all of them; the reducer only ever sees "params for partition p" and "which partitions to include".

---

## 4. Bug fixes and lessons from the changelog/README that must carry into Python

From `CHANGELOG_optimize_mwc_tpe_2026-07-14.md`:
1. **A3 — stale knobs in validation**: candidates were validated with the *last search eval's* combtype/klipfilter/nkalt/nkpsf because those lived in a `common` block published only in the main loop. Python: decode is a pure function `x -> params`, and the reducer receives the full params dict on every call (no ambient state). This bug also motivates the regression test "replay logged X rows through decode+reducer kwargs and compare".
2. **A4 — final stage used wrong inner radius + stale knobs**: winner record must be a complete, self-contained object (`vw_*` incl. `vw_ir`, `mwc:7720-7727`), stored in the checkpoint and used for every downstream product; `final_setup.txt` must list *all* params.
3. **A1 — matched-filter kernel** is instrument-specific and changes every score; make it an explicit `Metric` option written to `run_setup`, and any kernel change invalidates comparisons across runs (a run started before the change is not self-consistent).
4. **A2 — small-sample penalty on retained apertures** (`nret`), never on geometric `nap` nor on correlated band-pixel counts; heavy masking near the IWA must *raise* the penalty.
5. **C8 — variables defined only inside a gated block** (`names_aug/lo_aug/hi_aug` → `OPT_LO` crash) and **C10 forward_function** ordering: vanish in Python but the lesson is "build all derived config once, unconditionally, before the loop".
6. **C7/C9 — resume correctness for the best marker/panels**: best-of-history must be recomputed from the restored history (`argmax` of shown Y), not from a cached index; best-config re-reductions after resume must use the stored winner config (`best_ir/best_comb/...`, guarded by a checkpoint version tag `BIR`). Python: version the checkpoint schema.
7. **B5 — `final_fullframe.fits`**: keep a "winner config over a wide zone" product hook, distinct from the scored annulus crop.

From `optimize_mwc_tpe_README.txt`:
8. **Injection PA convention is the first thing to break in a port** (rough edge 2: `+90°` in `mwcm_reduce`; HPB uses math angle). The smoke test must assert injected sources are *recovered* (median S/N ≫ 0) before any optimization; Python should ship `check_injection_recovery()` and have the reducer declare its angle convention.
9. Metric choice depends on the field: bright disk → prefer the difference metric or clean_subtract (rough edge 3). Keep both metrics selectable per run and logged.
10. Cost model: full-library fast RDI ~4 s vs per-target selection 10–30 s/eval; the search space itself changes per-eval cost — log wall time per eval (all four print ETAs; make it a column).

From `klip_best_params_run_20260620_082516.md`:
11. The "current best" note was written at **eval 29 of 500 from the search trace** (raw metric, no validation) — exactly the upward-biased number §4 warns about. Python's `best()` API should return the *validated* record and label unvalidated search maxima as such.
12. A best that "sits inside the search space (no edge/ceiling pinning)" is a useful health check — report per-param distance-to-bound of the winner.

Other cross-variant lessons visible in the code:
13. `IDL 'and' is not short-circuit` workarounds (`cmax_cal=Inf`, `mwc:4842`; `resume_skip` guard, 5523) — don't port; but do keep the semantics: ceiling **off unless given** (NEAR/MWC) vs always on (LMIR/HPB) should be one explicit option with a documented default.
14. Concurrency via files (done-flags, `out_suffix`) reappears in all four; whatever executor is used, outputs must be keyed by a unique job tag and stale outputs deleted before dispatch.
15. Accumulating shared logs: LMIR overwrites `optimize_tpe_results.txt` in a shared root (2035); MWC copies a shared log into the run dir on resume (5239–5248). Python: one log per run directory, append-only, with `run_tag` in every row.
16. LMIR's reset of the running best at `it==n_init` (2386–2394) and HPB's missing distinctness filter (1996–1997) are regressions relative to NEAR; adopt NEAR/MWC behaviour.
17. `k_eff` clamp not written back to history (MWC 6357–6358): projected values must be what is logged, otherwise replay/regression tests disagree with what was reduced.
