# Command-line reference

Generated from `klip-tpe <command> --help` (`python scripts/gen_cli_doc.py`).  `near` and
`generic` share the same search / protocol arguments; `--instrument near | nomic | generic`
selects the data adapter (`klip-tpe generic` is `near --instrument generic`).  `resume` and
`extend` take the same data arguments as the run they continue.


## `klip-tpe generic`

```
usage: klip-tpe generic [-h] [--instrument {near,nomic,generic}]
                        [--root ROOT [ROOT ...]] [--cube CUBE [CUBE ...]]
                        [--angles ANGLES [ANGLES ...]] [--psf PSF [PSF ...]]
                        [--ref-cube REF_CUBE [REF_CUBE ...]]
                        [--names NAMES [NAMES ...]]
                        [--star-flux STAR_FLUX [STAR_FLUX ...]]
                        [--pxscale PXSCALE] [--lam LAM] [--diam DIAM]
                        [--angle-sign ANGLE_SIGN] [--crop-half CROP_HALF]
                        [--center CENTER CENTER] [--wv-index WV_INDEX]
                        [--no-tags] [--nights NIGHTS [NIGHTS ...]]
                        [--backend {klip,pyklip,vip}] [--obj OBJ [OBJ ...]]
                        [--frames FRAMES] [--binned] [--pre-bin PRE_BIN]
                        [--parang-sign PARANG_SIGN] [--truenorth TRUENORTH]
                        [--fwhm-px FWHM_PX] [--name NAME] [--r-ee R_EE]
                        [--groups GROUPS [GROUPS ...]]
                        [--group-min-frames GROUP_MIN_FRAMES]
                        [--group-smooth GROUP_SMOOTH]
                        [--max-frames MAX_FRAMES] [--workers WORKERS]
                        [--pool {auto,processes,threads}]
                        [--weighting {equal,sqrt_texp}] [--fast]
                        [--no-library] [--no-clean-subtract] [--ladder-pair]
                        [--n-min-ref N_MIN_REF] [--ref-frac REF_FRAC]
                        [--known RHO PA] [--metric {mawet,fmmf}]
                        [--run-dir RUN_DIR]
                        [--ann-edges ANN_EDGES [ANN_EDGES ...]]
                        [--n-iter N_ITER [N_ITER ...]]
                        [--n-init N_INIT [N_INIT ...]]
                        [--mode {tpe,random,grid}] [--blocks BLOCKS]
                        [--seed SEED] [--contrast0 CONTRAST0]
                        [--use-contrast USE_CONTRAST [USE_CONTRAST ...]]
                        [--cmax-cal CMAX_CAL] [--k-max K_MAX] [--n-top N_TOP]
                        [--n-valid N_VALID] [--global-block] [--no-framesel]
                        [--no-selection] [--max-drop MAX_DROP] [--tag TAG]
                        [--from-idl-setup RUN_SETUP_TXT] [--opt-width]
                        [--width-range W0 W1] [--r-cap R_CAP]
                        [--k-mode {search,scan_rescore,scan}]
                        [--k-scan-max K_SCAN_MAX]
                        [--stitch-every STITCH_EVERY] [--verify] [--no-verify]
                        [--verify-n-boot VERIFY_N_BOOT] [--param-verify]
                        [--no-param-verify] [--n-pv N_PV]
                        [--pv-divmin PV_DIVMIN] [--candidates]
                        [--write-setup-files] [--no-setup-files]
                        [--legacy-stitch] [--display] [--no-display]
                        [--display-every DISPLAY_EVERY]
                        [--pdf-every PDF_EVERY] [--movie-every MOVIE_EVERY]
                        [--show [SHOW]] [--aliens]
                        [--window-scale WINDOW_SCALE] [--no-fm-curve]
                        [--fm-preview] [--no-fm-preview]

options:
  -h, --help            show this help message and exit
  --instrument {near,nomic,generic}
                        near: NEAR campaign tree under --root; nomic: pyNOMIC
                        working directory under --root; generic: any
                        registered cube (--cube/--angles/--psf ...)
  --root ROOT [ROOT ...]
                        NEAR: data root. nomic: pyNOMIC working directory (one
                        per --obj, or one for all). generic: optional base
                        directory for the run directory
  --nights NIGHTS [NIGHTS ...]
  --backend {klip,pyklip,vip}
                        PSF-subtraction engine: built-in annular KLIP
                        (default), pyKLIP or VIP (must be installed)
  --max-frames MAX_FRAMES
                        use only the first N frames per night (tests)
  --workers WORKERS     worker budget: 'auto' (default) = all cores, an
                        integer pins it, -k = all but k
  --pool {auto,processes,threads}
                        how the budget is spent: forked worker processes
                        (default; the IDL bridges) or threads
  --weighting {equal,sqrt_texp}
  --fast                single-basis KLIP (angsep/anglemax not searched)
  --no-library
  --no-clean-subtract
  --ladder-pair         two-source annulus: use the pre-2026-09-05 two-rung
                        radial ladder instead of both sources at the annulus'
                        area-weighted mid radius (only to reproduce older IDL
                        runs)
  --n-min-ref N_MIN_REF
  --ref-frac REF_FRAC
  --known RHO PA
  --metric {mawet,fmmf}
                        detection metric: 'mawet' (default) filters with the
                        injection PSF; 'fmmf' forward-models that PSF through
                        each configuration's own subtraction and filters with
                        the result (klip_tpe.fmmf)
  --run-dir RUN_DIR     output directory (default:
                        <root>/comb/opt/run_YYYYMMDD_HHMMSS, the IDL layout;
                        generic: ./runs/run_...)
  --ann-edges ANN_EDGES [ANN_EDGES ...]
  --n-iter N_ITER [N_ITER ...]
  --n-init N_INIT [N_INIT ...]
  --mode {tpe,random,grid}
  --blocks BLOCKS       partitions | univariate | full
  --seed SEED
  --contrast0 CONTRAST0
  --use-contrast USE_CONTRAST [USE_CONTRAST ...]
                        forced contrast per annulus (0 = calibrate)
  --cmax-cal CMAX_CAL
  --k-max K_MAX
  --n-top N_TOP
  --n-valid N_VALID
  --global-block        one parameter block for all nights
  --no-framesel
  --no-selection        no drop1/drop2 night-selection dims
  --max-drop MAX_DROP   drop slots of the partition selection (2 = IDL
                        drop1/drop2; more for nights x groups)
  --tag TAG
  --from-idl-setup RUN_SETUP_TXT
                        mirror an IDL run's settings (annulus, budget,
                        contrast, validation, TPE) for a matched Python run
  --opt-width           adaptive annuli: --ann-edges gives the first inner
                        edge; 'width' becomes a searched dim
  --width-range W0 W1
  --r-cap R_CAP         outer radius cap (px) for --opt-width
  --k-mode {search,scan_rescore,scan}
                        search: k_klip searched; scan/scan_rescore: per-eval
                        k-scan (k not searched)
  --k-scan-max K_SCAN_MAX
  --stitch-every STITCH_EVERY
                        running-stitch cadence (evals); 0 = annulus end only
  --verify              klip_tpe.verify report + limit curve after each
                        annulus
  --no-verify
  --verify-n-boot VERIFY_N_BOOT
  --param-verify        parameter-ensemble persistence stage (default: on when
                        per-night blocks exist)
  --no-param-verify
  --n-pv N_PV
  --pv-divmin PV_DIVMIN
  --candidates          blind candidate search on the running/final stitch
  --write-setup-files
  --no-setup-files
  --legacy-stitch       equal-weight stitch, no contrast-curve seam trim
  --display             live display: per-eval step panels, books, progress
                        movie (default on)
  --no-display
  --display-every DISPLAY_EVERY
                        render cadence in evaluations
  --pdf-every PDF_EVERY
  --movie-every MOVIE_EVERY
                        rebuild annulusNN/progress.gif + .mp4 every N
                        evaluations (default: --pdf-every; 0 = annulus end
                        only)
  --show [SHOW]         live panel on screen: --show (a matplotlib window),
                        --show inline (update a Jupyter output cell in place),
                        --show auto (inline inside a notebook, window
                        otherwise)
  --aliens              play the IDL launch movie during the first calibration
                        (+ intro.gif)
  --window-scale WINDOW_SCALE
                        live window size as a fraction of the 1850x990 panel
                        (1.0 = the IDL window, 1:1 pixels)
  --no-fm-curve         skip the KLIP-FM cross-check curve after each annulus
                        (A 9.1)
  --fm-preview          live KLIP-FM preview at each new best (A 9.3; one
                        extra reduction per new best; default on)
  --no-fm-preview

generic cubes (--instrument generic / klip-tpe generic):
  --cube CUBE [CUBE ...]
                        registered cube(s), (n, ny, nx) FITS; one per
                        partition
  --angles ANGLES [ANGLES ...]
                        derotation angles FITS (one per cube; CCW to North-up)
  --psf PSF [PSF ...]   generic: off-axis PSF template FITS (one, or one per
                        cube); nomic: airy (default) | frame | gaussian
  --ref-cube REF_CUBE [REF_CUBE ...]
                        PSF-reference cube(s) for RDI/ARDI (one per cube)
  --names NAMES [NAMES ...]
                        partition names (default: cube file stems)
  --star-flux STAR_FLUX [STAR_FLUX ...]
                        star flux in science-frame units per cube, or 'halo'
                        (scale the PSF template to the halo)
  --pxscale PXSCALE     arcsec / px
  --lam LAM             wavelength (m)
  --diam DIAM           aperture diameter (m) for lambda/D
  --angle-sign ANGLE_SIGN
                        -1 flips the derotation-angle sign
  --crop-half CROP_HALF
                        crop to 2h+1 px around --center (generic default: full
                        frame; nomic default: 75)
  --center CENTER CENTER
                        star pixel (x y); default: array centre
  --wv-index WV_INDEX   channel of a 4-d IFS cube
  --no-tags             no data-derived frame-quality tags (no frame
                        selection)

pyNOMIC (--instrument nomic):
  --obj OBJ [OBJ ...]   pyNOMIC object name(s) (the <obj>_NOMIC_*.npz prefix)
  --frames FRAMES       frame directory under the working dir (masked |
                        aligned)
  --binned              use pyNOMIC's <obj>_NOMIC_binned_evaluated.npz frames
  --pre-bin PRE_BIN     mean-bin every N frames at load time (memory)
  --parang-sign PARANG_SIGN
                        -1 flips the parallactic-angle sign
  --truenorth TRUENORTH
  --fwhm-px FWHM_PX     override the FWHM (default: Airy fit / 1.028 l/D)
  --name NAME           partition-name prefix (default: --obj)
  --r-ee R_EE           core radius (px) for the frame PSF normalisation (1.5
                        l/D)
  --groups GROUPS [GROUPS ...]
                        image groups as partitions: 'auto' (pyNOMIC's star-
                        position split), JD split times, or @file with one
                        integer label per frame; default: no grouping
                        (partitions = chop states)
  --group-min-frames GROUP_MIN_FRAMES
                        groups with fewer frames are left out
  --group-smooth GROUP_SMOOTH
                        smoothing kernel (frames) of the auto split
```

## `klip-tpe near`

```
usage: klip-tpe near [-h] [--instrument {near,nomic,generic}]
                     [--root ROOT [ROOT ...]] [--cube CUBE [CUBE ...]]
                     [--angles ANGLES [ANGLES ...]] [--psf PSF [PSF ...]]
                     [--ref-cube REF_CUBE [REF_CUBE ...]]
                     [--names NAMES [NAMES ...]]
                     [--star-flux STAR_FLUX [STAR_FLUX ...]]
                     [--pxscale PXSCALE] [--lam LAM] [--diam DIAM]
                     [--angle-sign ANGLE_SIGN] [--crop-half CROP_HALF]
                     [--center CENTER CENTER] [--wv-index WV_INDEX]
                     [--no-tags] [--nights NIGHTS [NIGHTS ...]]
                     [--backend {klip,pyklip,vip}] [--obj OBJ [OBJ ...]]
                     [--frames FRAMES] [--binned] [--pre-bin PRE_BIN]
                     [--parang-sign PARANG_SIGN] [--truenorth TRUENORTH]
                     [--fwhm-px FWHM_PX] [--name NAME] [--r-ee R_EE]
                     [--groups GROUPS [GROUPS ...]]
                     [--group-min-frames GROUP_MIN_FRAMES]
                     [--group-smooth GROUP_SMOOTH] [--max-frames MAX_FRAMES]
                     [--workers WORKERS] [--pool {auto,processes,threads}]
                     [--weighting {equal,sqrt_texp}] [--fast] [--no-library]
                     [--no-clean-subtract] [--ladder-pair]
                     [--n-min-ref N_MIN_REF] [--ref-frac REF_FRAC]
                     [--known RHO PA] [--metric {mawet,fmmf}]
                     [--run-dir RUN_DIR]
                     [--ann-edges ANN_EDGES [ANN_EDGES ...]]
                     [--n-iter N_ITER [N_ITER ...]]
                     [--n-init N_INIT [N_INIT ...]] [--mode {tpe,random,grid}]
                     [--blocks BLOCKS] [--seed SEED] [--contrast0 CONTRAST0]
                     [--use-contrast USE_CONTRAST [USE_CONTRAST ...]]
                     [--cmax-cal CMAX_CAL] [--k-max K_MAX] [--n-top N_TOP]
                     [--n-valid N_VALID] [--global-block] [--no-framesel]
                     [--no-selection] [--max-drop MAX_DROP] [--tag TAG]
                     [--from-idl-setup RUN_SETUP_TXT] [--opt-width]
                     [--width-range W0 W1] [--r-cap R_CAP]
                     [--k-mode {search,scan_rescore,scan}]
                     [--k-scan-max K_SCAN_MAX] [--stitch-every STITCH_EVERY]
                     [--verify] [--no-verify] [--verify-n-boot VERIFY_N_BOOT]
                     [--param-verify] [--no-param-verify] [--n-pv N_PV]
                     [--pv-divmin PV_DIVMIN] [--candidates]
                     [--write-setup-files] [--no-setup-files]
                     [--legacy-stitch] [--display] [--no-display]
                     [--display-every DISPLAY_EVERY] [--pdf-every PDF_EVERY]
                     [--movie-every MOVIE_EVERY] [--show [SHOW]] [--aliens]
                     [--window-scale WINDOW_SCALE] [--no-fm-curve]
                     [--fm-preview] [--no-fm-preview]

options:
  -h, --help            show this help message and exit
  --instrument {near,nomic,generic}
                        near: NEAR campaign tree under --root; nomic: pyNOMIC
                        working directory under --root; generic: any
                        registered cube (--cube/--angles/--psf ...)
  --root ROOT [ROOT ...]
                        NEAR: data root. nomic: pyNOMIC working directory (one
                        per --obj, or one for all). generic: optional base
                        directory for the run directory
  --nights NIGHTS [NIGHTS ...]
  --backend {klip,pyklip,vip}
                        PSF-subtraction engine: built-in annular KLIP
                        (default), pyKLIP or VIP (must be installed)
  --max-frames MAX_FRAMES
                        use only the first N frames per night (tests)
  --workers WORKERS     worker budget: 'auto' (default) = all cores, an
                        integer pins it, -k = all but k
  --pool {auto,processes,threads}
                        how the budget is spent: forked worker processes
                        (default; the IDL bridges) or threads
  --weighting {equal,sqrt_texp}
  --fast                single-basis KLIP (angsep/anglemax not searched)
  --no-library
  --no-clean-subtract
  --ladder-pair         two-source annulus: use the pre-2026-09-05 two-rung
                        radial ladder instead of both sources at the annulus'
                        area-weighted mid radius (only to reproduce older IDL
                        runs)
  --n-min-ref N_MIN_REF
  --ref-frac REF_FRAC
  --known RHO PA
  --metric {mawet,fmmf}
                        detection metric: 'mawet' (default) filters with the
                        injection PSF; 'fmmf' forward-models that PSF through
                        each configuration's own subtraction and filters with
                        the result (klip_tpe.fmmf)
  --run-dir RUN_DIR     output directory (default:
                        <root>/comb/opt/run_YYYYMMDD_HHMMSS, the IDL layout;
                        generic: ./runs/run_...)
  --ann-edges ANN_EDGES [ANN_EDGES ...]
  --n-iter N_ITER [N_ITER ...]
  --n-init N_INIT [N_INIT ...]
  --mode {tpe,random,grid}
  --blocks BLOCKS       partitions | univariate | full
  --seed SEED
  --contrast0 CONTRAST0
  --use-contrast USE_CONTRAST [USE_CONTRAST ...]
                        forced contrast per annulus (0 = calibrate)
  --cmax-cal CMAX_CAL
  --k-max K_MAX
  --n-top N_TOP
  --n-valid N_VALID
  --global-block        one parameter block for all nights
  --no-framesel
  --no-selection        no drop1/drop2 night-selection dims
  --max-drop MAX_DROP   drop slots of the partition selection (2 = IDL
                        drop1/drop2; more for nights x groups)
  --tag TAG
  --from-idl-setup RUN_SETUP_TXT
                        mirror an IDL run's settings (annulus, budget,
                        contrast, validation, TPE) for a matched Python run
  --opt-width           adaptive annuli: --ann-edges gives the first inner
                        edge; 'width' becomes a searched dim
  --width-range W0 W1
  --r-cap R_CAP         outer radius cap (px) for --opt-width
  --k-mode {search,scan_rescore,scan}
                        search: k_klip searched; scan/scan_rescore: per-eval
                        k-scan (k not searched)
  --k-scan-max K_SCAN_MAX
  --stitch-every STITCH_EVERY
                        running-stitch cadence (evals); 0 = annulus end only
  --verify              klip_tpe.verify report + limit curve after each
                        annulus
  --no-verify
  --verify-n-boot VERIFY_N_BOOT
  --param-verify        parameter-ensemble persistence stage (default: on when
                        per-night blocks exist)
  --no-param-verify
  --n-pv N_PV
  --pv-divmin PV_DIVMIN
  --candidates          blind candidate search on the running/final stitch
  --write-setup-files
  --no-setup-files
  --legacy-stitch       equal-weight stitch, no contrast-curve seam trim
  --display             live display: per-eval step panels, books, progress
                        movie (default on)
  --no-display
  --display-every DISPLAY_EVERY
                        render cadence in evaluations
  --pdf-every PDF_EVERY
  --movie-every MOVIE_EVERY
                        rebuild annulusNN/progress.gif + .mp4 every N
                        evaluations (default: --pdf-every; 0 = annulus end
                        only)
  --show [SHOW]         live panel on screen: --show (a matplotlib window),
                        --show inline (update a Jupyter output cell in place),
                        --show auto (inline inside a notebook, window
                        otherwise)
  --aliens              play the IDL launch movie during the first calibration
                        (+ intro.gif)
  --window-scale WINDOW_SCALE
                        live window size as a fraction of the 1850x990 panel
                        (1.0 = the IDL window, 1:1 pixels)
  --no-fm-curve         skip the KLIP-FM cross-check curve after each annulus
                        (A 9.1)
  --fm-preview          live KLIP-FM preview at each new best (A 9.3; one
                        extra reduction per new best; default on)
  --no-fm-preview

generic cubes (--instrument generic / klip-tpe generic):
  --cube CUBE [CUBE ...]
                        registered cube(s), (n, ny, nx) FITS; one per
                        partition
  --angles ANGLES [ANGLES ...]
                        derotation angles FITS (one per cube; CCW to North-up)
  --psf PSF [PSF ...]   generic: off-axis PSF template FITS (one, or one per
                        cube); nomic: airy (default) | frame | gaussian
  --ref-cube REF_CUBE [REF_CUBE ...]
                        PSF-reference cube(s) for RDI/ARDI (one per cube)
  --names NAMES [NAMES ...]
                        partition names (default: cube file stems)
  --star-flux STAR_FLUX [STAR_FLUX ...]
                        star flux in science-frame units per cube, or 'halo'
                        (scale the PSF template to the halo)
  --pxscale PXSCALE     arcsec / px
  --lam LAM             wavelength (m)
  --diam DIAM           aperture diameter (m) for lambda/D
  --angle-sign ANGLE_SIGN
                        -1 flips the derotation-angle sign
  --crop-half CROP_HALF
                        crop to 2h+1 px around --center (generic default: full
                        frame; nomic default: 75)
  --center CENTER CENTER
                        star pixel (x y); default: array centre
  --wv-index WV_INDEX   channel of a 4-d IFS cube
  --no-tags             no data-derived frame-quality tags (no frame
                        selection)

pyNOMIC (--instrument nomic):
  --obj OBJ [OBJ ...]   pyNOMIC object name(s) (the <obj>_NOMIC_*.npz prefix)
  --frames FRAMES       frame directory under the working dir (masked |
                        aligned)
  --binned              use pyNOMIC's <obj>_NOMIC_binned_evaluated.npz frames
  --pre-bin PRE_BIN     mean-bin every N frames at load time (memory)
  --parang-sign PARANG_SIGN
                        -1 flips the parallactic-angle sign
  --truenorth TRUENORTH
  --fwhm-px FWHM_PX     override the FWHM (default: Airy fit / 1.028 l/D)
  --name NAME           partition-name prefix (default: --obj)
  --r-ee R_EE           core radius (px) for the frame PSF normalisation (1.5
                        l/D)
  --groups GROUPS [GROUPS ...]
                        image groups as partitions: 'auto' (pyNOMIC's star-
                        position split), JD split times, or @file with one
                        integer label per frame; default: no grouping
                        (partitions = chop states)
  --group-min-frames GROUP_MIN_FRAMES
                        groups with fewer frames are left out
  --group-smooth GROUP_SMOOTH
                        smoothing kernel (frames) of the auto split
```

## `klip-tpe resume`

```
usage: klip-tpe resume [-h] [--instrument {near,nomic,generic}]
                       [--root ROOT [ROOT ...]] [--cube CUBE [CUBE ...]]
                       [--angles ANGLES [ANGLES ...]] [--psf PSF [PSF ...]]
                       [--ref-cube REF_CUBE [REF_CUBE ...]]
                       [--names NAMES [NAMES ...]]
                       [--star-flux STAR_FLUX [STAR_FLUX ...]]
                       [--pxscale PXSCALE] [--lam LAM] [--diam DIAM]
                       [--angle-sign ANGLE_SIGN] [--crop-half CROP_HALF]
                       [--center CENTER CENTER] [--wv-index WV_INDEX]
                       [--no-tags] [--nights NIGHTS [NIGHTS ...]]
                       [--backend {klip,pyklip,vip}] [--obj OBJ [OBJ ...]]
                       [--frames FRAMES] [--binned] [--pre-bin PRE_BIN]
                       [--parang-sign PARANG_SIGN] [--truenorth TRUENORTH]
                       [--fwhm-px FWHM_PX] [--name NAME] [--r-ee R_EE]
                       [--groups GROUPS [GROUPS ...]]
                       [--group-min-frames GROUP_MIN_FRAMES]
                       [--group-smooth GROUP_SMOOTH] [--max-frames MAX_FRAMES]
                       [--workers WORKERS] [--pool {auto,processes,threads}]
                       [--weighting {equal,sqrt_texp}] [--fast] [--no-library]
                       [--no-clean-subtract] [--ladder-pair]
                       [--n-min-ref N_MIN_REF] [--ref-frac REF_FRAC]
                       [--known RHO PA] [--metric {mawet,fmmf}]
                       [--run-dir RUN_DIR] [--display] [--no-display]
                       [--display-every DISPLAY_EVERY] [--pdf-every PDF_EVERY]
                       [--movie-every MOVIE_EVERY] [--show [SHOW]] [--aliens]
                       [--window-scale WINDOW_SCALE]

options:
  -h, --help            show this help message and exit
  --instrument {near,nomic,generic}
                        near: NEAR campaign tree under --root; nomic: pyNOMIC
                        working directory under --root; generic: any
                        registered cube (--cube/--angles/--psf ...)
  --root ROOT [ROOT ...]
                        NEAR: data root. nomic: pyNOMIC working directory (one
                        per --obj, or one for all). generic: optional base
                        directory for the run directory
  --nights NIGHTS [NIGHTS ...]
  --backend {klip,pyklip,vip}
                        PSF-subtraction engine: built-in annular KLIP
                        (default), pyKLIP or VIP (must be installed)
  --max-frames MAX_FRAMES
                        use only the first N frames per night (tests)
  --workers WORKERS     worker budget: 'auto' (default) = all cores, an
                        integer pins it, -k = all but k
  --pool {auto,processes,threads}
                        how the budget is spent: forked worker processes
                        (default; the IDL bridges) or threads
  --weighting {equal,sqrt_texp}
  --fast                single-basis KLIP (angsep/anglemax not searched)
  --no-library
  --no-clean-subtract
  --ladder-pair         two-source annulus: use the pre-2026-09-05 two-rung
                        radial ladder instead of both sources at the annulus'
                        area-weighted mid radius (only to reproduce older IDL
                        runs)
  --n-min-ref N_MIN_REF
  --ref-frac REF_FRAC
  --known RHO PA
  --metric {mawet,fmmf}
                        detection metric: 'mawet' (default) filters with the
                        injection PSF; 'fmmf' forward-models that PSF through
                        each configuration's own subtraction and filters with
                        the result (klip_tpe.fmmf)
  --run-dir RUN_DIR     run directory, or 'last' (default) for the newest
                        run_* under <root>/comb/opt
  --display
  --no-display
  --display-every DISPLAY_EVERY
  --pdf-every PDF_EVERY
  --movie-every MOVIE_EVERY
  --show [SHOW]
  --aliens
  --window-scale WINDOW_SCALE

generic cubes (--instrument generic / klip-tpe generic):
  --cube CUBE [CUBE ...]
                        registered cube(s), (n, ny, nx) FITS; one per
                        partition
  --angles ANGLES [ANGLES ...]
                        derotation angles FITS (one per cube; CCW to North-up)
  --psf PSF [PSF ...]   generic: off-axis PSF template FITS (one, or one per
                        cube); nomic: airy (default) | frame | gaussian
  --ref-cube REF_CUBE [REF_CUBE ...]
                        PSF-reference cube(s) for RDI/ARDI (one per cube)
  --names NAMES [NAMES ...]
                        partition names (default: cube file stems)
  --star-flux STAR_FLUX [STAR_FLUX ...]
                        star flux in science-frame units per cube, or 'halo'
                        (scale the PSF template to the halo)
  --pxscale PXSCALE     arcsec / px
  --lam LAM             wavelength (m)
  --diam DIAM           aperture diameter (m) for lambda/D
  --angle-sign ANGLE_SIGN
                        -1 flips the derotation-angle sign
  --crop-half CROP_HALF
                        crop to 2h+1 px around --center (generic default: full
                        frame; nomic default: 75)
  --center CENTER CENTER
                        star pixel (x y); default: array centre
  --wv-index WV_INDEX   channel of a 4-d IFS cube
  --no-tags             no data-derived frame-quality tags (no frame
                        selection)

pyNOMIC (--instrument nomic):
  --obj OBJ [OBJ ...]   pyNOMIC object name(s) (the <obj>_NOMIC_*.npz prefix)
  --frames FRAMES       frame directory under the working dir (masked |
                        aligned)
  --binned              use pyNOMIC's <obj>_NOMIC_binned_evaluated.npz frames
  --pre-bin PRE_BIN     mean-bin every N frames at load time (memory)
  --parang-sign PARANG_SIGN
                        -1 flips the parallactic-angle sign
  --truenorth TRUENORTH
  --fwhm-px FWHM_PX     override the FWHM (default: Airy fit / 1.028 l/D)
  --name NAME           partition-name prefix (default: --obj)
  --r-ee R_EE           core radius (px) for the frame PSF normalisation (1.5
                        l/D)
  --groups GROUPS [GROUPS ...]
                        image groups as partitions: 'auto' (pyNOMIC's star-
                        position split), JD split times, or @file with one
                        integer label per frame; default: no grouping
                        (partitions = chop states)
  --group-min-frames GROUP_MIN_FRAMES
                        groups with fewer frames are left out
  --group-smooth GROUP_SMOOTH
                        smoothing kernel (frames) of the auto split
```

## `klip-tpe extend`

```
usage: klip-tpe extend [-h] [--instrument {near,nomic,generic}]
                       [--root ROOT [ROOT ...]] [--cube CUBE [CUBE ...]]
                       [--angles ANGLES [ANGLES ...]] [--psf PSF [PSF ...]]
                       [--ref-cube REF_CUBE [REF_CUBE ...]]
                       [--names NAMES [NAMES ...]]
                       [--star-flux STAR_FLUX [STAR_FLUX ...]]
                       [--pxscale PXSCALE] [--lam LAM] [--diam DIAM]
                       [--angle-sign ANGLE_SIGN] [--crop-half CROP_HALF]
                       [--center CENTER CENTER] [--wv-index WV_INDEX]
                       [--no-tags] [--nights NIGHTS [NIGHTS ...]]
                       [--backend {klip,pyklip,vip}] [--obj OBJ [OBJ ...]]
                       [--frames FRAMES] [--binned] [--pre-bin PRE_BIN]
                       [--parang-sign PARANG_SIGN] [--truenorth TRUENORTH]
                       [--fwhm-px FWHM_PX] [--name NAME] [--r-ee R_EE]
                       [--groups GROUPS [GROUPS ...]]
                       [--group-min-frames GROUP_MIN_FRAMES]
                       [--group-smooth GROUP_SMOOTH] [--max-frames MAX_FRAMES]
                       [--workers WORKERS] [--pool {auto,processes,threads}]
                       [--weighting {equal,sqrt_texp}] [--fast] [--no-library]
                       [--no-clean-subtract] [--ladder-pair]
                       [--n-min-ref N_MIN_REF] [--ref-frac REF_FRAC]
                       [--known RHO PA] [--metric {mawet,fmmf}] --run-dir
                       RUN_DIR --n-iter N_ITER [N_ITER ...]

options:
  -h, --help            show this help message and exit
  --instrument {near,nomic,generic}
                        near: NEAR campaign tree under --root; nomic: pyNOMIC
                        working directory under --root; generic: any
                        registered cube (--cube/--angles/--psf ...)
  --root ROOT [ROOT ...]
                        NEAR: data root. nomic: pyNOMIC working directory (one
                        per --obj, or one for all). generic: optional base
                        directory for the run directory
  --nights NIGHTS [NIGHTS ...]
  --backend {klip,pyklip,vip}
                        PSF-subtraction engine: built-in annular KLIP
                        (default), pyKLIP or VIP (must be installed)
  --max-frames MAX_FRAMES
                        use only the first N frames per night (tests)
  --workers WORKERS     worker budget: 'auto' (default) = all cores, an
                        integer pins it, -k = all but k
  --pool {auto,processes,threads}
                        how the budget is spent: forked worker processes
                        (default; the IDL bridges) or threads
  --weighting {equal,sqrt_texp}
  --fast                single-basis KLIP (angsep/anglemax not searched)
  --no-library
  --no-clean-subtract
  --ladder-pair         two-source annulus: use the pre-2026-09-05 two-rung
                        radial ladder instead of both sources at the annulus'
                        area-weighted mid radius (only to reproduce older IDL
                        runs)
  --n-min-ref N_MIN_REF
  --ref-frac REF_FRAC
  --known RHO PA
  --metric {mawet,fmmf}
                        detection metric: 'mawet' (default) filters with the
                        injection PSF; 'fmmf' forward-models that PSF through
                        each configuration's own subtraction and filters with
                        the result (klip_tpe.fmmf)
  --run-dir RUN_DIR
  --n-iter N_ITER [N_ITER ...]
                        new total evaluations per annulus

generic cubes (--instrument generic / klip-tpe generic):
  --cube CUBE [CUBE ...]
                        registered cube(s), (n, ny, nx) FITS; one per
                        partition
  --angles ANGLES [ANGLES ...]
                        derotation angles FITS (one per cube; CCW to North-up)
  --psf PSF [PSF ...]   generic: off-axis PSF template FITS (one, or one per
                        cube); nomic: airy (default) | frame | gaussian
  --ref-cube REF_CUBE [REF_CUBE ...]
                        PSF-reference cube(s) for RDI/ARDI (one per cube)
  --names NAMES [NAMES ...]
                        partition names (default: cube file stems)
  --star-flux STAR_FLUX [STAR_FLUX ...]
                        star flux in science-frame units per cube, or 'halo'
                        (scale the PSF template to the halo)
  --pxscale PXSCALE     arcsec / px
  --lam LAM             wavelength (m)
  --diam DIAM           aperture diameter (m) for lambda/D
  --angle-sign ANGLE_SIGN
                        -1 flips the derotation-angle sign
  --crop-half CROP_HALF
                        crop to 2h+1 px around --center (generic default: full
                        frame; nomic default: 75)
  --center CENTER CENTER
                        star pixel (x y); default: array centre
  --wv-index WV_INDEX   channel of a 4-d IFS cube
  --no-tags             no data-derived frame-quality tags (no frame
                        selection)

pyNOMIC (--instrument nomic):
  --obj OBJ [OBJ ...]   pyNOMIC object name(s) (the <obj>_NOMIC_*.npz prefix)
  --frames FRAMES       frame directory under the working dir (masked |
                        aligned)
  --binned              use pyNOMIC's <obj>_NOMIC_binned_evaluated.npz frames
  --pre-bin PRE_BIN     mean-bin every N frames at load time (memory)
  --parang-sign PARANG_SIGN
                        -1 flips the parallactic-angle sign
  --truenorth TRUENORTH
  --fwhm-px FWHM_PX     override the FWHM (default: Airy fit / 1.028 l/D)
  --name NAME           partition-name prefix (default: --obj)
  --r-ee R_EE           core radius (px) for the frame PSF normalisation (1.5
                        l/D)
  --groups GROUPS [GROUPS ...]
                        image groups as partitions: 'auto' (pyNOMIC's star-
                        position split), JD split times, or @file with one
                        integer label per frame; default: no grouping
                        (partitions = chop states)
  --group-min-frames GROUP_MIN_FRAMES
                        groups with fewer frames are left out
  --group-smooth GROUP_SMOOTH
                        smoothing kernel (frames) of the auto split
```

## `klip-tpe plots`

```
usage: klip-tpe plots [-h] --run-dir RUN_DIR

options:
  -h, --help         show this help message and exit
  --run-dir RUN_DIR
```

## `klip-tpe replay`

```
usage: klip-tpe replay [-h] [--instrument {near,nomic,generic}]
                       [--root ROOT [ROOT ...]] [--cube CUBE [CUBE ...]]
                       [--angles ANGLES [ANGLES ...]] [--psf PSF [PSF ...]]
                       [--ref-cube REF_CUBE [REF_CUBE ...]]
                       [--names NAMES [NAMES ...]]
                       [--star-flux STAR_FLUX [STAR_FLUX ...]]
                       [--pxscale PXSCALE] [--lam LAM] [--diam DIAM]
                       [--angle-sign ANGLE_SIGN] [--crop-half CROP_HALF]
                       [--center CENTER CENTER] [--wv-index WV_INDEX]
                       [--no-tags] [--nights NIGHTS [NIGHTS ...]]
                       [--backend {klip,pyklip,vip}] [--obj OBJ [OBJ ...]]
                       [--frames FRAMES] [--binned] [--pre-bin PRE_BIN]
                       [--parang-sign PARANG_SIGN] [--truenorth TRUENORTH]
                       [--fwhm-px FWHM_PX] [--name NAME] [--r-ee R_EE]
                       [--groups GROUPS [GROUPS ...]]
                       [--group-min-frames GROUP_MIN_FRAMES]
                       [--group-smooth GROUP_SMOOTH] [--max-frames MAX_FRAMES]
                       [--workers WORKERS] [--pool {auto,processes,threads}]
                       [--weighting {equal,sqrt_texp}] [--fast] [--no-library]
                       [--no-clean-subtract] [--ladder-pair]
                       [--n-min-ref N_MIN_REF] [--ref-frac REF_FRAC]
                       [--known RHO PA] [--metric {mawet,fmmf}] --idl-log
                       IDL_LOG [--run-dir RUN_DIR] [--n N] [--repeat REPEAT]
                       [--contrast CONTRAST]
                       [--ann-edges ANN_EDGES [ANN_EDGES ...]] [--k-max K_MAX]
                       [--anglemax-hi ANGLEMAX_HI] [--seed SEED]

options:
  -h, --help            show this help message and exit
  --instrument {near,nomic,generic}
                        near: NEAR campaign tree under --root; nomic: pyNOMIC
                        working directory under --root; generic: any
                        registered cube (--cube/--angles/--psf ...)
  --root ROOT [ROOT ...]
                        NEAR: data root. nomic: pyNOMIC working directory (one
                        per --obj, or one for all). generic: optional base
                        directory for the run directory
  --nights NIGHTS [NIGHTS ...]
  --backend {klip,pyklip,vip}
                        PSF-subtraction engine: built-in annular KLIP
                        (default), pyKLIP or VIP (must be installed)
  --max-frames MAX_FRAMES
                        use only the first N frames per night (tests)
  --workers WORKERS     worker budget: 'auto' (default) = all cores, an
                        integer pins it, -k = all but k
  --pool {auto,processes,threads}
                        how the budget is spent: forked worker processes
                        (default; the IDL bridges) or threads
  --weighting {equal,sqrt_texp}
  --fast                single-basis KLIP (angsep/anglemax not searched)
  --no-library
  --no-clean-subtract
  --ladder-pair         two-source annulus: use the pre-2026-09-05 two-rung
                        radial ladder instead of both sources at the annulus'
                        area-weighted mid radius (only to reproduce older IDL
                        runs)
  --n-min-ref N_MIN_REF
  --ref-frac REF_FRAC
  --known RHO PA
  --metric {mawet,fmmf}
                        detection metric: 'mawet' (default) filters with the
                        injection PSF; 'fmmf' forward-models that PSF through
                        each configuration's own subtraction and filters with
                        the result (klip_tpe.fmmf)
  --idl-log IDL_LOG
  --run-dir RUN_DIR
  --n N
  --repeat REPEAT
  --contrast CONTRAST
  --ann-edges ANN_EDGES [ANN_EDGES ...]
  --k-max K_MAX
  --anglemax-hi ANGLEMAX_HI
  --seed SEED

generic cubes (--instrument generic / klip-tpe generic):
  --cube CUBE [CUBE ...]
                        registered cube(s), (n, ny, nx) FITS; one per
                        partition
  --angles ANGLES [ANGLES ...]
                        derotation angles FITS (one per cube; CCW to North-up)
  --psf PSF [PSF ...]   generic: off-axis PSF template FITS (one, or one per
                        cube); nomic: airy (default) | frame | gaussian
  --ref-cube REF_CUBE [REF_CUBE ...]
                        PSF-reference cube(s) for RDI/ARDI (one per cube)
  --names NAMES [NAMES ...]
                        partition names (default: cube file stems)
  --star-flux STAR_FLUX [STAR_FLUX ...]
                        star flux in science-frame units per cube, or 'halo'
                        (scale the PSF template to the halo)
  --pxscale PXSCALE     arcsec / px
  --lam LAM             wavelength (m)
  --diam DIAM           aperture diameter (m) for lambda/D
  --angle-sign ANGLE_SIGN
                        -1 flips the derotation-angle sign
  --crop-half CROP_HALF
                        crop to 2h+1 px around --center (generic default: full
                        frame; nomic default: 75)
  --center CENTER CENTER
                        star pixel (x y); default: array centre
  --wv-index WV_INDEX   channel of a 4-d IFS cube
  --no-tags             no data-derived frame-quality tags (no frame
                        selection)

pyNOMIC (--instrument nomic):
  --obj OBJ [OBJ ...]   pyNOMIC object name(s) (the <obj>_NOMIC_*.npz prefix)
  --frames FRAMES       frame directory under the working dir (masked |
                        aligned)
  --binned              use pyNOMIC's <obj>_NOMIC_binned_evaluated.npz frames
  --pre-bin PRE_BIN     mean-bin every N frames at load time (memory)
  --parang-sign PARANG_SIGN
                        -1 flips the parallactic-angle sign
  --truenorth TRUENORTH
  --fwhm-px FWHM_PX     override the FWHM (default: Airy fit / 1.028 l/D)
  --name NAME           partition-name prefix (default: --obj)
  --r-ee R_EE           core radius (px) for the frame PSF normalisation (1.5
                        l/D)
  --groups GROUPS [GROUPS ...]
                        image groups as partitions: 'auto' (pyNOMIC's star-
                        position split), JD split times, or @file with one
                        integer label per frame; default: no grouping
                        (partitions = chop states)
  --group-min-frames GROUP_MIN_FRAMES
                        groups with fewer frames are left out
  --group-smooth GROUP_SMOOTH
                        smoothing kernel (frames) of the auto split
```

## `klip-tpe compare`

```
usage: klip-tpe compare [-h] [--instrument {near,nomic,generic}]
                        [--root ROOT [ROOT ...]] [--cube CUBE [CUBE ...]]
                        [--angles ANGLES [ANGLES ...]] [--psf PSF [PSF ...]]
                        [--ref-cube REF_CUBE [REF_CUBE ...]]
                        [--names NAMES [NAMES ...]]
                        [--star-flux STAR_FLUX [STAR_FLUX ...]]
                        [--pxscale PXSCALE] [--lam LAM] [--diam DIAM]
                        [--angle-sign ANGLE_SIGN] [--crop-half CROP_HALF]
                        [--center CENTER CENTER] [--wv-index WV_INDEX]
                        [--no-tags] [--nights NIGHTS [NIGHTS ...]]
                        [--backend {klip,pyklip,vip}] [--obj OBJ [OBJ ...]]
                        [--frames FRAMES] [--binned] [--pre-bin PRE_BIN]
                        [--parang-sign PARANG_SIGN] [--truenorth TRUENORTH]
                        [--fwhm-px FWHM_PX] [--name NAME] [--r-ee R_EE]
                        [--groups GROUPS [GROUPS ...]]
                        [--group-min-frames GROUP_MIN_FRAMES]
                        [--group-smooth GROUP_SMOOTH]
                        [--max-frames MAX_FRAMES] [--workers WORKERS]
                        [--pool {auto,processes,threads}]
                        [--weighting {equal,sqrt_texp}] [--fast]
                        [--no-library] [--no-clean-subtract] [--ladder-pair]
                        [--n-min-ref N_MIN_REF] [--ref-frac REF_FRAC]
                        [--known RHO PA] [--metric {mawet,fmmf}] --idl-run
                        IDL_RUN --out OUT [--annulus ANNULUS]
                        [--ann-edges ANN_EDGES [ANN_EDGES ...]]
                        [--contrast CONTRAST] [--contrast-from-idl]
                        [--max-new MAX_NEW] [--stride STRIDE]
                        [--stride-offset STRIDE_OFFSET] [--repeat REPEAT]
                        [--seed SEED] [--k-max K_MAX] [--global-block]
                        [--no-framesel] [--no-selection] [--opt-width]
                        [--width-range WIDTH_RANGE WIDTH_RANGE]
                        [--k-mode K_MODE]

options:
  -h, --help            show this help message and exit
  --instrument {near,nomic,generic}
                        near: NEAR campaign tree under --root; nomic: pyNOMIC
                        working directory under --root; generic: any
                        registered cube (--cube/--angles/--psf ...)
  --root ROOT [ROOT ...]
                        NEAR: data root. nomic: pyNOMIC working directory (one
                        per --obj, or one for all). generic: optional base
                        directory for the run directory
  --nights NIGHTS [NIGHTS ...]
  --backend {klip,pyklip,vip}
                        PSF-subtraction engine: built-in annular KLIP
                        (default), pyKLIP or VIP (must be installed)
  --max-frames MAX_FRAMES
                        use only the first N frames per night (tests)
  --workers WORKERS     worker budget: 'auto' (default) = all cores, an
                        integer pins it, -k = all but k
  --pool {auto,processes,threads}
                        how the budget is spent: forked worker processes
                        (default; the IDL bridges) or threads
  --weighting {equal,sqrt_texp}
  --fast                single-basis KLIP (angsep/anglemax not searched)
  --no-library
  --no-clean-subtract
  --ladder-pair         two-source annulus: use the pre-2026-09-05 two-rung
                        radial ladder instead of both sources at the annulus'
                        area-weighted mid radius (only to reproduce older IDL
                        runs)
  --n-min-ref N_MIN_REF
  --ref-frac REF_FRAC
  --known RHO PA
  --metric {mawet,fmmf}
                        detection metric: 'mawet' (default) filters with the
                        injection PSF; 'fmmf' forward-models that PSF through
                        each configuration's own subtraction and filters with
                        the result (klip_tpe.fmmf)
  --idl-run IDL_RUN     IDL run directory (optimize_tpe_results.txt +
                        run_setup.txt)
  --out OUT             output directory (incremental: re-run to pick up new
                        IDL evals)
  --annulus ANNULUS
  --ann-edges ANN_EDGES [ANN_EDGES ...]
  --contrast CONTRAST
  --contrast-from-idl
  --max-new MAX_NEW     replay at most N new IDL evaluations this call
  --stride STRIDE       replay every N-th IDL iteration only
  --stride-offset STRIDE_OFFSET
  --repeat REPEAT       Python re-scores per configuration (fresh azimuths)
  --seed SEED
  --k-max K_MAX
  --global-block
  --no-framesel
  --no-selection
  --opt-width
  --width-range WIDTH_RANGE WIDTH_RANGE
  --k-mode K_MODE

generic cubes (--instrument generic / klip-tpe generic):
  --cube CUBE [CUBE ...]
                        registered cube(s), (n, ny, nx) FITS; one per
                        partition
  --angles ANGLES [ANGLES ...]
                        derotation angles FITS (one per cube; CCW to North-up)
  --psf PSF [PSF ...]   generic: off-axis PSF template FITS (one, or one per
                        cube); nomic: airy (default) | frame | gaussian
  --ref-cube REF_CUBE [REF_CUBE ...]
                        PSF-reference cube(s) for RDI/ARDI (one per cube)
  --names NAMES [NAMES ...]
                        partition names (default: cube file stems)
  --star-flux STAR_FLUX [STAR_FLUX ...]
                        star flux in science-frame units per cube, or 'halo'
                        (scale the PSF template to the halo)
  --pxscale PXSCALE     arcsec / px
  --lam LAM             wavelength (m)
  --diam DIAM           aperture diameter (m) for lambda/D
  --angle-sign ANGLE_SIGN
                        -1 flips the derotation-angle sign
  --crop-half CROP_HALF
                        crop to 2h+1 px around --center (generic default: full
                        frame; nomic default: 75)
  --center CENTER CENTER
                        star pixel (x y); default: array centre
  --wv-index WV_INDEX   channel of a 4-d IFS cube
  --no-tags             no data-derived frame-quality tags (no frame
                        selection)

pyNOMIC (--instrument nomic):
  --obj OBJ [OBJ ...]   pyNOMIC object name(s) (the <obj>_NOMIC_*.npz prefix)
  --frames FRAMES       frame directory under the working dir (masked |
                        aligned)
  --binned              use pyNOMIC's <obj>_NOMIC_binned_evaluated.npz frames
  --pre-bin PRE_BIN     mean-bin every N frames at load time (memory)
  --parang-sign PARANG_SIGN
                        -1 flips the parallactic-angle sign
  --truenorth TRUENORTH
  --fwhm-px FWHM_PX     override the FWHM (default: Airy fit / 1.028 l/D)
  --name NAME           partition-name prefix (default: --obj)
  --r-ee R_EE           core radius (px) for the frame PSF normalisation (1.5
                        l/D)
  --groups GROUPS [GROUPS ...]
                        image groups as partitions: 'auto' (pyNOMIC's star-
                        position split), JD split times, or @file with one
                        integer label per frame; default: no grouping
                        (partitions = chop states)
  --group-min-frames GROUP_MIN_FRAMES
                        groups with fewer frames are left out
  --group-smooth GROUP_SMOOTH
                        smoothing kernel (frames) of the auto split
```

## `klip-tpe testbed`

```
usage: klip-tpe testbed [-h] [--nblock NBLOCK] [--bdim BDIM] [--n-iter N_ITER]
                        [--n-init N_INIT] [--nseed NSEED] [--noise NOISE]
                        [--ridge RIDGE] [--pbest PBEST] [--transposed]
                        [--out OUT]

options:
  -h, --help       show this help message and exit
  --nblock NBLOCK
  --bdim BDIM
  --n-iter N_ITER
  --n-init N_INIT
  --nseed NSEED
  --noise NOISE
  --ridge RIDGE
  --pbest PBEST
  --transposed
  --out OUT
```
