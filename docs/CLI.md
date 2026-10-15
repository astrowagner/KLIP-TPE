# Command-line reference

Generated from `klip-tpe <command> --help` (`python scripts/gen_cli_doc.py`).  `near` and
`generic` share the same search / protocol arguments; `--instrument near | nomic | generic`
selects the data adapter (`klip-tpe generic` is `near --instrument generic`).  `resume`,
`extend`, `fit-chain` and `rereduce` take the same data arguments as the run they read.


## `klip-tpe generic`

```
usage: klip-tpe generic [-h] [--instrument {near,nomic,generic,ales}]
                        [--root ROOT [ROOT ...]] [--cube CUBE [CUBE ...]]
                        [--angles ANGLES [ANGLES ...]] [--psf PSF [PSF ...]]
                        [--ref-cube REF_CUBE [REF_CUBE ...]]
                        [--names NAMES [NAMES ...]]
                        [--star-flux STAR_FLUX [STAR_FLUX ...]]
                        [--pxscale PXSCALE] [--lam LAM] [--diam DIAM]
                        [--angle-sign ANGLE_SIGN] [--crop-half CROP_HALF]
                        [--center CENTER CENTER] [--wv-index WV_INDEX]
                        [--no-tags] [--forbidden-pa PA HALFWIDTH]
                        [--mask-line PA HALFWIDTH] [--ales-dataset NAME]
                        [--ales-nights NIGHT [NIGHT ...]] [--ales-target NAME]
                        [--ales-ref NAME] [--ales-reflib] [--ales-ref-bin N]
                        [--ales-reflib-science {all,searched}] [--ales-sdi]
                        [--ales-sdi-cap N] [--ales-sdi-lenders {all,selected}]
                        [--ales-groups N] [--ales-channels C0 C1]
                        [--ales-only gNN [gNN ...]] [--ales-max-excluded N]
                        [--ales-prefer-um UM] [--ales-representative gNN]
                        [--ales-angle-offset DEG] [--ales-tag-r-in PX]
                        [--ales-cache DIR]
                        [--ales-selection {binary,two_slot}]
                        [--nights NIGHTS [NIGHTS ...]]
                        [--backend {klip,pyklip,vip}] [--obj OBJ [OBJ ...]]
                        [--frames FRAMES] [--binned] [--pre-bin PRE_BIN]
                        [--parang-sign PARANG_SIGN] [--truenorth TRUENORTH]
                        [--fwhm-px FWHM_PX] [--name NAME] [--r-ee R_EE]
                        [--groups GROUPS [GROUPS ...]]
                        [--group-min-frames GROUP_MIN_FRAMES]
                        [--group-smooth GROUP_SMOOTH]
                        [--max-frames MAX_FRAMES] [--workers WORKERS]
                        [--pool {auto,processes,threads}]
                        [--threads-per-job THREADS_PER_JOB]
                        [--weighting {equal,sqrt_texp}] [--fast]
                        [--klip-gram {target,zone}] [--no-library]
                        [--no-clean-subtract] [--ladder-pair]
                        [--n-min-ref N_MIN_REF] [--ref-frac REF_FRAC]
                        [--bin-min N] [--bin-max N] [--known RHO PA]
                        [--cube-name NAME] [--subtract-sources FILE]
                        [--subtract-line FILE] [--inject-sources FILE]
                        [--metric {mawet,fmmf}] [--run-dir RUN_DIR]
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
                        [--no-param-verify] [--n-remeasure N]
                        [--n-sources N [N ...]] [--min-r-fwhm X] [--n-pv N_PV]
                        [--pv-divmin PV_DIVMIN] [--candidates]
                        [--write-setup-files] [--no-setup-files]
                        [--legacy-stitch] [--batch BATCH]
                        [--batch-mode {pipeline,sync}]
                        [--edge-padding {between,all}] [--display]
                        [--no-display] [--display-every DISPLAY_EVERY]
                        [--pdf-every PDF_EVERY] [--movie-every MOVIE_EVERY]
                        [--show [SHOW]] [--aliens]
                        [--window-scale WINDOW_SCALE] [--no-fm-curve]
                        [--fm-preview] [--no-fm-preview]
                        [--no-liveness-check | --strict-liveness]

options:
  -h, --help            show this help message and exit
  --instrument {near,nomic,generic,ales}
                        near: NEAR campaign tree under --root; nomic: pyNOMIC
                        working directory under --root; generic: any
                        registered cube (--cube/--angles/--psf ...); ales: an
                        LBTI/ALES reduction's products under --root,
                        wavelength groups as partitions
  --root ROOT [ROOT ...]
                        NEAR: data root. nomic: pyNOMIC working directory (one
                        per --obj, or one for all). generic: optional base
                        directory for the run directory. ales: the reduction's
                        products directory, read only (the run goes to --run-
                        dir), or --ales-dataset
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
  --threads-per-job THREADS_PER_JOB
                        threads per partition job (default 4). The pool gets
                        workers / threads-per-job processes, never fewer than
                        min(workers, 2 x partitions)
  --weighting {equal,sqrt_texp}
  --fast                single-basis KLIP (angsep/anglemax not searched)
  --klip-gram {target,zone}
                        how the per-target KLIP forms each target's reference
                        covariance. 'target' (default) works from its
                        references' pixels and forms all modes. 'zone' reads
                        it off one Gram matrix per zone and projects the
                        target without forming the modes, the same KLIP to
                        float64 rounding but not to the bit, and about twice
                        as fast per target at small bins. A run keeps the mode
                        it was started or last resumed with (its
                        klip_gram.json) unless this is given
  --no-library
  --no-clean-subtract
  --ladder-pair         two-source annulus: use the pre-2026-09-05 two-rung
                        radial ladder instead of both sources at the annulus'
                        area-weighted mid radius (only to reproduce older IDL
                        runs)
  --n-min-ref N_MIN_REF
  --ref-frac REF_FRAC
  --bin-min N           generic / ales: floor of the temporal bin, as a
                        projection (MinBinGuard): a draw below it is evaluated
                        at N frames per bin. Cost, not correctness: on a long
                        sequence bin 1 costs tens of times more than bin 5
                        (default 1 = no floor)
  --bin-max N           generic / ales: top of the searched temporal-bin
                        range, frames per bin (default: ales 30, generic
                        nframes / 8). A winner sitting on it
                        (distance_to_bounds bin = 0) wants it higher
  --known RHO PA
  --cube-name NAME      near: the cube read in each night's directory (default
                        AB_cube_skysub_cen_clean.fits, or AB_cube_crop150.fits
                        where only that is there), e.g. a reduction's
                        AB_cube_skysub_cen_clean_abline.fits. Recorded in the
                        run's checkpoint, and kept on resume and extend when
                        not typed
  --subtract-sources FILE
                        point sources taken out of every frame once, at load:
                        one per line, rho (arcsec), PA (deg) and contrast,
                        each injected with the negative of its contrast by the
                        injection code, so placement, true north, centre, flux
                        unit and throughput are a draw's. Every reduction sees
                        the edited frames, the KLIP-FM model cube does not.
                        Recorded in the run's checkpoint, and kept on resume
                        and extend when not typed
  --subtract-line FILE  a line through the star taken out of every frame once,
                        at load, from a node file: 'pa', 'step' (px) and
                        optional 'rho_max' lines, then |s| (arcsec) with the
                        ridge's peak contrast on each half-line. The chain is
                        a source every step px from the first node to rho_max
                        (default the crop edge), each at a(s) step / L, with L
                        the injection PSF's line integral; as --subtract-
                        sources otherwise (klip-tpe fit-chain --line-nodes
                        writes such a file)
  --inject-sources FILE
                        point sources added to every frame once, at load,
                        after any subtraction: one per line, rho (arcsec), PA
                        (deg) and contrast, by the code a draw's injection
                        uses, e.g. planets put on a line to see what its
                        subtraction does to them. Every reduction sees them,
                        the KLIP-FM model cube does not. Recorded in the run's
                        checkpoint, and kept on resume and extend when not
                        typed
  --metric {mawet,fmmf}
                        detection metric: 'mawet' (default) filters with the
                        injection PSF; 'fmmf' forward-models that PSF through
                        each configuration's own subtraction and filters with
                        the result (klip_tpe.fmmf)
  --run-dir RUN_DIR     output directory (default:
                        <root>/comb/opt/run_YYYYMMDD_HHMMSS, the IDL layout;
                        generic and ales: ./runs/run_...)
  --ann-edges ANN_EDGES [ANN_EDGES ...]
  --n-iter N_ITER [N_ITER ...]
  --n-init N_INIT [N_INIT ...]
  --mode {tpe,random,grid}
  --blocks BLOCKS       TPE density model: univariate (default; the IDL
                        reference) | partitions | full
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
  --n-remeasure N       score each trial as the MEAN of N independent
                        injection draws (default 1). The objective's per-draw
                        scatter is ~0.9 S/N on typical data, so N=3 divides it
                        by sqrt(3) and roughly halves the optimism of the
                        reported best; see docs/BUDGET.md
  --n-sources N [N ...]
                        injected sources per annulus, the last value repeating
                        (default: klip-tpe's rule by the annulus' outer edge,
                        and 0 asks for the rule in that annulus). The cap that
                        keeps each source's noise ring clean still applies,
                        and the log says when it binds
  --min-r-fwhm X        no injection inside X FWHM of the star, in every stage
                        (default 1.5). On NEAR 1.0 FWHM is 1.0 lambda/D to
                        half a percent
  --n-pv N_PV
  --pv-divmin PV_DIVMIN
  --candidates          blind candidate search on the running/final stitch
  --write-setup-files
  --no-setup-files
  --legacy-stitch       equal-weight stitch, no contrast-curve seam trim
  --batch BATCH         trials in flight at once (default 1). TPE then
                        proposes from a posterior a few trials old, which
                        changes the search's trajectory
  --batch-mode {pipeline,sync}
                        with --batch: send the next proposal as soon as the
                        oldest trial is in (pipeline, the default), or propose
                        whole batches from one posterior and wait for all of a
                        batch (sync)
  --edge-padding {between,all}
                        pad annulus edges only where two annuli meet (between,
                        the default) or on every edge including the innermost
                        and outermost (all, as the IDL does)
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
                        (1.0 = the IDL window, 1:1 pixels), shrunk to fit a
                        smaller screen; the window keeps this size
  --no-fm-curve         skip the KLIP-FM cross-check curve after each annulus
                        (A 9.1)
  --fm-preview          live KLIP-FM preview at each new best (A 9.3; one
                        extra reduction per new best; default on)
  --no-fm-preview
  --no-liveness-check   skip the pre-flight that moves each searched reduction
                        dimension alone and warns when one changes nothing
                        (klip_tpe.liveness; ~1 + ndim reductions, new runs
                        only)
  --strict-liveness     stop before the first evaluation when the pre-flight
                        finds a dimension that changes nothing, instead of
                        warning

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
                        star flux per cube, in the science frames' own units
                        and in the template's normalisation aperture; omit it
                        and the contrast axis is in template units
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
  --forbidden-pa PA HALFWIDTH
                        near / generic / ales: no injection within HALFWIDTH
                        deg of position angle PA (a known companion's azimuth,
                        a disk, a line through the star); may repeat
  --mask-line PA HALFWIDTH
                        near / generic / ales: leave the pixels within
                        HALFWIDTH px of the line through the star at position
                        angle PA (both sides) out of the noise apertures; may
                        repeat. Pair it with --forbidden-pa on the same PA and
                        PA+180, so injections stay off it too

LBTI/ALES (--instrument ales):
  --ales-dataset NAME   a public ALES data set in the release layout,
                        downloaded once into $KLIP_TPE_DATA
                        (klip_tpe.datasets: lbti_ales_mwc758), one partition
                        per night and wavelength group, in place of --root
  --ales-nights NIGHT [NIGHT ...]
                        with --ales-dataset: the nights to load, as dates
                        (20191113) or labels (n1), all by default
  --ales-target NAME    prefix of the product files under --root
                        (NAME_cube_cen_clean.fits ...; default WISPIT2)
  --ales-ref NAME       reference star whose products sit beside the target's
                        (HIP91748): its group means become each group's RDI
                        library (default: none, RDI off)
  --ales-reflib         search the reference star as a library: nkeep_psfref,
                        the reference means most correlated with each target
                        frame, searched from 0 (plain ADI, to the bit) to the
                        whole pool, on top of the ADI references (needs
                        --ales-ref)
  --ales-ref-bin N      the reference star's frames in consecutive means of N
                        before anything else (default 5; 1 = every frame)
  --ales-reflib-science {all,searched}
                        the science pool of the searched library: every frame
                        the ADI rule admits (default) or a searched count too,
                        nkeep_altroll, whose 0 is pure RDI
  --ales-sdi            ADI, SDI and RDI in one KLIP step: the other groups'
                        frames, rescaled to each group's wavelength, join its
                        library as a pool; nkeep_sdi (the rows most correlated
                        with each target frame among those the displacement
                        rule admits) is searched from 0 (no SDI) to --ales-
                        sdi-cap. A reference star given (--ales-ref) is the
                        library's third pool, searched. nkeep_sdi =
                        nkeep_psfref = 0 is plain ADI, to the bit
  --ales-sdi-cap N      most-correlated SDI rows a target frame may take, the
                        top of nkeep_sdi (default 100; the per-target cost
                        grows with the square of the basis)
  --ales-sdi-lenders {all,selected}
                        which groups lend SDI rows: every group, excluded ones
                        too (default), or only the groups the configuration
                        includes
  --ales-groups N       wavelength groups over the usable channels, equally
                        spaced in ln(lambda) (default 20; 15 = one R = 40
                        resolution element each)
  --ales-channels C0 C1
                        first and last usable channel, inclusive (default 6
                        88)
  --ales-only gNN [gNN ...]
                        load these groups only (checks and short runs)
  --ales-max-excluded N
                        the group flags may switch off at most N groups; a
                        proposal past that has groups switched back on,
                        nearest --ales-prefer-um first (default 10; -1 = no
                        bound)
  --ales-prefer-um UM   wavelength the group bound restores groups towards
                        (default 3.7)
  --ales-representative gNN
                        the group whose FWHM and matched filter stand for the
                        combined image (default: the group at the mean
                        wavelength)
  --ales-angle-offset DEG
                        added to LBT_PARA once: the derotation is CCW by
                        LBT_PARA + this (default 180)
  --ales-tag-r-in PX    inner radius of the frame-quality tags' annulus
                        (default 2 spaxels: the star's core and halo; 6, the
                        generic default, is background-dominated here)
  --ales-cache DIR      directory for the group cubes as .npy, written once
                        and read back after (a group mean reads the whole
                        cube, which may sit on a slow volume)
  --ales-selection {binary,two_slot}
                        group flags: one 0/1 dimension per group (default) or
                        --max-drop drop slots

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
usage: klip-tpe near [-h] [--instrument {near,nomic,generic,ales}]
                     [--root ROOT [ROOT ...]] [--cube CUBE [CUBE ...]]
                     [--angles ANGLES [ANGLES ...]] [--psf PSF [PSF ...]]
                     [--ref-cube REF_CUBE [REF_CUBE ...]]
                     [--names NAMES [NAMES ...]]
                     [--star-flux STAR_FLUX [STAR_FLUX ...]]
                     [--pxscale PXSCALE] [--lam LAM] [--diam DIAM]
                     [--angle-sign ANGLE_SIGN] [--crop-half CROP_HALF]
                     [--center CENTER CENTER] [--wv-index WV_INDEX]
                     [--no-tags] [--forbidden-pa PA HALFWIDTH]
                     [--mask-line PA HALFWIDTH] [--ales-dataset NAME]
                     [--ales-nights NIGHT [NIGHT ...]] [--ales-target NAME]
                     [--ales-ref NAME] [--ales-reflib] [--ales-ref-bin N]
                     [--ales-reflib-science {all,searched}] [--ales-sdi]
                     [--ales-sdi-cap N] [--ales-sdi-lenders {all,selected}]
                     [--ales-groups N] [--ales-channels C0 C1]
                     [--ales-only gNN [gNN ...]] [--ales-max-excluded N]
                     [--ales-prefer-um UM] [--ales-representative gNN]
                     [--ales-angle-offset DEG] [--ales-tag-r-in PX]
                     [--ales-cache DIR] [--ales-selection {binary,two_slot}]
                     [--nights NIGHTS [NIGHTS ...]]
                     [--backend {klip,pyklip,vip}] [--obj OBJ [OBJ ...]]
                     [--frames FRAMES] [--binned] [--pre-bin PRE_BIN]
                     [--parang-sign PARANG_SIGN] [--truenorth TRUENORTH]
                     [--fwhm-px FWHM_PX] [--name NAME] [--r-ee R_EE]
                     [--groups GROUPS [GROUPS ...]]
                     [--group-min-frames GROUP_MIN_FRAMES]
                     [--group-smooth GROUP_SMOOTH] [--max-frames MAX_FRAMES]
                     [--workers WORKERS] [--pool {auto,processes,threads}]
                     [--threads-per-job THREADS_PER_JOB]
                     [--weighting {equal,sqrt_texp}] [--fast]
                     [--klip-gram {target,zone}] [--no-library]
                     [--no-clean-subtract] [--ladder-pair]
                     [--n-min-ref N_MIN_REF] [--ref-frac REF_FRAC]
                     [--bin-min N] [--bin-max N] [--known RHO PA]
                     [--cube-name NAME] [--subtract-sources FILE]
                     [--subtract-line FILE] [--inject-sources FILE]
                     [--metric {mawet,fmmf}] [--run-dir RUN_DIR]
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
                     [--param-verify] [--no-param-verify] [--n-remeasure N]
                     [--n-sources N [N ...]] [--min-r-fwhm X] [--n-pv N_PV]
                     [--pv-divmin PV_DIVMIN] [--candidates]
                     [--write-setup-files] [--no-setup-files]
                     [--legacy-stitch] [--batch BATCH]
                     [--batch-mode {pipeline,sync}]
                     [--edge-padding {between,all}] [--display] [--no-display]
                     [--display-every DISPLAY_EVERY] [--pdf-every PDF_EVERY]
                     [--movie-every MOVIE_EVERY] [--show [SHOW]] [--aliens]
                     [--window-scale WINDOW_SCALE] [--no-fm-curve]
                     [--fm-preview] [--no-fm-preview]
                     [--no-liveness-check | --strict-liveness]

options:
  -h, --help            show this help message and exit
  --instrument {near,nomic,generic,ales}
                        near: NEAR campaign tree under --root; nomic: pyNOMIC
                        working directory under --root; generic: any
                        registered cube (--cube/--angles/--psf ...); ales: an
                        LBTI/ALES reduction's products under --root,
                        wavelength groups as partitions
  --root ROOT [ROOT ...]
                        NEAR: data root. nomic: pyNOMIC working directory (one
                        per --obj, or one for all). generic: optional base
                        directory for the run directory. ales: the reduction's
                        products directory, read only (the run goes to --run-
                        dir), or --ales-dataset
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
  --threads-per-job THREADS_PER_JOB
                        threads per partition job (default 4). The pool gets
                        workers / threads-per-job processes, never fewer than
                        min(workers, 2 x partitions)
  --weighting {equal,sqrt_texp}
  --fast                single-basis KLIP (angsep/anglemax not searched)
  --klip-gram {target,zone}
                        how the per-target KLIP forms each target's reference
                        covariance. 'target' (default) works from its
                        references' pixels and forms all modes. 'zone' reads
                        it off one Gram matrix per zone and projects the
                        target without forming the modes, the same KLIP to
                        float64 rounding but not to the bit, and about twice
                        as fast per target at small bins. A run keeps the mode
                        it was started or last resumed with (its
                        klip_gram.json) unless this is given
  --no-library
  --no-clean-subtract
  --ladder-pair         two-source annulus: use the pre-2026-09-05 two-rung
                        radial ladder instead of both sources at the annulus'
                        area-weighted mid radius (only to reproduce older IDL
                        runs)
  --n-min-ref N_MIN_REF
  --ref-frac REF_FRAC
  --bin-min N           generic / ales: floor of the temporal bin, as a
                        projection (MinBinGuard): a draw below it is evaluated
                        at N frames per bin. Cost, not correctness: on a long
                        sequence bin 1 costs tens of times more than bin 5
                        (default 1 = no floor)
  --bin-max N           generic / ales: top of the searched temporal-bin
                        range, frames per bin (default: ales 30, generic
                        nframes / 8). A winner sitting on it
                        (distance_to_bounds bin = 0) wants it higher
  --known RHO PA
  --cube-name NAME      near: the cube read in each night's directory (default
                        AB_cube_skysub_cen_clean.fits, or AB_cube_crop150.fits
                        where only that is there), e.g. a reduction's
                        AB_cube_skysub_cen_clean_abline.fits. Recorded in the
                        run's checkpoint, and kept on resume and extend when
                        not typed
  --subtract-sources FILE
                        point sources taken out of every frame once, at load:
                        one per line, rho (arcsec), PA (deg) and contrast,
                        each injected with the negative of its contrast by the
                        injection code, so placement, true north, centre, flux
                        unit and throughput are a draw's. Every reduction sees
                        the edited frames, the KLIP-FM model cube does not.
                        Recorded in the run's checkpoint, and kept on resume
                        and extend when not typed
  --subtract-line FILE  a line through the star taken out of every frame once,
                        at load, from a node file: 'pa', 'step' (px) and
                        optional 'rho_max' lines, then |s| (arcsec) with the
                        ridge's peak contrast on each half-line. The chain is
                        a source every step px from the first node to rho_max
                        (default the crop edge), each at a(s) step / L, with L
                        the injection PSF's line integral; as --subtract-
                        sources otherwise (klip-tpe fit-chain --line-nodes
                        writes such a file)
  --inject-sources FILE
                        point sources added to every frame once, at load,
                        after any subtraction: one per line, rho (arcsec), PA
                        (deg) and contrast, by the code a draw's injection
                        uses, e.g. planets put on a line to see what its
                        subtraction does to them. Every reduction sees them,
                        the KLIP-FM model cube does not. Recorded in the run's
                        checkpoint, and kept on resume and extend when not
                        typed
  --metric {mawet,fmmf}
                        detection metric: 'mawet' (default) filters with the
                        injection PSF; 'fmmf' forward-models that PSF through
                        each configuration's own subtraction and filters with
                        the result (klip_tpe.fmmf)
  --run-dir RUN_DIR     output directory (default:
                        <root>/comb/opt/run_YYYYMMDD_HHMMSS, the IDL layout;
                        generic and ales: ./runs/run_...)
  --ann-edges ANN_EDGES [ANN_EDGES ...]
  --n-iter N_ITER [N_ITER ...]
  --n-init N_INIT [N_INIT ...]
  --mode {tpe,random,grid}
  --blocks BLOCKS       TPE density model: univariate (default; the IDL
                        reference) | partitions | full
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
  --n-remeasure N       score each trial as the MEAN of N independent
                        injection draws (default 1). The objective's per-draw
                        scatter is ~0.9 S/N on typical data, so N=3 divides it
                        by sqrt(3) and roughly halves the optimism of the
                        reported best; see docs/BUDGET.md
  --n-sources N [N ...]
                        injected sources per annulus, the last value repeating
                        (default: klip-tpe's rule by the annulus' outer edge,
                        and 0 asks for the rule in that annulus). The cap that
                        keeps each source's noise ring clean still applies,
                        and the log says when it binds
  --min-r-fwhm X        no injection inside X FWHM of the star, in every stage
                        (default 1.5). On NEAR 1.0 FWHM is 1.0 lambda/D to
                        half a percent
  --n-pv N_PV
  --pv-divmin PV_DIVMIN
  --candidates          blind candidate search on the running/final stitch
  --write-setup-files
  --no-setup-files
  --legacy-stitch       equal-weight stitch, no contrast-curve seam trim
  --batch BATCH         trials in flight at once (default 1). TPE then
                        proposes from a posterior a few trials old, which
                        changes the search's trajectory
  --batch-mode {pipeline,sync}
                        with --batch: send the next proposal as soon as the
                        oldest trial is in (pipeline, the default), or propose
                        whole batches from one posterior and wait for all of a
                        batch (sync)
  --edge-padding {between,all}
                        pad annulus edges only where two annuli meet (between,
                        the default) or on every edge including the innermost
                        and outermost (all, as the IDL does)
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
                        (1.0 = the IDL window, 1:1 pixels), shrunk to fit a
                        smaller screen; the window keeps this size
  --no-fm-curve         skip the KLIP-FM cross-check curve after each annulus
                        (A 9.1)
  --fm-preview          live KLIP-FM preview at each new best (A 9.3; one
                        extra reduction per new best; default on)
  --no-fm-preview
  --no-liveness-check   skip the pre-flight that moves each searched reduction
                        dimension alone and warns when one changes nothing
                        (klip_tpe.liveness; ~1 + ndim reductions, new runs
                        only)
  --strict-liveness     stop before the first evaluation when the pre-flight
                        finds a dimension that changes nothing, instead of
                        warning

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
                        star flux per cube, in the science frames' own units
                        and in the template's normalisation aperture; omit it
                        and the contrast axis is in template units
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
  --forbidden-pa PA HALFWIDTH
                        near / generic / ales: no injection within HALFWIDTH
                        deg of position angle PA (a known companion's azimuth,
                        a disk, a line through the star); may repeat
  --mask-line PA HALFWIDTH
                        near / generic / ales: leave the pixels within
                        HALFWIDTH px of the line through the star at position
                        angle PA (both sides) out of the noise apertures; may
                        repeat. Pair it with --forbidden-pa on the same PA and
                        PA+180, so injections stay off it too

LBTI/ALES (--instrument ales):
  --ales-dataset NAME   a public ALES data set in the release layout,
                        downloaded once into $KLIP_TPE_DATA
                        (klip_tpe.datasets: lbti_ales_mwc758), one partition
                        per night and wavelength group, in place of --root
  --ales-nights NIGHT [NIGHT ...]
                        with --ales-dataset: the nights to load, as dates
                        (20191113) or labels (n1), all by default
  --ales-target NAME    prefix of the product files under --root
                        (NAME_cube_cen_clean.fits ...; default WISPIT2)
  --ales-ref NAME       reference star whose products sit beside the target's
                        (HIP91748): its group means become each group's RDI
                        library (default: none, RDI off)
  --ales-reflib         search the reference star as a library: nkeep_psfref,
                        the reference means most correlated with each target
                        frame, searched from 0 (plain ADI, to the bit) to the
                        whole pool, on top of the ADI references (needs
                        --ales-ref)
  --ales-ref-bin N      the reference star's frames in consecutive means of N
                        before anything else (default 5; 1 = every frame)
  --ales-reflib-science {all,searched}
                        the science pool of the searched library: every frame
                        the ADI rule admits (default) or a searched count too,
                        nkeep_altroll, whose 0 is pure RDI
  --ales-sdi            ADI, SDI and RDI in one KLIP step: the other groups'
                        frames, rescaled to each group's wavelength, join its
                        library as a pool; nkeep_sdi (the rows most correlated
                        with each target frame among those the displacement
                        rule admits) is searched from 0 (no SDI) to --ales-
                        sdi-cap. A reference star given (--ales-ref) is the
                        library's third pool, searched. nkeep_sdi =
                        nkeep_psfref = 0 is plain ADI, to the bit
  --ales-sdi-cap N      most-correlated SDI rows a target frame may take, the
                        top of nkeep_sdi (default 100; the per-target cost
                        grows with the square of the basis)
  --ales-sdi-lenders {all,selected}
                        which groups lend SDI rows: every group, excluded ones
                        too (default), or only the groups the configuration
                        includes
  --ales-groups N       wavelength groups over the usable channels, equally
                        spaced in ln(lambda) (default 20; 15 = one R = 40
                        resolution element each)
  --ales-channels C0 C1
                        first and last usable channel, inclusive (default 6
                        88)
  --ales-only gNN [gNN ...]
                        load these groups only (checks and short runs)
  --ales-max-excluded N
                        the group flags may switch off at most N groups; a
                        proposal past that has groups switched back on,
                        nearest --ales-prefer-um first (default 10; -1 = no
                        bound)
  --ales-prefer-um UM   wavelength the group bound restores groups towards
                        (default 3.7)
  --ales-representative gNN
                        the group whose FWHM and matched filter stand for the
                        combined image (default: the group at the mean
                        wavelength)
  --ales-angle-offset DEG
                        added to LBT_PARA once: the derotation is CCW by
                        LBT_PARA + this (default 180)
  --ales-tag-r-in PX    inner radius of the frame-quality tags' annulus
                        (default 2 spaxels: the star's core and halo; 6, the
                        generic default, is background-dominated here)
  --ales-cache DIR      directory for the group cubes as .npy, written once
                        and read back after (a group mean reads the whole
                        cube, which may sit on a slow volume)
  --ales-selection {binary,two_slot}
                        group flags: one 0/1 dimension per group (default) or
                        --max-drop drop slots

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
usage: klip-tpe resume [-h] [--instrument {near,nomic,generic,ales}]
                       [--root ROOT [ROOT ...]] [--cube CUBE [CUBE ...]]
                       [--angles ANGLES [ANGLES ...]] [--psf PSF [PSF ...]]
                       [--ref-cube REF_CUBE [REF_CUBE ...]]
                       [--names NAMES [NAMES ...]]
                       [--star-flux STAR_FLUX [STAR_FLUX ...]]
                       [--pxscale PXSCALE] [--lam LAM] [--diam DIAM]
                       [--angle-sign ANGLE_SIGN] [--crop-half CROP_HALF]
                       [--center CENTER CENTER] [--wv-index WV_INDEX]
                       [--no-tags] [--forbidden-pa PA HALFWIDTH]
                       [--mask-line PA HALFWIDTH] [--ales-dataset NAME]
                       [--ales-nights NIGHT [NIGHT ...]] [--ales-target NAME]
                       [--ales-ref NAME] [--ales-reflib] [--ales-ref-bin N]
                       [--ales-reflib-science {all,searched}] [--ales-sdi]
                       [--ales-sdi-cap N] [--ales-sdi-lenders {all,selected}]
                       [--ales-groups N] [--ales-channels C0 C1]
                       [--ales-only gNN [gNN ...]] [--ales-max-excluded N]
                       [--ales-prefer-um UM] [--ales-representative gNN]
                       [--ales-angle-offset DEG] [--ales-tag-r-in PX]
                       [--ales-cache DIR] [--ales-selection {binary,two_slot}]
                       [--nights NIGHTS [NIGHTS ...]]
                       [--backend {klip,pyklip,vip}] [--obj OBJ [OBJ ...]]
                       [--frames FRAMES] [--binned] [--pre-bin PRE_BIN]
                       [--parang-sign PARANG_SIGN] [--truenorth TRUENORTH]
                       [--fwhm-px FWHM_PX] [--name NAME] [--r-ee R_EE]
                       [--groups GROUPS [GROUPS ...]]
                       [--group-min-frames GROUP_MIN_FRAMES]
                       [--group-smooth GROUP_SMOOTH] [--max-frames MAX_FRAMES]
                       [--workers WORKERS] [--pool {auto,processes,threads}]
                       [--threads-per-job THREADS_PER_JOB]
                       [--weighting {equal,sqrt_texp}] [--fast]
                       [--klip-gram {target,zone}] [--no-library]
                       [--no-clean-subtract] [--ladder-pair]
                       [--n-min-ref N_MIN_REF] [--ref-frac REF_FRAC]
                       [--bin-min N] [--bin-max N] [--known RHO PA]
                       [--cube-name NAME] [--subtract-sources FILE]
                       [--subtract-line FILE] [--inject-sources FILE]
                       [--metric {mawet,fmmf}] [--run-dir RUN_DIR] [--display]
                       [--no-display] [--display-every DISPLAY_EVERY]
                       [--pdf-every PDF_EVERY] [--movie-every MOVIE_EVERY]
                       [--show [SHOW]] [--aliens]
                       [--window-scale WINDOW_SCALE]

options:
  -h, --help            show this help message and exit
  --instrument {near,nomic,generic,ales}
                        near: NEAR campaign tree under --root; nomic: pyNOMIC
                        working directory under --root; generic: any
                        registered cube (--cube/--angles/--psf ...); ales: an
                        LBTI/ALES reduction's products under --root,
                        wavelength groups as partitions
  --root ROOT [ROOT ...]
                        NEAR: data root. nomic: pyNOMIC working directory (one
                        per --obj, or one for all). generic: optional base
                        directory for the run directory. ales: the reduction's
                        products directory, read only (the run goes to --run-
                        dir), or --ales-dataset
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
  --threads-per-job THREADS_PER_JOB
                        threads per partition job (default 4). The pool gets
                        workers / threads-per-job processes, never fewer than
                        min(workers, 2 x partitions)
  --weighting {equal,sqrt_texp}
  --fast                single-basis KLIP (angsep/anglemax not searched)
  --klip-gram {target,zone}
                        how the per-target KLIP forms each target's reference
                        covariance. 'target' (default) works from its
                        references' pixels and forms all modes. 'zone' reads
                        it off one Gram matrix per zone and projects the
                        target without forming the modes, the same KLIP to
                        float64 rounding but not to the bit, and about twice
                        as fast per target at small bins. A run keeps the mode
                        it was started or last resumed with (its
                        klip_gram.json) unless this is given
  --no-library
  --no-clean-subtract
  --ladder-pair         two-source annulus: use the pre-2026-09-05 two-rung
                        radial ladder instead of both sources at the annulus'
                        area-weighted mid radius (only to reproduce older IDL
                        runs)
  --n-min-ref N_MIN_REF
  --ref-frac REF_FRAC
  --bin-min N           generic / ales: floor of the temporal bin, as a
                        projection (MinBinGuard): a draw below it is evaluated
                        at N frames per bin. Cost, not correctness: on a long
                        sequence bin 1 costs tens of times more than bin 5
                        (default 1 = no floor)
  --bin-max N           generic / ales: top of the searched temporal-bin
                        range, frames per bin (default: ales 30, generic
                        nframes / 8). A winner sitting on it
                        (distance_to_bounds bin = 0) wants it higher
  --known RHO PA
  --cube-name NAME      near: the cube read in each night's directory (default
                        AB_cube_skysub_cen_clean.fits, or AB_cube_crop150.fits
                        where only that is there), e.g. a reduction's
                        AB_cube_skysub_cen_clean_abline.fits. Recorded in the
                        run's checkpoint, and kept on resume and extend when
                        not typed
  --subtract-sources FILE
                        point sources taken out of every frame once, at load:
                        one per line, rho (arcsec), PA (deg) and contrast,
                        each injected with the negative of its contrast by the
                        injection code, so placement, true north, centre, flux
                        unit and throughput are a draw's. Every reduction sees
                        the edited frames, the KLIP-FM model cube does not.
                        Recorded in the run's checkpoint, and kept on resume
                        and extend when not typed
  --subtract-line FILE  a line through the star taken out of every frame once,
                        at load, from a node file: 'pa', 'step' (px) and
                        optional 'rho_max' lines, then |s| (arcsec) with the
                        ridge's peak contrast on each half-line. The chain is
                        a source every step px from the first node to rho_max
                        (default the crop edge), each at a(s) step / L, with L
                        the injection PSF's line integral; as --subtract-
                        sources otherwise (klip-tpe fit-chain --line-nodes
                        writes such a file)
  --inject-sources FILE
                        point sources added to every frame once, at load,
                        after any subtraction: one per line, rho (arcsec), PA
                        (deg) and contrast, by the code a draw's injection
                        uses, e.g. planets put on a line to see what its
                        subtraction does to them. Every reduction sees them,
                        the KLIP-FM model cube does not. Recorded in the run's
                        checkpoint, and kept on resume and extend when not
                        typed
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
                        star flux per cube, in the science frames' own units
                        and in the template's normalisation aperture; omit it
                        and the contrast axis is in template units
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
  --forbidden-pa PA HALFWIDTH
                        near / generic / ales: no injection within HALFWIDTH
                        deg of position angle PA (a known companion's azimuth,
                        a disk, a line through the star); may repeat
  --mask-line PA HALFWIDTH
                        near / generic / ales: leave the pixels within
                        HALFWIDTH px of the line through the star at position
                        angle PA (both sides) out of the noise apertures; may
                        repeat. Pair it with --forbidden-pa on the same PA and
                        PA+180, so injections stay off it too

LBTI/ALES (--instrument ales):
  --ales-dataset NAME   a public ALES data set in the release layout,
                        downloaded once into $KLIP_TPE_DATA
                        (klip_tpe.datasets: lbti_ales_mwc758), one partition
                        per night and wavelength group, in place of --root
  --ales-nights NIGHT [NIGHT ...]
                        with --ales-dataset: the nights to load, as dates
                        (20191113) or labels (n1), all by default
  --ales-target NAME    prefix of the product files under --root
                        (NAME_cube_cen_clean.fits ...; default WISPIT2)
  --ales-ref NAME       reference star whose products sit beside the target's
                        (HIP91748): its group means become each group's RDI
                        library (default: none, RDI off)
  --ales-reflib         search the reference star as a library: nkeep_psfref,
                        the reference means most correlated with each target
                        frame, searched from 0 (plain ADI, to the bit) to the
                        whole pool, on top of the ADI references (needs
                        --ales-ref)
  --ales-ref-bin N      the reference star's frames in consecutive means of N
                        before anything else (default 5; 1 = every frame)
  --ales-reflib-science {all,searched}
                        the science pool of the searched library: every frame
                        the ADI rule admits (default) or a searched count too,
                        nkeep_altroll, whose 0 is pure RDI
  --ales-sdi            ADI, SDI and RDI in one KLIP step: the other groups'
                        frames, rescaled to each group's wavelength, join its
                        library as a pool; nkeep_sdi (the rows most correlated
                        with each target frame among those the displacement
                        rule admits) is searched from 0 (no SDI) to --ales-
                        sdi-cap. A reference star given (--ales-ref) is the
                        library's third pool, searched. nkeep_sdi =
                        nkeep_psfref = 0 is plain ADI, to the bit
  --ales-sdi-cap N      most-correlated SDI rows a target frame may take, the
                        top of nkeep_sdi (default 100; the per-target cost
                        grows with the square of the basis)
  --ales-sdi-lenders {all,selected}
                        which groups lend SDI rows: every group, excluded ones
                        too (default), or only the groups the configuration
                        includes
  --ales-groups N       wavelength groups over the usable channels, equally
                        spaced in ln(lambda) (default 20; 15 = one R = 40
                        resolution element each)
  --ales-channels C0 C1
                        first and last usable channel, inclusive (default 6
                        88)
  --ales-only gNN [gNN ...]
                        load these groups only (checks and short runs)
  --ales-max-excluded N
                        the group flags may switch off at most N groups; a
                        proposal past that has groups switched back on,
                        nearest --ales-prefer-um first (default 10; -1 = no
                        bound)
  --ales-prefer-um UM   wavelength the group bound restores groups towards
                        (default 3.7)
  --ales-representative gNN
                        the group whose FWHM and matched filter stand for the
                        combined image (default: the group at the mean
                        wavelength)
  --ales-angle-offset DEG
                        added to LBT_PARA once: the derotation is CCW by
                        LBT_PARA + this (default 180)
  --ales-tag-r-in PX    inner radius of the frame-quality tags' annulus
                        (default 2 spaxels: the star's core and halo; 6, the
                        generic default, is background-dominated here)
  --ales-cache DIR      directory for the group cubes as .npy, written once
                        and read back after (a group mean reads the whole
                        cube, which may sit on a slow volume)
  --ales-selection {binary,two_slot}
                        group flags: one 0/1 dimension per group (default) or
                        --max-drop drop slots

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
usage: klip-tpe extend [-h] [--instrument {near,nomic,generic,ales}]
                       [--root ROOT [ROOT ...]] [--cube CUBE [CUBE ...]]
                       [--angles ANGLES [ANGLES ...]] [--psf PSF [PSF ...]]
                       [--ref-cube REF_CUBE [REF_CUBE ...]]
                       [--names NAMES [NAMES ...]]
                       [--star-flux STAR_FLUX [STAR_FLUX ...]]
                       [--pxscale PXSCALE] [--lam LAM] [--diam DIAM]
                       [--angle-sign ANGLE_SIGN] [--crop-half CROP_HALF]
                       [--center CENTER CENTER] [--wv-index WV_INDEX]
                       [--no-tags] [--forbidden-pa PA HALFWIDTH]
                       [--mask-line PA HALFWIDTH] [--ales-dataset NAME]
                       [--ales-nights NIGHT [NIGHT ...]] [--ales-target NAME]
                       [--ales-ref NAME] [--ales-reflib] [--ales-ref-bin N]
                       [--ales-reflib-science {all,searched}] [--ales-sdi]
                       [--ales-sdi-cap N] [--ales-sdi-lenders {all,selected}]
                       [--ales-groups N] [--ales-channels C0 C1]
                       [--ales-only gNN [gNN ...]] [--ales-max-excluded N]
                       [--ales-prefer-um UM] [--ales-representative gNN]
                       [--ales-angle-offset DEG] [--ales-tag-r-in PX]
                       [--ales-cache DIR] [--ales-selection {binary,two_slot}]
                       [--nights NIGHTS [NIGHTS ...]]
                       [--backend {klip,pyklip,vip}] [--obj OBJ [OBJ ...]]
                       [--frames FRAMES] [--binned] [--pre-bin PRE_BIN]
                       [--parang-sign PARANG_SIGN] [--truenorth TRUENORTH]
                       [--fwhm-px FWHM_PX] [--name NAME] [--r-ee R_EE]
                       [--groups GROUPS [GROUPS ...]]
                       [--group-min-frames GROUP_MIN_FRAMES]
                       [--group-smooth GROUP_SMOOTH] [--max-frames MAX_FRAMES]
                       [--workers WORKERS] [--pool {auto,processes,threads}]
                       [--threads-per-job THREADS_PER_JOB]
                       [--weighting {equal,sqrt_texp}] [--fast]
                       [--klip-gram {target,zone}] [--no-library]
                       [--no-clean-subtract] [--ladder-pair]
                       [--n-min-ref N_MIN_REF] [--ref-frac REF_FRAC]
                       [--bin-min N] [--bin-max N] [--known RHO PA]
                       [--cube-name NAME] [--subtract-sources FILE]
                       [--subtract-line FILE] [--inject-sources FILE]
                       [--metric {mawet,fmmf}] --run-dir RUN_DIR --n-iter
                       N_ITER [N_ITER ...]

options:
  -h, --help            show this help message and exit
  --instrument {near,nomic,generic,ales}
                        near: NEAR campaign tree under --root; nomic: pyNOMIC
                        working directory under --root; generic: any
                        registered cube (--cube/--angles/--psf ...); ales: an
                        LBTI/ALES reduction's products under --root,
                        wavelength groups as partitions
  --root ROOT [ROOT ...]
                        NEAR: data root. nomic: pyNOMIC working directory (one
                        per --obj, or one for all). generic: optional base
                        directory for the run directory. ales: the reduction's
                        products directory, read only (the run goes to --run-
                        dir), or --ales-dataset
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
  --threads-per-job THREADS_PER_JOB
                        threads per partition job (default 4). The pool gets
                        workers / threads-per-job processes, never fewer than
                        min(workers, 2 x partitions)
  --weighting {equal,sqrt_texp}
  --fast                single-basis KLIP (angsep/anglemax not searched)
  --klip-gram {target,zone}
                        how the per-target KLIP forms each target's reference
                        covariance. 'target' (default) works from its
                        references' pixels and forms all modes. 'zone' reads
                        it off one Gram matrix per zone and projects the
                        target without forming the modes, the same KLIP to
                        float64 rounding but not to the bit, and about twice
                        as fast per target at small bins. A run keeps the mode
                        it was started or last resumed with (its
                        klip_gram.json) unless this is given
  --no-library
  --no-clean-subtract
  --ladder-pair         two-source annulus: use the pre-2026-09-05 two-rung
                        radial ladder instead of both sources at the annulus'
                        area-weighted mid radius (only to reproduce older IDL
                        runs)
  --n-min-ref N_MIN_REF
  --ref-frac REF_FRAC
  --bin-min N           generic / ales: floor of the temporal bin, as a
                        projection (MinBinGuard): a draw below it is evaluated
                        at N frames per bin. Cost, not correctness: on a long
                        sequence bin 1 costs tens of times more than bin 5
                        (default 1 = no floor)
  --bin-max N           generic / ales: top of the searched temporal-bin
                        range, frames per bin (default: ales 30, generic
                        nframes / 8). A winner sitting on it
                        (distance_to_bounds bin = 0) wants it higher
  --known RHO PA
  --cube-name NAME      near: the cube read in each night's directory (default
                        AB_cube_skysub_cen_clean.fits, or AB_cube_crop150.fits
                        where only that is there), e.g. a reduction's
                        AB_cube_skysub_cen_clean_abline.fits. Recorded in the
                        run's checkpoint, and kept on resume and extend when
                        not typed
  --subtract-sources FILE
                        point sources taken out of every frame once, at load:
                        one per line, rho (arcsec), PA (deg) and contrast,
                        each injected with the negative of its contrast by the
                        injection code, so placement, true north, centre, flux
                        unit and throughput are a draw's. Every reduction sees
                        the edited frames, the KLIP-FM model cube does not.
                        Recorded in the run's checkpoint, and kept on resume
                        and extend when not typed
  --subtract-line FILE  a line through the star taken out of every frame once,
                        at load, from a node file: 'pa', 'step' (px) and
                        optional 'rho_max' lines, then |s| (arcsec) with the
                        ridge's peak contrast on each half-line. The chain is
                        a source every step px from the first node to rho_max
                        (default the crop edge), each at a(s) step / L, with L
                        the injection PSF's line integral; as --subtract-
                        sources otherwise (klip-tpe fit-chain --line-nodes
                        writes such a file)
  --inject-sources FILE
                        point sources added to every frame once, at load,
                        after any subtraction: one per line, rho (arcsec), PA
                        (deg) and contrast, by the code a draw's injection
                        uses, e.g. planets put on a line to see what its
                        subtraction does to them. Every reduction sees them,
                        the KLIP-FM model cube does not. Recorded in the run's
                        checkpoint, and kept on resume and extend when not
                        typed
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
                        star flux per cube, in the science frames' own units
                        and in the template's normalisation aperture; omit it
                        and the contrast axis is in template units
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
  --forbidden-pa PA HALFWIDTH
                        near / generic / ales: no injection within HALFWIDTH
                        deg of position angle PA (a known companion's azimuth,
                        a disk, a line through the star); may repeat
  --mask-line PA HALFWIDTH
                        near / generic / ales: leave the pixels within
                        HALFWIDTH px of the line through the star at position
                        angle PA (both sides) out of the noise apertures; may
                        repeat. Pair it with --forbidden-pa on the same PA and
                        PA+180, so injections stay off it too

LBTI/ALES (--instrument ales):
  --ales-dataset NAME   a public ALES data set in the release layout,
                        downloaded once into $KLIP_TPE_DATA
                        (klip_tpe.datasets: lbti_ales_mwc758), one partition
                        per night and wavelength group, in place of --root
  --ales-nights NIGHT [NIGHT ...]
                        with --ales-dataset: the nights to load, as dates
                        (20191113) or labels (n1), all by default
  --ales-target NAME    prefix of the product files under --root
                        (NAME_cube_cen_clean.fits ...; default WISPIT2)
  --ales-ref NAME       reference star whose products sit beside the target's
                        (HIP91748): its group means become each group's RDI
                        library (default: none, RDI off)
  --ales-reflib         search the reference star as a library: nkeep_psfref,
                        the reference means most correlated with each target
                        frame, searched from 0 (plain ADI, to the bit) to the
                        whole pool, on top of the ADI references (needs
                        --ales-ref)
  --ales-ref-bin N      the reference star's frames in consecutive means of N
                        before anything else (default 5; 1 = every frame)
  --ales-reflib-science {all,searched}
                        the science pool of the searched library: every frame
                        the ADI rule admits (default) or a searched count too,
                        nkeep_altroll, whose 0 is pure RDI
  --ales-sdi            ADI, SDI and RDI in one KLIP step: the other groups'
                        frames, rescaled to each group's wavelength, join its
                        library as a pool; nkeep_sdi (the rows most correlated
                        with each target frame among those the displacement
                        rule admits) is searched from 0 (no SDI) to --ales-
                        sdi-cap. A reference star given (--ales-ref) is the
                        library's third pool, searched. nkeep_sdi =
                        nkeep_psfref = 0 is plain ADI, to the bit
  --ales-sdi-cap N      most-correlated SDI rows a target frame may take, the
                        top of nkeep_sdi (default 100; the per-target cost
                        grows with the square of the basis)
  --ales-sdi-lenders {all,selected}
                        which groups lend SDI rows: every group, excluded ones
                        too (default), or only the groups the configuration
                        includes
  --ales-groups N       wavelength groups over the usable channels, equally
                        spaced in ln(lambda) (default 20; 15 = one R = 40
                        resolution element each)
  --ales-channels C0 C1
                        first and last usable channel, inclusive (default 6
                        88)
  --ales-only gNN [gNN ...]
                        load these groups only (checks and short runs)
  --ales-max-excluded N
                        the group flags may switch off at most N groups; a
                        proposal past that has groups switched back on,
                        nearest --ales-prefer-um first (default 10; -1 = no
                        bound)
  --ales-prefer-um UM   wavelength the group bound restores groups towards
                        (default 3.7)
  --ales-representative gNN
                        the group whose FWHM and matched filter stand for the
                        combined image (default: the group at the mean
                        wavelength)
  --ales-angle-offset DEG
                        added to LBT_PARA once: the derotation is CCW by
                        LBT_PARA + this (default 180)
  --ales-tag-r-in PX    inner radius of the frame-quality tags' annulus
                        (default 2 spaxels: the star's core and halo; 6, the
                        generic default, is background-dominated here)
  --ales-cache DIR      directory for the group cubes as .npy, written once
                        and read back after (a group mean reads the whole
                        cube, which may sit on a slow volume)
  --ales-selection {binary,two_slot}
                        group flags: one 0/1 dimension per group (default) or
                        --max-drop drop slots

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

## `klip-tpe fit-chain`

```
usage: klip-tpe fit-chain [-h] [--instrument {near,nomic,generic,ales}]
                          [--root ROOT [ROOT ...]] [--cube CUBE [CUBE ...]]
                          [--angles ANGLES [ANGLES ...]] [--psf PSF [PSF ...]]
                          [--ref-cube REF_CUBE [REF_CUBE ...]]
                          [--names NAMES [NAMES ...]]
                          [--star-flux STAR_FLUX [STAR_FLUX ...]]
                          [--pxscale PXSCALE] [--lam LAM] [--diam DIAM]
                          [--angle-sign ANGLE_SIGN] [--crop-half CROP_HALF]
                          [--center CENTER CENTER] [--wv-index WV_INDEX]
                          [--no-tags] [--forbidden-pa PA HALFWIDTH]
                          [--mask-line PA HALFWIDTH] [--ales-dataset NAME]
                          [--ales-nights NIGHT [NIGHT ...]]
                          [--ales-target NAME] [--ales-ref NAME]
                          [--ales-reflib] [--ales-ref-bin N]
                          [--ales-reflib-science {all,searched}] [--ales-sdi]
                          [--ales-sdi-cap N]
                          [--ales-sdi-lenders {all,selected}]
                          [--ales-groups N] [--ales-channels C0 C1]
                          [--ales-only gNN [gNN ...]] [--ales-max-excluded N]
                          [--ales-prefer-um UM] [--ales-representative gNN]
                          [--ales-angle-offset DEG] [--ales-tag-r-in PX]
                          [--ales-cache DIR]
                          [--ales-selection {binary,two_slot}]
                          [--nights NIGHTS [NIGHTS ...]]
                          [--backend {klip,pyklip,vip}] [--obj OBJ [OBJ ...]]
                          [--frames FRAMES] [--binned] [--pre-bin PRE_BIN]
                          [--parang-sign PARANG_SIGN] [--truenorth TRUENORTH]
                          [--fwhm-px FWHM_PX] [--name NAME] [--r-ee R_EE]
                          [--groups GROUPS [GROUPS ...]]
                          [--group-min-frames GROUP_MIN_FRAMES]
                          [--group-smooth GROUP_SMOOTH]
                          [--max-frames MAX_FRAMES] [--workers WORKERS]
                          [--pool {auto,processes,threads}]
                          [--threads-per-job THREADS_PER_JOB]
                          [--weighting {equal,sqrt_texp}] [--fast]
                          [--klip-gram {target,zone}] [--no-library]
                          [--no-clean-subtract] [--ladder-pair]
                          [--n-min-ref N_MIN_REF] [--ref-frac REF_FRAC]
                          [--bin-min N] [--bin-max N] [--known RHO PA]
                          [--cube-name NAME] [--subtract-sources FILE]
                          [--subtract-line FILE] [--inject-sources FILE]
                          [--metric {mawet,fmmf}] [--run-dir RUN_DIR]
                          [--chain FILE] [--line-nodes FILE] [--per-night]
                          [--line-pa PA] [--halfwidth PX]
                          [--rho-range RMIN RMAX] [--mask RHO PA RADIUS]
                          [--mask-candidates TIERS] [--no-mask-known]
                          [--out DIR]

options:
  -h, --help            show this help message and exit
  --instrument {near,nomic,generic,ales}
                        near: NEAR campaign tree under --root; nomic: pyNOMIC
                        working directory under --root; generic: any
                        registered cube (--cube/--angles/--psf ...); ales: an
                        LBTI/ALES reduction's products under --root,
                        wavelength groups as partitions
  --root ROOT [ROOT ...]
                        NEAR: data root. nomic: pyNOMIC working directory (one
                        per --obj, or one for all). generic: optional base
                        directory for the run directory. ales: the reduction's
                        products directory, read only (the run goes to --run-
                        dir), or --ales-dataset
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
  --threads-per-job THREADS_PER_JOB
                        threads per partition job (default 4). The pool gets
                        workers / threads-per-job processes, never fewer than
                        min(workers, 2 x partitions)
  --weighting {equal,sqrt_texp}
  --fast                single-basis KLIP (angsep/anglemax not searched)
  --klip-gram {target,zone}
                        how the per-target KLIP forms each target's reference
                        covariance. 'target' (default) works from its
                        references' pixels and forms all modes. 'zone' reads
                        it off one Gram matrix per zone and projects the
                        target without forming the modes, the same KLIP to
                        float64 rounding but not to the bit, and about twice
                        as fast per target at small bins. A run keeps the mode
                        it was started or last resumed with (its
                        klip_gram.json) unless this is given
  --no-library
  --no-clean-subtract
  --ladder-pair         two-source annulus: use the pre-2026-09-05 two-rung
                        radial ladder instead of both sources at the annulus'
                        area-weighted mid radius (only to reproduce older IDL
                        runs)
  --n-min-ref N_MIN_REF
  --ref-frac REF_FRAC
  --bin-min N           generic / ales: floor of the temporal bin, as a
                        projection (MinBinGuard): a draw below it is evaluated
                        at N frames per bin. Cost, not correctness: on a long
                        sequence bin 1 costs tens of times more than bin 5
                        (default 1 = no floor)
  --bin-max N           generic / ales: top of the searched temporal-bin
                        range, frames per bin (default: ales 30, generic
                        nframes / 8). A winner sitting on it
                        (distance_to_bounds bin = 0) wants it higher
  --known RHO PA
  --cube-name NAME      near: the cube read in each night's directory (default
                        AB_cube_skysub_cen_clean.fits, or AB_cube_crop150.fits
                        where only that is there), e.g. a reduction's
                        AB_cube_skysub_cen_clean_abline.fits. Recorded in the
                        run's checkpoint, and kept on resume and extend when
                        not typed
  --subtract-sources FILE
                        point sources taken out of every frame once, at load:
                        one per line, rho (arcsec), PA (deg) and contrast,
                        each injected with the negative of its contrast by the
                        injection code, so placement, true north, centre, flux
                        unit and throughput are a draw's. Every reduction sees
                        the edited frames, the KLIP-FM model cube does not.
                        Recorded in the run's checkpoint, and kept on resume
                        and extend when not typed
  --subtract-line FILE  a line through the star taken out of every frame once,
                        at load, from a node file: 'pa', 'step' (px) and
                        optional 'rho_max' lines, then |s| (arcsec) with the
                        ridge's peak contrast on each half-line. The chain is
                        a source every step px from the first node to rho_max
                        (default the crop edge), each at a(s) step / L, with L
                        the injection PSF's line integral; as --subtract-
                        sources otherwise (klip-tpe fit-chain --line-nodes
                        writes such a file)
  --inject-sources FILE
                        point sources added to every frame once, at load,
                        after any subtraction: one per line, rho (arcsec), PA
                        (deg) and contrast, by the code a draw's injection
                        uses, e.g. planets put on a line to see what its
                        subtraction does to them. Every reduction sees them,
                        the KLIP-FM model cube does not. Recorded in the run's
                        checkpoint, and kept on resume and extend when not
                        typed
  --metric {mawet,fmmf}
                        detection metric: 'mawet' (default) filters with the
                        injection PSF; 'fmmf' forward-models that PSF through
                        each configuration's own subtraction and filters with
                        the result (klip_tpe.fmmf)
  --run-dir RUN_DIR     the finished run (default: the newest on record)
  --chain FILE          one source per line: rho (arcsec), PA (deg), contrast
                        per unit node value, and a node label; a source shared
                        by two nodes is on two lines. Without labels the whole
                        chain is one node and the fit scales it
  --line-nodes FILE     in place of --chain: a node file (as --subtract-line
                        reads), one basis per node and half-line, piecewise
                        linear and held at the last node to rho_max; the
                        fitted ridge contrasts go to line_fitted.txt, ready
                        for --subtract-line
  --per-night           then one amplitude per night for the fitted chain, its
                        shape held fixed (per_night.json)
  --line-pa PA          PA of the line the fit region follows (default: the
                        chain's own line)
  --halfwidth PX        fit the pixels within PX of the line (default 12)
  --rho-range RMIN RMAX
                        fit only the pixels RMIN-RMAX arcsec from the star,
                        e.g. 0.95 3.07 (default: the run's annuli)
  --mask RHO PA RADIUS  leave a circle out of the fit and the noise: RHO
                        (arcsec), PA (deg), RADIUS (px; 0 for 1.5 FWHM); may
                        repeat
  --mask-candidates TIERS
                        also leave out the run's candidates
                        (cand/candidates.txt) of these tiers, e.g. A or AB.
                        The line's own peaks can be listed as candidates, so
                        name only tiers that are not it
  --no-mask-known       keep the run's known sources in the fit
  --out DIR             output directory (default <run>_chainfit beside the
                        run, which is left as it is)

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
                        star flux per cube, in the science frames' own units
                        and in the template's normalisation aperture; omit it
                        and the contrast axis is in template units
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
  --forbidden-pa PA HALFWIDTH
                        near / generic / ales: no injection within HALFWIDTH
                        deg of position angle PA (a known companion's azimuth,
                        a disk, a line through the star); may repeat
  --mask-line PA HALFWIDTH
                        near / generic / ales: leave the pixels within
                        HALFWIDTH px of the line through the star at position
                        angle PA (both sides) out of the noise apertures; may
                        repeat. Pair it with --forbidden-pa on the same PA and
                        PA+180, so injections stay off it too

LBTI/ALES (--instrument ales):
  --ales-dataset NAME   a public ALES data set in the release layout,
                        downloaded once into $KLIP_TPE_DATA
                        (klip_tpe.datasets: lbti_ales_mwc758), one partition
                        per night and wavelength group, in place of --root
  --ales-nights NIGHT [NIGHT ...]
                        with --ales-dataset: the nights to load, as dates
                        (20191113) or labels (n1), all by default
  --ales-target NAME    prefix of the product files under --root
                        (NAME_cube_cen_clean.fits ...; default WISPIT2)
  --ales-ref NAME       reference star whose products sit beside the target's
                        (HIP91748): its group means become each group's RDI
                        library (default: none, RDI off)
  --ales-reflib         search the reference star as a library: nkeep_psfref,
                        the reference means most correlated with each target
                        frame, searched from 0 (plain ADI, to the bit) to the
                        whole pool, on top of the ADI references (needs
                        --ales-ref)
  --ales-ref-bin N      the reference star's frames in consecutive means of N
                        before anything else (default 5; 1 = every frame)
  --ales-reflib-science {all,searched}
                        the science pool of the searched library: every frame
                        the ADI rule admits (default) or a searched count too,
                        nkeep_altroll, whose 0 is pure RDI
  --ales-sdi            ADI, SDI and RDI in one KLIP step: the other groups'
                        frames, rescaled to each group's wavelength, join its
                        library as a pool; nkeep_sdi (the rows most correlated
                        with each target frame among those the displacement
                        rule admits) is searched from 0 (no SDI) to --ales-
                        sdi-cap. A reference star given (--ales-ref) is the
                        library's third pool, searched. nkeep_sdi =
                        nkeep_psfref = 0 is plain ADI, to the bit
  --ales-sdi-cap N      most-correlated SDI rows a target frame may take, the
                        top of nkeep_sdi (default 100; the per-target cost
                        grows with the square of the basis)
  --ales-sdi-lenders {all,selected}
                        which groups lend SDI rows: every group, excluded ones
                        too (default), or only the groups the configuration
                        includes
  --ales-groups N       wavelength groups over the usable channels, equally
                        spaced in ln(lambda) (default 20; 15 = one R = 40
                        resolution element each)
  --ales-channels C0 C1
                        first and last usable channel, inclusive (default 6
                        88)
  --ales-only gNN [gNN ...]
                        load these groups only (checks and short runs)
  --ales-max-excluded N
                        the group flags may switch off at most N groups; a
                        proposal past that has groups switched back on,
                        nearest --ales-prefer-um first (default 10; -1 = no
                        bound)
  --ales-prefer-um UM   wavelength the group bound restores groups towards
                        (default 3.7)
  --ales-representative gNN
                        the group whose FWHM and matched filter stand for the
                        combined image (default: the group at the mean
                        wavelength)
  --ales-angle-offset DEG
                        added to LBT_PARA once: the derotation is CCW by
                        LBT_PARA + this (default 180)
  --ales-tag-r-in PX    inner radius of the frame-quality tags' annulus
                        (default 2 spaxels: the star's core and halo; 6, the
                        generic default, is background-dominated here)
  --ales-cache DIR      directory for the group cubes as .npy, written once
                        and read back after (a group mean reads the whole
                        cube, which may sit on a slow volume)
  --ales-selection {binary,two_slot}
                        group flags: one 0/1 dimension per group (default) or
                        --max-drop drop slots

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

## `klip-tpe rereduce`

```
usage: klip-tpe rereduce [-h] [--instrument {near,nomic,generic,ales}]
                         [--root ROOT [ROOT ...]] [--cube CUBE [CUBE ...]]
                         [--angles ANGLES [ANGLES ...]] [--psf PSF [PSF ...]]
                         [--ref-cube REF_CUBE [REF_CUBE ...]]
                         [--names NAMES [NAMES ...]]
                         [--star-flux STAR_FLUX [STAR_FLUX ...]]
                         [--pxscale PXSCALE] [--lam LAM] [--diam DIAM]
                         [--angle-sign ANGLE_SIGN] [--crop-half CROP_HALF]
                         [--center CENTER CENTER] [--wv-index WV_INDEX]
                         [--no-tags] [--forbidden-pa PA HALFWIDTH]
                         [--mask-line PA HALFWIDTH] [--ales-dataset NAME]
                         [--ales-nights NIGHT [NIGHT ...]]
                         [--ales-target NAME] [--ales-ref NAME]
                         [--ales-reflib] [--ales-ref-bin N]
                         [--ales-reflib-science {all,searched}] [--ales-sdi]
                         [--ales-sdi-cap N]
                         [--ales-sdi-lenders {all,selected}] [--ales-groups N]
                         [--ales-channels C0 C1] [--ales-only gNN [gNN ...]]
                         [--ales-max-excluded N] [--ales-prefer-um UM]
                         [--ales-representative gNN] [--ales-angle-offset DEG]
                         [--ales-tag-r-in PX] [--ales-cache DIR]
                         [--ales-selection {binary,two_slot}]
                         [--nights NIGHTS [NIGHTS ...]]
                         [--backend {klip,pyklip,vip}] [--obj OBJ [OBJ ...]]
                         [--frames FRAMES] [--binned] [--pre-bin PRE_BIN]
                         [--parang-sign PARANG_SIGN] [--truenorth TRUENORTH]
                         [--fwhm-px FWHM_PX] [--name NAME] [--r-ee R_EE]
                         [--groups GROUPS [GROUPS ...]]
                         [--group-min-frames GROUP_MIN_FRAMES]
                         [--group-smooth GROUP_SMOOTH]
                         [--max-frames MAX_FRAMES] [--workers WORKERS]
                         [--pool {auto,processes,threads}]
                         [--threads-per-job THREADS_PER_JOB]
                         [--weighting {equal,sqrt_texp}] [--fast]
                         [--klip-gram {target,zone}] [--no-library]
                         [--no-clean-subtract] [--ladder-pair]
                         [--n-min-ref N_MIN_REF] [--ref-frac REF_FRAC]
                         [--bin-min N] [--bin-max N] [--known RHO PA]
                         [--cube-name NAME] [--subtract-sources FILE]
                         [--subtract-line FILE] [--inject-sources FILE]
                         [--metric {mawet,fmmf}] [--run-dir RUN_DIR]
                         [--verify-sources FILE] [--no-verify] [--out DIR]
                         [--overwrite]

options:
  -h, --help            show this help message and exit
  --instrument {near,nomic,generic,ales}
                        near: NEAR campaign tree under --root; nomic: pyNOMIC
                        working directory under --root; generic: any
                        registered cube (--cube/--angles/--psf ...); ales: an
                        LBTI/ALES reduction's products under --root,
                        wavelength groups as partitions
  --root ROOT [ROOT ...]
                        NEAR: data root. nomic: pyNOMIC working directory (one
                        per --obj, or one for all). generic: optional base
                        directory for the run directory. ales: the reduction's
                        products directory, read only (the run goes to --run-
                        dir), or --ales-dataset
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
  --threads-per-job THREADS_PER_JOB
                        threads per partition job (default 4). The pool gets
                        workers / threads-per-job processes, never fewer than
                        min(workers, 2 x partitions)
  --weighting {equal,sqrt_texp}
  --fast                single-basis KLIP (angsep/anglemax not searched)
  --klip-gram {target,zone}
                        how the per-target KLIP forms each target's reference
                        covariance. 'target' (default) works from its
                        references' pixels and forms all modes. 'zone' reads
                        it off one Gram matrix per zone and projects the
                        target without forming the modes, the same KLIP to
                        float64 rounding but not to the bit, and about twice
                        as fast per target at small bins. A run keeps the mode
                        it was started or last resumed with (its
                        klip_gram.json) unless this is given
  --no-library
  --no-clean-subtract
  --ladder-pair         two-source annulus: use the pre-2026-09-05 two-rung
                        radial ladder instead of both sources at the annulus'
                        area-weighted mid radius (only to reproduce older IDL
                        runs)
  --n-min-ref N_MIN_REF
  --ref-frac REF_FRAC
  --bin-min N           generic / ales: floor of the temporal bin, as a
                        projection (MinBinGuard): a draw below it is evaluated
                        at N frames per bin. Cost, not correctness: on a long
                        sequence bin 1 costs tens of times more than bin 5
                        (default 1 = no floor)
  --bin-max N           generic / ales: top of the searched temporal-bin
                        range, frames per bin (default: ales 30, generic
                        nframes / 8). A winner sitting on it
                        (distance_to_bounds bin = 0) wants it higher
  --known RHO PA
  --cube-name NAME      near: the cube read in each night's directory (default
                        AB_cube_skysub_cen_clean.fits, or AB_cube_crop150.fits
                        where only that is there), e.g. a reduction's
                        AB_cube_skysub_cen_clean_abline.fits. Recorded in the
                        run's checkpoint, and kept on resume and extend when
                        not typed
  --subtract-sources FILE
                        point sources taken out of every frame once, at load:
                        one per line, rho (arcsec), PA (deg) and contrast,
                        each injected with the negative of its contrast by the
                        injection code, so placement, true north, centre, flux
                        unit and throughput are a draw's. Every reduction sees
                        the edited frames, the KLIP-FM model cube does not.
                        Recorded in the run's checkpoint, and kept on resume
                        and extend when not typed
  --subtract-line FILE  a line through the star taken out of every frame once,
                        at load, from a node file: 'pa', 'step' (px) and
                        optional 'rho_max' lines, then |s| (arcsec) with the
                        ridge's peak contrast on each half-line. The chain is
                        a source every step px from the first node to rho_max
                        (default the crop edge), each at a(s) step / L, with L
                        the injection PSF's line integral; as --subtract-
                        sources otherwise (klip-tpe fit-chain --line-nodes
                        writes such a file)
  --inject-sources FILE
                        point sources added to every frame once, at load,
                        after any subtraction: one per line, rho (arcsec), PA
                        (deg) and contrast, by the code a draw's injection
                        uses, e.g. planets put on a line to see what its
                        subtraction does to them. Every reduction sees them,
                        the KLIP-FM model cube does not. Recorded in the run's
                        checkpoint, and kept on resume and extend when not
                        typed
  --metric {mawet,fmmf}
                        detection metric: 'mawet' (default) filters with the
                        injection PSF; 'fmmf' forward-models that PSF through
                        each configuration's own subtraction and filters with
                        the result (klip_tpe.fmmf)
  --run-dir RUN_DIR     the finished run (default: the newest on record)
  --verify-sources FILE
                        the final verify's injections: one per line, rho
                        (arcsec), PA (deg) and contrast, all of them in every
                        annulus' injected reduction (default: the winners' own
                        validation injections, as in the run)
  --no-verify           the stitch and its S/N map only
  --out DIR             output directory (default <run>_rereduce beside the
                        run, which is left as it is)
  --overwrite           write over an earlier re-reduction in --out

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
                        star flux per cube, in the science frames' own units
                        and in the template's normalisation aperture; omit it
                        and the contrast axis is in template units
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
  --forbidden-pa PA HALFWIDTH
                        near / generic / ales: no injection within HALFWIDTH
                        deg of position angle PA (a known companion's azimuth,
                        a disk, a line through the star); may repeat
  --mask-line PA HALFWIDTH
                        near / generic / ales: leave the pixels within
                        HALFWIDTH px of the line through the star at position
                        angle PA (both sides) out of the noise apertures; may
                        repeat. Pair it with --forbidden-pa on the same PA and
                        PA+180, so injections stay off it too

LBTI/ALES (--instrument ales):
  --ales-dataset NAME   a public ALES data set in the release layout,
                        downloaded once into $KLIP_TPE_DATA
                        (klip_tpe.datasets: lbti_ales_mwc758), one partition
                        per night and wavelength group, in place of --root
  --ales-nights NIGHT [NIGHT ...]
                        with --ales-dataset: the nights to load, as dates
                        (20191113) or labels (n1), all by default
  --ales-target NAME    prefix of the product files under --root
                        (NAME_cube_cen_clean.fits ...; default WISPIT2)
  --ales-ref NAME       reference star whose products sit beside the target's
                        (HIP91748): its group means become each group's RDI
                        library (default: none, RDI off)
  --ales-reflib         search the reference star as a library: nkeep_psfref,
                        the reference means most correlated with each target
                        frame, searched from 0 (plain ADI, to the bit) to the
                        whole pool, on top of the ADI references (needs
                        --ales-ref)
  --ales-ref-bin N      the reference star's frames in consecutive means of N
                        before anything else (default 5; 1 = every frame)
  --ales-reflib-science {all,searched}
                        the science pool of the searched library: every frame
                        the ADI rule admits (default) or a searched count too,
                        nkeep_altroll, whose 0 is pure RDI
  --ales-sdi            ADI, SDI and RDI in one KLIP step: the other groups'
                        frames, rescaled to each group's wavelength, join its
                        library as a pool; nkeep_sdi (the rows most correlated
                        with each target frame among those the displacement
                        rule admits) is searched from 0 (no SDI) to --ales-
                        sdi-cap. A reference star given (--ales-ref) is the
                        library's third pool, searched. nkeep_sdi =
                        nkeep_psfref = 0 is plain ADI, to the bit
  --ales-sdi-cap N      most-correlated SDI rows a target frame may take, the
                        top of nkeep_sdi (default 100; the per-target cost
                        grows with the square of the basis)
  --ales-sdi-lenders {all,selected}
                        which groups lend SDI rows: every group, excluded ones
                        too (default), or only the groups the configuration
                        includes
  --ales-groups N       wavelength groups over the usable channels, equally
                        spaced in ln(lambda) (default 20; 15 = one R = 40
                        resolution element each)
  --ales-channels C0 C1
                        first and last usable channel, inclusive (default 6
                        88)
  --ales-only gNN [gNN ...]
                        load these groups only (checks and short runs)
  --ales-max-excluded N
                        the group flags may switch off at most N groups; a
                        proposal past that has groups switched back on,
                        nearest --ales-prefer-um first (default 10; -1 = no
                        bound)
  --ales-prefer-um UM   wavelength the group bound restores groups towards
                        (default 3.7)
  --ales-representative gNN
                        the group whose FWHM and matched filter stand for the
                        combined image (default: the group at the mean
                        wavelength)
  --ales-angle-offset DEG
                        added to LBT_PARA once: the derotation is CCW by
                        LBT_PARA + this (default 180)
  --ales-tag-r-in PX    inner radius of the frame-quality tags' annulus
                        (default 2 spaxels: the star's core and halo; 6, the
                        generic default, is background-dominated here)
  --ales-cache DIR      directory for the group cubes as .npy, written once
                        and read back after (a group mean reads the whole
                        cube, which may sit on a slow volume)
  --ales-selection {binary,two_slot}
                        group flags: one 0/1 dimension per group (default) or
                        --max-drop drop slots

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

## `klip-tpe render`

```
usage: klip-tpe render [-h] --run-dir RUN_DIR [--every EVERY]
                       [--annulus ANNULUS [ANNULUS ...]] [--dpi DPI]
                       [--out OUT] [--no-movie]

options:
  -h, --help            show this help message and exit
  --run-dir RUN_DIR
  --every EVERY         every Nth evaluation of each annulus (its last is
                        always drawn)
  --annulus ANNULUS [ANNULUS ...]
                        only these annuli (1-based)
  --dpi DPI             text scale of the panel; 100 = the live panel's
  --out OUT             output directory (default <run-dir>/steps_rebuilt; the
                        live steps/ and opt_steps.gif are left as they are)
  --no-movie            skip <out>/opt_steps.gif
```

## `klip-tpe replay`

```
usage: klip-tpe replay [-h] [--instrument {near,nomic,generic,ales}]
                       [--root ROOT [ROOT ...]] [--cube CUBE [CUBE ...]]
                       [--angles ANGLES [ANGLES ...]] [--psf PSF [PSF ...]]
                       [--ref-cube REF_CUBE [REF_CUBE ...]]
                       [--names NAMES [NAMES ...]]
                       [--star-flux STAR_FLUX [STAR_FLUX ...]]
                       [--pxscale PXSCALE] [--lam LAM] [--diam DIAM]
                       [--angle-sign ANGLE_SIGN] [--crop-half CROP_HALF]
                       [--center CENTER CENTER] [--wv-index WV_INDEX]
                       [--no-tags] [--forbidden-pa PA HALFWIDTH]
                       [--mask-line PA HALFWIDTH] [--ales-dataset NAME]
                       [--ales-nights NIGHT [NIGHT ...]] [--ales-target NAME]
                       [--ales-ref NAME] [--ales-reflib] [--ales-ref-bin N]
                       [--ales-reflib-science {all,searched}] [--ales-sdi]
                       [--ales-sdi-cap N] [--ales-sdi-lenders {all,selected}]
                       [--ales-groups N] [--ales-channels C0 C1]
                       [--ales-only gNN [gNN ...]] [--ales-max-excluded N]
                       [--ales-prefer-um UM] [--ales-representative gNN]
                       [--ales-angle-offset DEG] [--ales-tag-r-in PX]
                       [--ales-cache DIR] [--ales-selection {binary,two_slot}]
                       [--nights NIGHTS [NIGHTS ...]]
                       [--backend {klip,pyklip,vip}] [--obj OBJ [OBJ ...]]
                       [--frames FRAMES] [--binned] [--pre-bin PRE_BIN]
                       [--parang-sign PARANG_SIGN] [--truenorth TRUENORTH]
                       [--fwhm-px FWHM_PX] [--name NAME] [--r-ee R_EE]
                       [--groups GROUPS [GROUPS ...]]
                       [--group-min-frames GROUP_MIN_FRAMES]
                       [--group-smooth GROUP_SMOOTH] [--max-frames MAX_FRAMES]
                       [--workers WORKERS] [--pool {auto,processes,threads}]
                       [--threads-per-job THREADS_PER_JOB]
                       [--weighting {equal,sqrt_texp}] [--fast]
                       [--klip-gram {target,zone}] [--no-library]
                       [--no-clean-subtract] [--ladder-pair]
                       [--n-min-ref N_MIN_REF] [--ref-frac REF_FRAC]
                       [--bin-min N] [--bin-max N] [--known RHO PA]
                       [--cube-name NAME] [--subtract-sources FILE]
                       [--subtract-line FILE] [--inject-sources FILE]
                       [--metric {mawet,fmmf}] --idl-log IDL_LOG
                       [--run-dir RUN_DIR] [--n N] [--repeat REPEAT]
                       [--contrast CONTRAST]
                       [--ann-edges ANN_EDGES [ANN_EDGES ...]] [--k-max K_MAX]
                       [--anglemax-hi ANGLEMAX_HI] [--seed SEED]

options:
  -h, --help            show this help message and exit
  --instrument {near,nomic,generic,ales}
                        near: NEAR campaign tree under --root; nomic: pyNOMIC
                        working directory under --root; generic: any
                        registered cube (--cube/--angles/--psf ...); ales: an
                        LBTI/ALES reduction's products under --root,
                        wavelength groups as partitions
  --root ROOT [ROOT ...]
                        NEAR: data root. nomic: pyNOMIC working directory (one
                        per --obj, or one for all). generic: optional base
                        directory for the run directory. ales: the reduction's
                        products directory, read only (the run goes to --run-
                        dir), or --ales-dataset
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
  --threads-per-job THREADS_PER_JOB
                        threads per partition job (default 4). The pool gets
                        workers / threads-per-job processes, never fewer than
                        min(workers, 2 x partitions)
  --weighting {equal,sqrt_texp}
  --fast                single-basis KLIP (angsep/anglemax not searched)
  --klip-gram {target,zone}
                        how the per-target KLIP forms each target's reference
                        covariance. 'target' (default) works from its
                        references' pixels and forms all modes. 'zone' reads
                        it off one Gram matrix per zone and projects the
                        target without forming the modes, the same KLIP to
                        float64 rounding but not to the bit, and about twice
                        as fast per target at small bins. A run keeps the mode
                        it was started or last resumed with (its
                        klip_gram.json) unless this is given
  --no-library
  --no-clean-subtract
  --ladder-pair         two-source annulus: use the pre-2026-09-05 two-rung
                        radial ladder instead of both sources at the annulus'
                        area-weighted mid radius (only to reproduce older IDL
                        runs)
  --n-min-ref N_MIN_REF
  --ref-frac REF_FRAC
  --bin-min N           generic / ales: floor of the temporal bin, as a
                        projection (MinBinGuard): a draw below it is evaluated
                        at N frames per bin. Cost, not correctness: on a long
                        sequence bin 1 costs tens of times more than bin 5
                        (default 1 = no floor)
  --bin-max N           generic / ales: top of the searched temporal-bin
                        range, frames per bin (default: ales 30, generic
                        nframes / 8). A winner sitting on it
                        (distance_to_bounds bin = 0) wants it higher
  --known RHO PA
  --cube-name NAME      near: the cube read in each night's directory (default
                        AB_cube_skysub_cen_clean.fits, or AB_cube_crop150.fits
                        where only that is there), e.g. a reduction's
                        AB_cube_skysub_cen_clean_abline.fits. Recorded in the
                        run's checkpoint, and kept on resume and extend when
                        not typed
  --subtract-sources FILE
                        point sources taken out of every frame once, at load:
                        one per line, rho (arcsec), PA (deg) and contrast,
                        each injected with the negative of its contrast by the
                        injection code, so placement, true north, centre, flux
                        unit and throughput are a draw's. Every reduction sees
                        the edited frames, the KLIP-FM model cube does not.
                        Recorded in the run's checkpoint, and kept on resume
                        and extend when not typed
  --subtract-line FILE  a line through the star taken out of every frame once,
                        at load, from a node file: 'pa', 'step' (px) and
                        optional 'rho_max' lines, then |s| (arcsec) with the
                        ridge's peak contrast on each half-line. The chain is
                        a source every step px from the first node to rho_max
                        (default the crop edge), each at a(s) step / L, with L
                        the injection PSF's line integral; as --subtract-
                        sources otherwise (klip-tpe fit-chain --line-nodes
                        writes such a file)
  --inject-sources FILE
                        point sources added to every frame once, at load,
                        after any subtraction: one per line, rho (arcsec), PA
                        (deg) and contrast, by the code a draw's injection
                        uses, e.g. planets put on a line to see what its
                        subtraction does to them. Every reduction sees them,
                        the KLIP-FM model cube does not. Recorded in the run's
                        checkpoint, and kept on resume and extend when not
                        typed
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
                        star flux per cube, in the science frames' own units
                        and in the template's normalisation aperture; omit it
                        and the contrast axis is in template units
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
  --forbidden-pa PA HALFWIDTH
                        near / generic / ales: no injection within HALFWIDTH
                        deg of position angle PA (a known companion's azimuth,
                        a disk, a line through the star); may repeat
  --mask-line PA HALFWIDTH
                        near / generic / ales: leave the pixels within
                        HALFWIDTH px of the line through the star at position
                        angle PA (both sides) out of the noise apertures; may
                        repeat. Pair it with --forbidden-pa on the same PA and
                        PA+180, so injections stay off it too

LBTI/ALES (--instrument ales):
  --ales-dataset NAME   a public ALES data set in the release layout,
                        downloaded once into $KLIP_TPE_DATA
                        (klip_tpe.datasets: lbti_ales_mwc758), one partition
                        per night and wavelength group, in place of --root
  --ales-nights NIGHT [NIGHT ...]
                        with --ales-dataset: the nights to load, as dates
                        (20191113) or labels (n1), all by default
  --ales-target NAME    prefix of the product files under --root
                        (NAME_cube_cen_clean.fits ...; default WISPIT2)
  --ales-ref NAME       reference star whose products sit beside the target's
                        (HIP91748): its group means become each group's RDI
                        library (default: none, RDI off)
  --ales-reflib         search the reference star as a library: nkeep_psfref,
                        the reference means most correlated with each target
                        frame, searched from 0 (plain ADI, to the bit) to the
                        whole pool, on top of the ADI references (needs
                        --ales-ref)
  --ales-ref-bin N      the reference star's frames in consecutive means of N
                        before anything else (default 5; 1 = every frame)
  --ales-reflib-science {all,searched}
                        the science pool of the searched library: every frame
                        the ADI rule admits (default) or a searched count too,
                        nkeep_altroll, whose 0 is pure RDI
  --ales-sdi            ADI, SDI and RDI in one KLIP step: the other groups'
                        frames, rescaled to each group's wavelength, join its
                        library as a pool; nkeep_sdi (the rows most correlated
                        with each target frame among those the displacement
                        rule admits) is searched from 0 (no SDI) to --ales-
                        sdi-cap. A reference star given (--ales-ref) is the
                        library's third pool, searched. nkeep_sdi =
                        nkeep_psfref = 0 is plain ADI, to the bit
  --ales-sdi-cap N      most-correlated SDI rows a target frame may take, the
                        top of nkeep_sdi (default 100; the per-target cost
                        grows with the square of the basis)
  --ales-sdi-lenders {all,selected}
                        which groups lend SDI rows: every group, excluded ones
                        too (default), or only the groups the configuration
                        includes
  --ales-groups N       wavelength groups over the usable channels, equally
                        spaced in ln(lambda) (default 20; 15 = one R = 40
                        resolution element each)
  --ales-channels C0 C1
                        first and last usable channel, inclusive (default 6
                        88)
  --ales-only gNN [gNN ...]
                        load these groups only (checks and short runs)
  --ales-max-excluded N
                        the group flags may switch off at most N groups; a
                        proposal past that has groups switched back on,
                        nearest --ales-prefer-um first (default 10; -1 = no
                        bound)
  --ales-prefer-um UM   wavelength the group bound restores groups towards
                        (default 3.7)
  --ales-representative gNN
                        the group whose FWHM and matched filter stand for the
                        combined image (default: the group at the mean
                        wavelength)
  --ales-angle-offset DEG
                        added to LBT_PARA once: the derotation is CCW by
                        LBT_PARA + this (default 180)
  --ales-tag-r-in PX    inner radius of the frame-quality tags' annulus
                        (default 2 spaxels: the star's core and halo; 6, the
                        generic default, is background-dominated here)
  --ales-cache DIR      directory for the group cubes as .npy, written once
                        and read back after (a group mean reads the whole
                        cube, which may sit on a slow volume)
  --ales-selection {binary,two_slot}
                        group flags: one 0/1 dimension per group (default) or
                        --max-drop drop slots

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
usage: klip-tpe compare [-h] [--instrument {near,nomic,generic,ales}]
                        [--root ROOT [ROOT ...]] [--cube CUBE [CUBE ...]]
                        [--angles ANGLES [ANGLES ...]] [--psf PSF [PSF ...]]
                        [--ref-cube REF_CUBE [REF_CUBE ...]]
                        [--names NAMES [NAMES ...]]
                        [--star-flux STAR_FLUX [STAR_FLUX ...]]
                        [--pxscale PXSCALE] [--lam LAM] [--diam DIAM]
                        [--angle-sign ANGLE_SIGN] [--crop-half CROP_HALF]
                        [--center CENTER CENTER] [--wv-index WV_INDEX]
                        [--no-tags] [--forbidden-pa PA HALFWIDTH]
                        [--mask-line PA HALFWIDTH] [--ales-dataset NAME]
                        [--ales-nights NIGHT [NIGHT ...]] [--ales-target NAME]
                        [--ales-ref NAME] [--ales-reflib] [--ales-ref-bin N]
                        [--ales-reflib-science {all,searched}] [--ales-sdi]
                        [--ales-sdi-cap N] [--ales-sdi-lenders {all,selected}]
                        [--ales-groups N] [--ales-channels C0 C1]
                        [--ales-only gNN [gNN ...]] [--ales-max-excluded N]
                        [--ales-prefer-um UM] [--ales-representative gNN]
                        [--ales-angle-offset DEG] [--ales-tag-r-in PX]
                        [--ales-cache DIR]
                        [--ales-selection {binary,two_slot}]
                        [--nights NIGHTS [NIGHTS ...]]
                        [--backend {klip,pyklip,vip}] [--obj OBJ [OBJ ...]]
                        [--frames FRAMES] [--binned] [--pre-bin PRE_BIN]
                        [--parang-sign PARANG_SIGN] [--truenorth TRUENORTH]
                        [--fwhm-px FWHM_PX] [--name NAME] [--r-ee R_EE]
                        [--groups GROUPS [GROUPS ...]]
                        [--group-min-frames GROUP_MIN_FRAMES]
                        [--group-smooth GROUP_SMOOTH]
                        [--max-frames MAX_FRAMES] [--workers WORKERS]
                        [--pool {auto,processes,threads}]
                        [--threads-per-job THREADS_PER_JOB]
                        [--weighting {equal,sqrt_texp}] [--fast]
                        [--klip-gram {target,zone}] [--no-library]
                        [--no-clean-subtract] [--ladder-pair]
                        [--n-min-ref N_MIN_REF] [--ref-frac REF_FRAC]
                        [--bin-min N] [--bin-max N] [--known RHO PA]
                        [--cube-name NAME] [--subtract-sources FILE]
                        [--subtract-line FILE] [--inject-sources FILE]
                        [--metric {mawet,fmmf}] --idl-run IDL_RUN --out OUT
                        [--annulus ANNULUS]
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
  --instrument {near,nomic,generic,ales}
                        near: NEAR campaign tree under --root; nomic: pyNOMIC
                        working directory under --root; generic: any
                        registered cube (--cube/--angles/--psf ...); ales: an
                        LBTI/ALES reduction's products under --root,
                        wavelength groups as partitions
  --root ROOT [ROOT ...]
                        NEAR: data root. nomic: pyNOMIC working directory (one
                        per --obj, or one for all). generic: optional base
                        directory for the run directory. ales: the reduction's
                        products directory, read only (the run goes to --run-
                        dir), or --ales-dataset
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
  --threads-per-job THREADS_PER_JOB
                        threads per partition job (default 4). The pool gets
                        workers / threads-per-job processes, never fewer than
                        min(workers, 2 x partitions)
  --weighting {equal,sqrt_texp}
  --fast                single-basis KLIP (angsep/anglemax not searched)
  --klip-gram {target,zone}
                        how the per-target KLIP forms each target's reference
                        covariance. 'target' (default) works from its
                        references' pixels and forms all modes. 'zone' reads
                        it off one Gram matrix per zone and projects the
                        target without forming the modes, the same KLIP to
                        float64 rounding but not to the bit, and about twice
                        as fast per target at small bins. A run keeps the mode
                        it was started or last resumed with (its
                        klip_gram.json) unless this is given
  --no-library
  --no-clean-subtract
  --ladder-pair         two-source annulus: use the pre-2026-09-05 two-rung
                        radial ladder instead of both sources at the annulus'
                        area-weighted mid radius (only to reproduce older IDL
                        runs)
  --n-min-ref N_MIN_REF
  --ref-frac REF_FRAC
  --bin-min N           generic / ales: floor of the temporal bin, as a
                        projection (MinBinGuard): a draw below it is evaluated
                        at N frames per bin. Cost, not correctness: on a long
                        sequence bin 1 costs tens of times more than bin 5
                        (default 1 = no floor)
  --bin-max N           generic / ales: top of the searched temporal-bin
                        range, frames per bin (default: ales 30, generic
                        nframes / 8). A winner sitting on it
                        (distance_to_bounds bin = 0) wants it higher
  --known RHO PA
  --cube-name NAME      near: the cube read in each night's directory (default
                        AB_cube_skysub_cen_clean.fits, or AB_cube_crop150.fits
                        where only that is there), e.g. a reduction's
                        AB_cube_skysub_cen_clean_abline.fits. Recorded in the
                        run's checkpoint, and kept on resume and extend when
                        not typed
  --subtract-sources FILE
                        point sources taken out of every frame once, at load:
                        one per line, rho (arcsec), PA (deg) and contrast,
                        each injected with the negative of its contrast by the
                        injection code, so placement, true north, centre, flux
                        unit and throughput are a draw's. Every reduction sees
                        the edited frames, the KLIP-FM model cube does not.
                        Recorded in the run's checkpoint, and kept on resume
                        and extend when not typed
  --subtract-line FILE  a line through the star taken out of every frame once,
                        at load, from a node file: 'pa', 'step' (px) and
                        optional 'rho_max' lines, then |s| (arcsec) with the
                        ridge's peak contrast on each half-line. The chain is
                        a source every step px from the first node to rho_max
                        (default the crop edge), each at a(s) step / L, with L
                        the injection PSF's line integral; as --subtract-
                        sources otherwise (klip-tpe fit-chain --line-nodes
                        writes such a file)
  --inject-sources FILE
                        point sources added to every frame once, at load,
                        after any subtraction: one per line, rho (arcsec), PA
                        (deg) and contrast, by the code a draw's injection
                        uses, e.g. planets put on a line to see what its
                        subtraction does to them. Every reduction sees them,
                        the KLIP-FM model cube does not. Recorded in the run's
                        checkpoint, and kept on resume and extend when not
                        typed
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
                        star flux per cube, in the science frames' own units
                        and in the template's normalisation aperture; omit it
                        and the contrast axis is in template units
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
  --forbidden-pa PA HALFWIDTH
                        near / generic / ales: no injection within HALFWIDTH
                        deg of position angle PA (a known companion's azimuth,
                        a disk, a line through the star); may repeat
  --mask-line PA HALFWIDTH
                        near / generic / ales: leave the pixels within
                        HALFWIDTH px of the line through the star at position
                        angle PA (both sides) out of the noise apertures; may
                        repeat. Pair it with --forbidden-pa on the same PA and
                        PA+180, so injections stay off it too

LBTI/ALES (--instrument ales):
  --ales-dataset NAME   a public ALES data set in the release layout,
                        downloaded once into $KLIP_TPE_DATA
                        (klip_tpe.datasets: lbti_ales_mwc758), one partition
                        per night and wavelength group, in place of --root
  --ales-nights NIGHT [NIGHT ...]
                        with --ales-dataset: the nights to load, as dates
                        (20191113) or labels (n1), all by default
  --ales-target NAME    prefix of the product files under --root
                        (NAME_cube_cen_clean.fits ...; default WISPIT2)
  --ales-ref NAME       reference star whose products sit beside the target's
                        (HIP91748): its group means become each group's RDI
                        library (default: none, RDI off)
  --ales-reflib         search the reference star as a library: nkeep_psfref,
                        the reference means most correlated with each target
                        frame, searched from 0 (plain ADI, to the bit) to the
                        whole pool, on top of the ADI references (needs
                        --ales-ref)
  --ales-ref-bin N      the reference star's frames in consecutive means of N
                        before anything else (default 5; 1 = every frame)
  --ales-reflib-science {all,searched}
                        the science pool of the searched library: every frame
                        the ADI rule admits (default) or a searched count too,
                        nkeep_altroll, whose 0 is pure RDI
  --ales-sdi            ADI, SDI and RDI in one KLIP step: the other groups'
                        frames, rescaled to each group's wavelength, join its
                        library as a pool; nkeep_sdi (the rows most correlated
                        with each target frame among those the displacement
                        rule admits) is searched from 0 (no SDI) to --ales-
                        sdi-cap. A reference star given (--ales-ref) is the
                        library's third pool, searched. nkeep_sdi =
                        nkeep_psfref = 0 is plain ADI, to the bit
  --ales-sdi-cap N      most-correlated SDI rows a target frame may take, the
                        top of nkeep_sdi (default 100; the per-target cost
                        grows with the square of the basis)
  --ales-sdi-lenders {all,selected}
                        which groups lend SDI rows: every group, excluded ones
                        too (default), or only the groups the configuration
                        includes
  --ales-groups N       wavelength groups over the usable channels, equally
                        spaced in ln(lambda) (default 20; 15 = one R = 40
                        resolution element each)
  --ales-channels C0 C1
                        first and last usable channel, inclusive (default 6
                        88)
  --ales-only gNN [gNN ...]
                        load these groups only (checks and short runs)
  --ales-max-excluded N
                        the group flags may switch off at most N groups; a
                        proposal past that has groups switched back on,
                        nearest --ales-prefer-um first (default 10; -1 = no
                        bound)
  --ales-prefer-um UM   wavelength the group bound restores groups towards
                        (default 3.7)
  --ales-representative gNN
                        the group whose FWHM and matched filter stand for the
                        combined image (default: the group at the mean
                        wavelength)
  --ales-angle-offset DEG
                        added to LBT_PARA once: the derotation is CCW by
                        LBT_PARA + this (default 180)
  --ales-tag-r-in PX    inner radius of the frame-quality tags' annulus
                        (default 2 spaxels: the star's core and halo; 6, the
                        generic default, is background-dominated here)
  --ales-cache DIR      directory for the group cubes as .npy, written once
                        and read back after (a group mean reads the whole
                        cube, which may sit on a slow volume)
  --ales-selection {binary,two_slot}
                        group flags: one 0/1 dimension per group (default) or
                        --max-drop drop slots

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
