# Plugging your own pipeline into klip-tpe

klip-tpe optimizes *reduction parameters* by injection–recovery: for every proposed
parameter vector it asks a **reducer** for a clean image and an image with synthetic
companions injected, scores the recovered S/N, and feeds that back to the TPE sampler.
Anything that can produce those two images can be optimized. There are three ways in,
from least to most work.

| you have | use | you write |
|---|---|---|
| a cube + angles and a function that PSF-subtracts a cube | `FunctionReducer` | one function |
| data in pyKLIP, VIP or spaceKLIP form | `backends.pyklip` / `backends.vip` / `backends.spaceklip` | nothing (see `docs/BACKENDS.md`) |
| a pipeline that must own the whole reduction (files on disk, its own injection, another language) | `ExternalReducer` | a `reduce(request)` method |

Everything downstream (calibration, TPE, validation, contrast curve, stitching, verify /
param_verify / candidates, display, checkpoint/resume, CLI `resume`/`extend`) is shared.

## 1. Data: the `Dataset`

```python
from klip_tpe import Dataset
ds = Dataset(cube,                     # (n, ny, nx) float32, star at ((nx-1)/2, (ny-1)/2), background-subtracted
             angles,                   # (n,) parallactic angle per frame, degrees
             tags={"corrs": c, "noises": s, "coronoise": r},   # optional per-frame quality tags (see §4)
             texp=3600.0,              # total on-source seconds (partition weights)
             name="night1",
             ref_cube=None)            # optional (m, ny, nx) reference-star frames for RDI/ARDI
```

Convention: rotating frame *i* counter-clockwise by `angles[i] + truenorth` (with pixel
(0,0) at the lower left) puts north up and east left. If your angles have the opposite
sign, negate them; a constant offset goes into `truenorth`. Check this once on a known
companion — it cannot be inferred from the code.

One `Dataset` per *partition* (night, roll, chop state, epoch …). Partitions get their own
parameter blocks and can be dropped by the search (`drop1/drop2` dims).

## 2. Quickest: `FunctionReducer`

Your function receives the cube **after** klip-tpe's pre-processing — companions injected
with the instrument PSF model, bad frames removed by the searched thresholds, high-pass
filtered by the searched `filter`, temporally binned by the searched `bin` — and must return
the PSF-subtracted, **derotated, combined** image:

```python
import numpy as np
from klip_tpe.backends import FunctionReducer
from klip_tpe.klip import derotate

def my_subtract(cube, angles, params, k_scan):
    """cube (n, ny, nx), angles (n,) deg, params dict, k_scan bool -> image (ny, nx)"""
    k = params["k_klip"]                      # searched parameters arrive resolved
    inrad, outrad = params["inrad"], params["outrad"]
    residual = my_pca(cube, k, inrad, outrad)                      # your algorithm
    return np.nanmean(derotate(residual, angles, params["truenorth"]), axis=0)

red = FunctionReducer(ds, my_subtract, pxscale=0.0179, lam_m=11.1e-6, diam_m=8.4,
                      injection_model=my_psf_model,                 # §3
                      fwhm_px=14.0, extra_params={"my_knob": 3})
```

* `params` always contains `k_klip, inrad, outrad, n_ang, angsep, anglemax, bin, filter,
  corr_thresh, noise_max, coronoise_max, truenorth, lam_over_d_px, fwhm_px, ref_cube`
  plus your `extra_params`. Anything in it can be searched by adding a `Param` of the same
  name to the space (§5).
* `k_scan=True` asks for `(k, ny, nx)` — one image per `k = 1..k_klip` from one
  decomposition. Return that and pass `supports_kscan=True` to unlock the Runner's scan
  k-modes; otherwise leave it and `k_klip` is searched like any other parameter.
* Pixels outside your zone should be NaN (the metric ignores NaN).
* `ref_cube` is the high-passed reference-star cube when the dataset has one and
  `use_rdi` is set.

## 3. Injection PSF

Contrast means "companion flux / star flux in the same units as the frames", so the
reducer needs a PSF model with a flux unit:

```python
from klip_tpe.injection import GaussianPSF, TemplatePSF, AiryPSF, FramePSF, LibraryPSF
GaussianPSF(fwhm_px, star_flux=F)               # analytic; F = star flux in frame units
TemplatePSF(unsat_psf_image, center=(x, y), ee_radius_px=r, star_flux=F)   # an unsaturated PSF image
AiryPSF(sigmax, sigmay, p, amp, frame_params=reffits)                       # (obstructed) Airy fit(s), pyNOMIC style
FramePSF(ds.cube, r_ee=1.5*lam_over_d_px)       # each frame's own star as its template
LibraryPSF(slices, seps_as, center, ee_radius_px, throughput_fn=...)         # separation-dependent (coronagraphs)
```

A model may also provide `throughput(rho_as)` (coronagraphic transmission) and
`matched_filter_kernel(rho_as, fwhm)` (used by the S/N metric instead of a Gaussian).

## 4. Frame-quality tags (optional)

If you supply `tags`, three searched dimensions select frames before subtraction:
`corrs >= corr_thresh`, `noises <= noise_max`, `coronoise <= coronoise_max`, normalised so
that 1 = median and the searched multipliers range 0.3–3. Any three per-frame quality
numbers work (PSF correlation, background rms, fit residual, FWHM, …). Without tags,
build the space with `opt_framesel=False`.

## 5. Space, objective, run

```python
from klip_tpe import Param, SearchSpace, kgrid, Objective, MawetPeakSNR, PositionSampler, Runner, RunConfig, CalibrationConfig
from klip_tpe.reducer import PartitionedReducer
from klip_tpe.instruments import near
from klip_tpe.instruments.nomic import make_guard         # instrument-agnostic guardrail builder

reds = PartitionedReducer({"n1": red1, "n2": red2}, max_workers=2)
space = near.make_space(reds, opt_framesel=True)          # the production 9-per-partition space (+ drop dims)
space.add(Param("my_knob", 1, 10, "int", default=3))      # your own parameter
space.project = make_guard(reds)                          # frame-selection snapping + reference-count guardrail
objective = Objective(MawetPeakSNR(pxscale=reds.pxscale, fwhm=reds.fwhm,
                                   kernel_fn=reds.matched_filter_kernel), clean_subtract=True)
sampler = PositionSampler(fwhm_as=reds.fwhm * reds.pxscale)
cfg = RunConfig(ann_edges=[20, 45], n_iter=300, n_init=50,
                calibration=CalibrationConfig(forced=[2e-4]))   # or leave forced=None to calibrate S/N 4–6
Runner(reds, space, objective, sampler, cfg, "runs/my_ann1").run()
```

Or build the space by hand:

```python
block = [Param("bin", 1, 20, "int", default=5), Param("filter", 0, 40, "int", default=10),
         Param("angsep", 0, 3, "float", default=0.5), Param("anglemax", 20, 120, "int", default=60),
         Param("k_klip", 1, 50, "int", grid=kgrid(50), default=10)]
space = SearchSpace().replicate(block, ["n1", "n2"]).with_selection("two_slot", partitions=["n1", "n2"])
```

Resume after any interruption with `Runner.resume(run_dir, reds, objective, sampler,
project=space.project)`; extend a finished run with `Runner.extend(...)`.

## 6. Full control: `ExternalReducer`

When the pipeline cannot take an in-memory cube, implement the whole request:

```python
from klip_tpe.backends import ExternalReducer
from klip_tpe import ReductionResult

class MyPipeline(ExternalReducer):
    supports_kscan = False
    def __init__(self, ...):
        super().__init__(pxscale=0.0179, fwhm_px=14.0, angles=my_parangs, tags=None, texp=3600, name="mine")
    def reduce(self, req):
        # req.params: dict; req.injections: [Source(rho_as, theta_deg, contrast), ...] or None
        image = run_my_pipeline(req.params, inject=req.injections)      # north up, star at ((nx-1)/2, (ny-1)/2)
        return ReductionResult(image.astype("float32"), weight=1.0, meta={"anything": "you like"})
```

`frame_angles()` must give the parallactic angles so the reference-count guardrail and the
binning-span rule can work; if you cannot, set `space.project = None` and leave
`angsep/anglemax` out of the space. Injections must follow the same PA convention as the
metric (PA east of north, north up after your derotation) — the easiest way to guarantee
that is to inject with `klip_tpe.injection.inject_sources` on your frames before your
pipeline runs.

## 7. Checklist before a long run

1. `python -m pytest -q` passes (your reducer can be exercised with `tests/test_backends.py`
   as a template).
2. One clean + one injected reduction at the default parameters: the injected companion
   appears at the expected (rho, PA) in `ReductionResult.image`.
3. Contrast: a source injected at contrast *c* has peak ≈ *c* × star peak × throughput.
4. `RunConfig(n_iter=6, n_init=3)` smoke run completes and writes `annulus01/winner.json`.
