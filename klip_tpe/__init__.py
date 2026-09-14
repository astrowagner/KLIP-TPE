"""klip_tpe -- Bayesian (TPE) optimization of high-contrast-imaging reduction
parameters, with a reference KLIP/ADI reducer, injection-recovery metrics,
calibration/validation protocol, checkpoint/resume and post-search verification.

Python port of the IDL ``optimize_near_2_tpe`` code base (KLIP-TPE paper).
"""
__version__ = "0.1.0"

# Import ``multiprocessing.synchronize`` here, at package import: the main thread, before a
# run has started a render thread or a worker pool.  ``multiprocessing.Queue.__init__`` ends
# in a *lazy* import of it -- CPython's own comment there reads "Can raise ImportError" --
# and a pool built while another thread happens to be importing the same module is handed a
# half-built one and raises, costing the whole run its worker processes.  It cost one
# RX J0534 search exactly that.  See the long note in :mod:`klip_tpe.parallel`; guarded,
# because a platform without a working ``sem_open`` must still be able to import klip-tpe.
try:                                                                      # noqa: E402
    import multiprocessing.synchronize                                    # noqa: E402,F401
except Exception:                                                         # pragma: no cover
    pass

from .space import Param, SearchSpace, Config, kgrid                     # noqa: E402,F401
from .optimizers import TPE, RandomSearch, GridSearch, History           # noqa: E402,F401
from .metrics import (MawetPeakSNR, InjectionDifferenceSNR, Objective,    # noqa: E402,F401
                      Source, mawet_peak_snr, radprof)
from .positions import PositionSampler                                   # noqa: E402,F401
from .injection import GaussianPSF, TemplatePSF, AiryPSF, FramePSF, LibraryPSF, inject_sources  # noqa: E402,F401
from .reducer import (Reducer, KLIPReducer, PartitionedReducer, Dataset,  # noqa: E402,F401
                      ReductionRequest, ReductionResult)
from .feasibility import ReferenceCountGuard, FrameSelectionGuard, compose  # noqa: E402,F401
from .reducer import reducer_class                                       # noqa: E402,F401
from .runner import Runner, RunConfig, CalibrationConfig, ValidationConfig  # noqa: E402,F401
from . import datasets                                                   # noqa: E402,F401
