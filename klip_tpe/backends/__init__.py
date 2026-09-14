"""PSF-subtraction backends other than the built-in annular KLIP.

Every backend is a :class:`~klip_tpe.reducer.KLIPReducer` subclass that keeps the
whole pre-processing chain (injection with the instrument PSF model, frame selection,
high-pass, destriping, angle-aware binning, reference-star handling) and replaces
only the "subtract + derotate + combine" step (:meth:`KLIPReducer._subtract`), so the
optimizer, the metric, the products and the verification stack see exactly the same
interface whichever code does the subtraction.

=====================  ============================================================
``backends.pyklip``     :class:`PyKLIPReducer` -- pyKLIP ``klip_parallelized`` (ADI /
                        RDI / ADI+RDI, algo klip | nmf | empca); k-scan native.
                        :func:`dataset_from_pyklip` ingests any ``pyklip`` Data object.
``backends.vip``        :class:`VIPReducer` -- VIP ``pca_annular`` / ``pca`` /
                        ``median_sub``.
``backends.spaceklip``  :func:`load_spaceklip` -- spaceKLIP database / stage-2 products
                        (JWST) to datasets, reduced with :class:`PyKLIPReducer` like
                        spaceKLIP itself does; optional webbpsf offset PSF for injection.
``backends.custom``     :class:`FunctionReducer` -- wrap your own subtraction function
                        (see ``docs/CUSTOM_PIPELINE.md``).
=====================  ============================================================

The external packages are imported lazily; ``klip_tpe`` itself does not depend on them.
"""
from .custom import FunctionReducer, ExternalReducer

__all__ = ["FunctionReducer", "ExternalReducer"]
