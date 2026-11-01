"""Import before numerical libraries in every entry point."""

import os

for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"


_configured = False


def configure():
    """Set library limits once, permitting repeated setup in tests/importers."""
    global _configured
    if _configured:
        return
    import cv2
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _configured = True
