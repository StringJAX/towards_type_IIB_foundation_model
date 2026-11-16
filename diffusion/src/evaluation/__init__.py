"""Independent diffusion sample loading and physical vacuum validation.

Import evaluate_samples from diffusion.src.evaluation.evaluate for file-based
runs, or validate_vacua here for arrays of fluxes and complex initializers.
"""

from .samples import load_samples
from .physics import get_geometry_model, validate_vacua

__all__ = ["load_samples", "get_geometry_model", "validate_vacua"]
