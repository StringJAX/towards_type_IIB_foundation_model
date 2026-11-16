"""Portable diffusion samples in physical units; no tokenizer encoding."""

from __future__ import annotations

import json
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np

SAMPLE_FORMAT = "diffusion_physical_v1"
HODGE_PAIRS = (
    (2, 120), (2, 128), (2, 132), (2, 144), (2, 272),
    (3, 99), (3, 107), (3, 115), (3, 119), (3, 243),
    (4, 98), (4, 118), (4, 122), (4, 130), (4, 142),
    (5, 89), (5, 149), (5, 153), (5, 165), (5, 213),
)
TARGETS = {"nflux": 40.0, "w0": 1.1, "gs_inv": 4.0, "unconditional": None}


def sample_hodge_pairs(metadata: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    """Read a nonempty ordered subset of the evaluation panel."""
    pairs = tuple(tuple(pair) for pair in metadata["hodge_pairs_h12_h11"])
    if not pairs or pairs != tuple(pair for pair in HODGE_PAIRS if pair in pairs):
        raise ValueError("Sample geometry order must be an ordered subset of the evaluation panel")
    return pairs


def sample_prompt_target(mode: str, metadata: dict[str, Any]) -> float | None:
    """Read a physical prompt target; gs_inv denotes 1/g_s."""
    if mode not in TARGETS:
        raise ValueError(f"Unknown evaluation mode {mode!r}")
    target = metadata.get("prompt_target", TARGETS[mode])
    if mode == "unconditional":
        if target is not None:
            raise ValueError("Unconditional samples cannot have an observable target")
        return None
    if target is None or not math.isfinite(target):
        raise ValueError("The prompt target must be finite")
    if mode == "nflux" and target != int(target):
        raise ValueError("nflux must be an integer")
    if mode == "w0" and target < 0:
        raise ValueError("w0 must be non-negative")
    if mode == "gs_inv" and target <= 0:
        raise ValueError("gs_inv must be positive")
    return float(target)


def validate_sample(sample: dict[str, np.ndarray], h12: int) -> None:
    """Require integer fluxes and finite complex z/tau arrays in physical units."""
    if not isinstance(sample, dict) or set(sample) != {"flux", "z", "tau"}:
        raise ValueError("Expected a diffusion sample dictionary with flux, z, and tau")
    flux, z, tau = (np.asarray(sample[key]) for key in ("flux", "z", "tau"))
    if flux.ndim != 2:
        raise ValueError("Flux must be a two-dimensional array")
    count = len(flux)
    if flux.shape != (count, 4 * (h12 + 1)) or z.shape != (count, h12) or tau.shape != (count,):
        raise ValueError(f"Invalid sample shapes for h12={h12}")
    if not np.issubdtype(flux.dtype, np.integer):
        raise ValueError("Saved fluxes must have integer dtype")
    if not np.iscomplexobj(z) or not np.iscomplexobj(tau):
        raise ValueError("Saved z and tau must be complex physical coordinates")
    if not all(np.isfinite(values).all() for values in (flux, z, tau)):
        raise ValueError("Sample coordinates must be finite")


def load_samples(path: str | Path, mode: str | None = None) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    """Read trusted sample pickles and their required JSON metadata.

    Older integer-encoded samples are deliberately rejected: regenerate them
    with the current sampler to retain the original continuous predictions.
    """
    path = Path(path)
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        raise ValueError(f"Missing sampling metadata: {metadata_path}. Regenerate samples with the current sampler.")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("sample_format") != SAMPLE_FORMAT:
        raise ValueError(
            f"Unsupported sample format in {path}: expected {SAMPLE_FORMAT!r}, "
            f"found {metadata.get('sample_format', 'legacy integer encoding')!r}. "
            "Rerun the 'Import the sampling helper and set shared options' block "
            "in diffusion/sample.ipynb to reload the current sampler. "
            "Regenerate samples under a new filename, then evaluate that file."
        )
    recorded_mode = metadata["mode"]
    if mode is not None and recorded_mode != mode:
        raise ValueError(f"Requested {mode}, but sample metadata records {recorded_mode}")
    sample_prompt_target(recorded_mode, metadata)
    pairs = sample_hodge_pairs(metadata)
    with path.open("rb") as handle:
        samples = pickle.load(handle)
    if len(samples) != len(pairs):
        raise ValueError("Expected one sample entry for each recorded geometry")
    for (h12, _), sample in zip(pairs, samples):
        validate_sample(sample, h12)
    return samples, metadata
