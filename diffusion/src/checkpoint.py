"""Checkpoint loading for FluxVacuumTransformer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .data.moduli import validate_moduli_statistics
from .masking import canonical_mask_schedule
from .model.flux_transformer import FluxVacuumTransformer, FluxVacuumTransformerConfig


def checkpoint_mask_schedule(payload: Mapping[str, Any]) -> str:
    """Read the schedule recorded by supervised or policy training.

    Args:
        payload (Mapping[str, Any]): Loaded checkpoint. The schedule is read
            from ``run_config["diffusion"]["mask_schedule"]``, otherwise from
            ``run_config["sampling"]["mask_schedule"]``.

    Returns:
        str: Canonical schedule name, ``"sine"`` or ``"linear"``.

    Raises:
        ValueError: If the schedule is missing or is not a supported name.
    """

    run_config = payload.get("run_config", {})
    if isinstance(run_config, Mapping):
        diffusion = run_config.get("diffusion", {})
        if isinstance(diffusion, Mapping) and "mask_schedule" in diffusion:
            return canonical_mask_schedule(diffusion["mask_schedule"])
        sampling = run_config.get("sampling", {})
        if isinstance(sampling, Mapping) and "mask_schedule" in sampling:
            return canonical_mask_schedule(sampling["mask_schedule"])
    raise ValueError(
        "Missing checkpoint mask_schedule: expected "
        "run_config['diffusion']['mask_schedule'] or "
        "run_config['sampling']['mask_schedule']."
    )


def load_flux_transformer_model(
    path: str | Path,
    *,
    device: str | torch.device,
) -> tuple[FluxVacuumTransformer, dict[str, Any]]:
    """Load a FluxVacuumTransformer checkpoint for inference.

    The file is loaded with ``weights_only=False``, i.e. fully unpickled, so
    only trusted checkpoints should be loaded. The state dictionary is loaded
    strictly; the model is moved to ``device``, put in evaluation mode and has
    gradients disabled.

    Args:
        path (str | Path): Checkpoint file.
        device (str | torch.device): Device for the model parameters.

    Returns:
        tuple[FluxVacuumTransformer, dict[str, Any]]: The model and the full
            checkpoint payload (tokenizer, geometry and run metadata included).

    Raises:
        ValueError: If the moduli statistics fail
            :func:`validate_moduli_statistics`.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_moduli_statistics(payload["moduli_statistics"])
    model = FluxVacuumTransformer(
        FluxVacuumTransformerConfig(**payload["model_config"]),
        moduli_statistics=payload["moduli_statistics"],
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, payload
