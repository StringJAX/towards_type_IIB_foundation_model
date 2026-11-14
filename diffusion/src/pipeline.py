"""Sample the transformer evaluation panel using the diffusion checkpoints.

The model, tokenizer, masking and samplers were extracted from
diffusion-transformer/src. This adapter exports the processed-sample format
consumed by transformer/evaluate.py, without importing JAX or a physics solver.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .checkpoint import (
    checkpoint_mask_schedule,
    checkpoint_time_conditioning,
    load_flux_transformer_model,
)
from .sampling import ConfidenceOrderFluxSampler, MaskGITFluxSampler, StructuredFluxSampler
from .sampling.batched import sample_flux_in_batches


REPO_ROOT: Path = Path(__file__).resolve().parents[2]
HODGE_PAIRS: tuple[tuple[int, int], ...] = (
    (2, 120), (2, 128), (2, 132), (2, 144), (2, 272),
    (3, 99), (3, 107), (3, 115), (3, 119), (3, 243),
    (4, 98), (4, 118), (4, 122), (4, 130), (4, 142),
    (5, 89), (5, 149), (5, 153), (5, 165), (5, 213),
)
# Targets match the hard-coded prompts in transformer/evaluate.py.
MODES: dict[str, tuple[str, str | None, float | None]] = {
    "unconditional": ("uncon_grpo_best.pt", None, None),
    "nflux": ("nflux_grpo_best.pt", "N_flux", 40.0),
    "w0": ("w0_abs_grpo_best.pt", "W0_abs", 1.1),
    "gs_inv": ("gs_grpo_best.pt", "g_s", 0.25),
}
SAMPLERS: dict[str, type[StructuredFluxSampler]] = {
    "random": StructuredFluxSampler,
    "confidence": ConfidenceOrderFluxSampler,
    "maskgit": MaskGITFluxSampler,
}


def to_evaluation_sample(
    generated: dict[str, Any], *, h12: int, mode: str, discr: float = 10.0,
) -> list[np.ndarray]:
    """Return [flux, round(discr*[Re z, Im z]), round(discr*tau)].

    gs_inv exports only Re(tau): the evaluator supplies Im(tau)=4 from its
    prompt. This intentionally follows the transformer's initializer convention.
    Rows are neither filtered nor deduplicated before physics evaluation.

    Args:
        generated (dict[str, Any]): Sampler output with ``flux`` of shape
            ``(N, 4 * (h12 + 1))`` and ``initializer`` holding complex ``z`` of
            shape ``(N, >= h12)`` and ``tau`` of shape ``(N,)``.
        h12 (int): ``h12`` of the prompted geometry; only the first ``h12``
            entries of ``z`` are exported.
        mode (str): Key of :data:`MODES`.
        discr (float): Positive, finite scale of the integer coordinate
            encoding. Defaults to 10.0.

    Returns:
        list[np.ndarray]: ``int64`` arrays ``[flux, moduli, tau]`` of shapes
            ``(N, 4 * (h12 + 1))``, ``(N, 2 * h12)`` with columns
            ``[Re z_1..Re z_h12, Im z_1..Im z_h12]``, and ``(N, 1)`` for
            ``gs_inv`` or ``(N, 2)`` otherwise.

    Raises:
        ValueError: If ``mode`` is unknown, ``discr`` is not positive and
            finite, the flux and initializer shapes disagree, or a scaled
            coordinate is not finite or does not fit in ``int64``.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}")
    if not math.isfinite(discr) or discr <= 0:
        raise ValueError("discr must be positive and finite")
    flux = generated["flux"].detach().cpu().numpy().astype(np.int64)
    z = generated["initializer"]["z"].detach().cpu().numpy()[:, :h12]
    tau = generated["initializer"]["tau"].detach().cpu().numpy()
    if flux.shape != (len(z), 4 * (h12 + 1)) or tau.shape != (len(z),):
        raise ValueError("Generated flux and initializer shapes disagree")
    moduli = np.concatenate((z.real, z.imag), axis=1).astype(np.float64)
    tau_parts = tau.real[:, None] if mode == "gs_inv" else np.column_stack((tau.real, tau.imag))

    def quantize(values: np.ndarray) -> np.ndarray:
        """Return ``rint(values * discr)`` as ``int64``, rejecting unrepresentable values."""
        rounded = np.rint(np.asarray(values, dtype=np.float64) * discr)
        if not np.isfinite(rounded).all() or np.any(np.abs(rounded) >= 2**63):
            raise ValueError("Initializer cannot be represented as scaled int64 coordinates")
        return rounded.astype(np.int64)

    return [flux, quantize(moduli), quantize(tau_parts)]


def sample_for_evaluation(
    mode: str,
    *,
    checkpoint: str | Path | None = None,
    num_samples: int = 1000,
    batch_size: int = 64,
    seed: int = 90261001,
    device: str = "auto",
    sampler: str | None = None,
    temperature: float | None = None,
    guidance_scale: float | None = None,
    discr: float = 10.0,
    show_progress: bool = True,
) -> tuple[list[list[np.ndarray]], dict[str, Any]]:
    """Sample all 20 geometries in exactly the transformer's prompt order.

    Defaults come from checkpoint validation settings, then sampling settings.
    Reproducibility requires the same seed, device, settings and batch size.

    Args:
        mode (str): Key of :data:`MODES`.
        checkpoint (str | Path | None): Checkpoint file. Defaults to the file
            named for ``mode`` in :data:`MODES`, under ``diffusion/checkpoints``.
        num_samples (int): Samples per geometry. Defaults to 1000.
        batch_size (int): Maximum samples per sampler call. Defaults to 64.
        seed (int): Base seed; geometry ``i`` of :data:`HODGE_PAIRS` uses
            ``(seed + i) mod 2**64``. Defaults to 90261001.
        device (str): Torch device, or ``"auto"`` for CUDA when available and
            CPU otherwise. Defaults to ``"auto"``.
        sampler (str | None): Key of :data:`SAMPLERS`; ``None`` takes the
            checkpoint's setting, else ``"random"``. Defaults to ``None``.
        temperature (float | None): Sampling temperature; ``None`` takes the
            checkpoint's setting, else 1.0. Defaults to ``None``.
        guidance_scale (float | None): Classifier-free guidance weight;
            ``None`` takes the checkpoint's setting, else 1.0. Defaults to
            ``None``.
        discr (float): Scale of the integer initializer encoding. Defaults to
            10.0.
        show_progress (bool): Whether to show a progress bar over geometries.
            Defaults to ``True``.

    Returns:
        tuple[list[list[np.ndarray]], dict[str, Any]]: One
            :func:`to_evaluation_sample` entry per geometry in :data:`HODGE_PAIRS`
            order, and the settings metadata written to the JSON sidecar.

    Raises:
        ValueError: If an argument is invalid (including a ``discr`` for which
            ``4 * discr`` is not an integer in mode ``gs_inv``), the checkpoint
            is invalid or not conditioned as ``mode`` requires, or its geometry
            metadata does not contain each evaluation geometry exactly once.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {tuple(MODES)}")
    for name, value in (("num_samples", num_samples), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(discr) or discr <= 0:
        raise ValueError("discr must be positive and finite")
    if mode == "gs_inv" and not np.isclose(discr * 4, round(discr * 4)):
        raise ValueError("discr must represent the evaluator's gs_inv=4 prompt exactly")
    filename, observable, target = MODES[mode]
    checkpoint = Path(checkpoint) if checkpoint is not None else REPO_ROOT / "diffusion/checkpoints" / filename
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model, payload = load_flux_transformer_model(checkpoint, device=device)
    if observable is None:
        if model.config.observable_conditioning_enabled:
            raise ValueError("Unconditional mode requires the unconditional checkpoint")
    elif not model.config.observable_conditioning_enabled or model.config.conditioned_observables != (observable,):
        raise ValueError(f"{mode} requires a checkpoint conditioned on {observable}")

    config = payload.get("run_config", {})
    defaults = {**config.get("sampling", {}), **config.get("validation", {})}
    sampler_name = sampler if sampler is not None else defaults.get("sampler", "random")
    if sampler_name not in SAMPLERS:
        raise ValueError(f"Unknown sampler {sampler_name!r}; choose from {tuple(SAMPLERS)}")
    temperature = float(defaults.get("temperature", 1.0) if temperature is None else temperature)
    guidance_scale = float(defaults.get("guidance_scale", 1.0) if guidance_scale is None else guidance_scale)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    time_conditioning = checkpoint_time_conditioning(payload)
    sampler_kwargs = {
        "num_timesteps": defaults.get("timesteps", 200) if time_conditioning == "discrete" else None,
        "mask_schedule": checkpoint_mask_schedule(payload),
        "time_conditioning": time_conditioning,
        "temperature": temperature,
        "guidance_scale": guidance_scale,
        "device": device,
    }
    flux_sampler = SAMPLERS[sampler_name](
        model, payload["tokenizer_metadata"], payload["geometry_metadata"], **sampler_kwargs,
    )
    geometries = payload["geometry_metadata"]["geometries"]
    by_pair = {(int(row["h12"]), int(row["h11"])): row for row in geometries}
    if len(by_pair) != len(geometries) or any(pair not in by_pair for pair in HODGE_PAIRS):
        raise ValueError("Checkpoint must contain each evaluation geometry exactly once")
    samples = []
    prompt_seeds = []
    for index, pair in enumerate(tqdm(HODGE_PAIRS, desc=mode, disable=not show_progress)):
        prompt_seed = (int(seed) + index) % (2**64)
        prompt_seeds.append(prompt_seed)
        generated = sample_flux_in_batches(
            flux_sampler, num_samples=num_samples, sampling_batch_size=batch_size,
            seed=prompt_seed, geometry_id=int(by_pair[pair]["geometry_id"]),
            observable_target=target, predict_initializer=True,
        )
        samples.append(to_evaluation_sample(generated, h12=pair[0], mode=mode, discr=discr))
    metadata = {
        "checkpoint": str(checkpoint.resolve()), "mode": mode,
        "observable": observable, "physical_target": target,
        "hodge_pairs_h12_h11": [list(pair) for pair in HODGE_PAIRS],
        "geometry_ids": [int(by_pair[pair]["geometry_id"]) for pair in HODGE_PAIRS],
        "num_samples_per_geometry": num_samples, "batch_size": batch_size,
        "seed": int(seed), "prompt_seeds": prompt_seeds,
        "sampler": sampler_name, **sampler_kwargs, "torch_version": str(torch.__version__),
        "discr": discr, "initializer_encoding": "round(discr * physical_coordinate)",
        "tau_imaginary_from_prompt": 4.0 if mode == "gs_inv" else None,
    }
    return samples, metadata


def sample_and_save(mode: str, *, output: str | Path | None = None, **kwargs: Any) -> Path:
    """Save portable NumPy arrays plus a JSON settings sidecar; refuse overwrites.

    The samples of :func:`sample_for_evaluation` are pickled (protocol 4) to
    ``output`` and the metadata written to ``output`` with suffix ``.json``.

    Args:
        mode (str): Key of :data:`MODES`.
        output (str | Path | None): Sample file. Defaults to
            ``diffusion/samples/diffusion_{mode}_samples.p``.
        **kwargs (Any): Keyword arguments for :func:`sample_for_evaluation`.

    Returns:
        Path: The sample file.

    Raises:
        ValueError: If ``mode`` is unknown or ``output`` has suffix ``.json``,
            or as for :func:`sample_for_evaluation`.
        FileExistsError: If the sample file or its sidecar already exists;
            this is checked before sampling.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}")
    output = Path(output) if output is not None else REPO_ROOT / "diffusion/samples" / f"diffusion_{mode}_samples.p"
    metadata_path = output.with_suffix(".json")
    if output == metadata_path:
        raise ValueError("Sample output must not use the .json suffix")
    for path in (output, metadata_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    samples, metadata = sample_for_evaluation(mode, **kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        pickle.dump(samples, handle, protocol=4)
    with metadata_path.open("x") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return output


def main() -> None:
    """Parse command-line arguments and run :func:`sample_and_save`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=90261001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sampler", choices=SAMPLERS)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--discr", type=float, default=10.0)
    args = vars(parser.parse_args())
    mode = args.pop("mode")
    print(f"Saved samples to {sample_and_save(mode, **args)}")


if __name__ == "__main__":
    main()
