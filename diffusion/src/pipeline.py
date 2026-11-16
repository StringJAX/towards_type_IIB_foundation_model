"""Sample diffusion checkpoints and save full-precision physical initializers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .checkpoint import (
    checkpoint_mask_schedule,
    load_flux_transformer_model,
)
from .sampling import ConfidenceOrderFluxSampler, MaskGITFluxSampler, StructuredFluxSampler
from .sampling.batched import sample_flux_in_batches
from .evaluation.samples import HODGE_PAIRS, SAMPLE_FORMAT


REPO_ROOT: Path = Path(__file__).resolve().parents[2]
# Default physical targets for the conditional checkpoints.
MODES: dict[str, tuple[str, str | None, float | None]] = {
    "unconditional": ("uncon_grpo.pt", None, None),
    "nflux": ("nflux_grpo.pt", "N_flux", 40.0),
    "w0": ("w0_grpo.pt", "W0_abs", 1.1),
    "gs_inv": ("gs_grpo.pt", "g_s", 0.25),
}
SAMPLERS: dict[str, type[StructuredFluxSampler]] = {
    "random": StructuredFluxSampler,
    "confidence": ConfidenceOrderFluxSampler,
    "maskgit": MaskGITFluxSampler,
}


def to_evaluation_sample(generated: dict[str, Any], *, h12: int) -> dict[str, np.ndarray]:
    """Export integer fluxes and complex physical initializers without rounding.

    The result has ``flux`` of shape ``(N, 4*(h12+1))``, ``z`` of shape
    ``(N, h12)``, and ``tau`` of shape ``(N,)``. All observable modes retain
    both real and imaginary parts of the model's predicted initializer.
    """
    sample = {
        "flux": generated["flux"].detach().cpu().numpy().astype(np.int64),
        "z": generated["initializer"]["z"].detach().cpu().numpy()[:, :h12].astype(np.complex128),
        "tau": generated["initializer"]["tau"].detach().cpu().numpy().astype(np.complex128),
    }
    count = len(sample["flux"])
    if (sample["flux"].shape != (count, 4 * (h12 + 1))
            or sample["z"].shape != (count, h12) or sample["tau"].shape != (count,)):
        raise ValueError("Generated flux and initializer shapes disagree")
    if not all(np.isfinite(value).all() for value in sample.values()):
        raise ValueError("Generated sample contains non-finite values")
    return sample


def sample_for_evaluation(
    mode: str | None = None,
    *,
    checkpoint: str | Path | None = None,
    h12: int | None = None,
    nflux: int | None = None,
    w0: float | None = None,
    gs_inv: float | None = None,
    num_samples: int = 1000,
    batch_size: int = 64,
    seed: int = 90261001,
    device: str = "auto",
    sampler: str | None = None,
    temperature: float | None = None,
    guidance_scale: float | None = None,
    show_progress: bool = True,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    """Sample checkpoint geometries in the evaluation-panel order.

    Defaults come from checkpoint validation settings, then sampling settings.
    Reproducibility requires the same seed, device, settings and batch size.

    Args:
        mode (str | None): Key of :data:`MODES`. If omitted, infer the mode
            from the supplied checkpoint's observable conditioner.
        checkpoint (str | Path | None): Checkpoint file. Defaults to the file
            named for ``mode`` in :data:`MODES`, under ``diffusion/checkpoints``.
            Bare filenames are also resolved inside that directory.
        h12 (int | None): Restrict sampling to this Hodge number. Use 5 for
            the fine-tuned and scratch benchmarks; None selects all available
            evaluation geometries.
        nflux (int | None): Integer N_flux target; only valid in nflux mode.
        w0 (float | None): Non-negative |W0| target; only valid in w0 mode.
        gs_inv (float | None): Positive 1/g_s target; only valid in gs_inv
            mode. The model receives its reciprocal, g_s. Omitted targets use
            the existing evaluation defaults (40, 1.1, and 4 respectively).
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
        show_progress (bool): Whether to show a progress bar over geometries.
            Defaults to ``True``.

    Returns:
        tuple[list[dict[str, np.ndarray]], dict[str, Any]]: One
            :func:`to_evaluation_sample` entry per geometry in :data:`HODGE_PAIRS`
            order, and the settings metadata written to the JSON sidecar.

    Raises:
        ValueError: If an argument is invalid, the checkpoint is invalid or
            not conditioned as ``mode`` requires, or its geometry
            metadata has duplicate geometries or no requested geometries.
    """
    if mode is not None and mode not in MODES:
        raise ValueError(f"mode must be one of {tuple(MODES)}")
    if mode is None and checkpoint is None:
        raise ValueError("Supply a checkpoint or a mode")
    if h12 is not None and (isinstance(h12, bool) or not isinstance(h12, int) or h12 <= 0):
        raise ValueError("h12 must be a positive integer")
    for name, value in (("num_samples", num_samples), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    checkpoint_dir = REPO_ROOT / "diffusion/checkpoints"
    checkpoint = Path(checkpoint) if checkpoint is not None else checkpoint_dir / MODES[mode][0]
    if not checkpoint.is_absolute() and checkpoint.parent == Path("."):
        checkpoint = checkpoint_dir / checkpoint
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model, payload = load_flux_transformer_model(checkpoint, device=device)
    if mode is None:
        if not model.config.observable_conditioning_enabled:
            mode = "unconditional"
        else:
            modes = {(value[1],): key for key, value in MODES.items() if value[1] is not None}
            mode = modes.get(tuple(model.config.conditioned_observables))
            if mode is None:
                raise ValueError("Evaluation export requires a single supported conditioned observable")
    _, observable, target = MODES[mode]
    requested = {"nflux": nflux, "w0": w0, "gs_inv": gs_inv}
    if any(value is not None and name != mode for name, value in requested.items()):
        raise ValueError(f"Only the {mode} target may be supplied for this checkpoint")
    if nflux is not None:
        if isinstance(nflux, bool) or not isinstance(nflux, int):
            raise ValueError("nflux must be an integer")
        target = float(nflux)
    if w0 is not None:
        if not math.isfinite(w0) or w0 < 0:
            raise ValueError("w0 must be non-negative and finite")
        target = float(w0)
    if gs_inv is not None:
        if not math.isfinite(gs_inv) or gs_inv <= 0:
            raise ValueError("gs_inv must be positive and finite")
        target = 1.0 / gs_inv
    prompt_target = (float(gs_inv) if gs_inv is not None else 1.0 / target) if mode == "gs_inv" else target
    if target is not None and not math.isfinite(target):
        raise ValueError("The physical target must be finite")
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
    time_conditioning = model.config.time_conditioning
    sampler_kwargs = {
        "num_timesteps": defaults.get("timesteps", config.get("diffusion", {}).get("timesteps", 200)) if time_conditioning == "discrete" else None,
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
    if len(by_pair) != len(geometries):
        raise ValueError("Checkpoint contains duplicate evaluation geometries")
    pairs = tuple(pair for pair in HODGE_PAIRS if pair in by_pair and (h12 is None or pair[0] == h12))
    if not pairs:
        raise ValueError(f"Checkpoint has no evaluation geometries for h12={h12}")
    samples = []
    prompt_seeds = []
    for pair in tqdm(pairs, desc=mode, disable=not show_progress):
        index = HODGE_PAIRS.index(pair)
        prompt_seed = (int(seed) + index) % (2**64)
        prompt_seeds.append(prompt_seed)
        generated = sample_flux_in_batches(
            flux_sampler, num_samples=num_samples, sampling_batch_size=batch_size,
            seed=prompt_seed, geometry_id=int(by_pair[pair]["geometry_id"]),
            observable_target=target, predict_initializer=True,
        )
        samples.append(to_evaluation_sample(generated, h12=pair[0]))
    metadata = {
        "checkpoint": str(checkpoint.resolve()), "mode": mode,
        "observable": observable, "physical_target": target,
        "prompt_target": prompt_target,
        "hodge_pairs_h12_h11": [list(pair) for pair in pairs],
        "geometry_ids": [int(by_pair[pair]["geometry_id"]) for pair in pairs],
        "num_samples_per_geometry": num_samples, "batch_size": batch_size,
        "seed": int(seed), "prompt_seeds": prompt_seeds,
        "sampler": sampler_name, **sampler_kwargs, "torch_version": str(torch.__version__),
        "sample_format": SAMPLE_FORMAT,
        "initializer_encoding": "complex physical coordinates",
    }
    return samples, metadata


def sample_and_save(mode: str | None = None, *, output: str | Path | None = None, **kwargs: Any) -> Path:
    """Save portable NumPy arrays plus a JSON settings sidecar; refuse overwrites.

    The samples of :func:`sample_for_evaluation` are pickled (protocol 4) to
    ``output`` and the metadata written to ``output`` with suffix ``.json``.

    Args:
        mode (str | None): Key of :data:`MODES`, or infer from ``checkpoint``.
        output (str | Path | None): Sample file. Defaults to
            ``diffusion/samples/diffusion_{checkpoint_stem}_samples.p`` when a
            checkpoint is supplied, otherwise ``diffusion_{mode}_samples.p``.
        **kwargs (Any): Keyword arguments for :func:`sample_for_evaluation`.

    Returns:
        Path: The sample file.

    Raises:
        ValueError: If ``mode`` is unknown or ``output`` has suffix ``.json``,
            or as for :func:`sample_for_evaluation`.
        FileExistsError: If the sample file or its sidecar already exists;
            this is checked before sampling.
    """
    if mode is not None and mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}")
    if mode is None and kwargs.get("checkpoint") is None:
        raise ValueError("Supply a checkpoint or a mode")
    stem = Path(kwargs["checkpoint"]).stem if kwargs.get("checkpoint") is not None else mode
    output = Path(output) if output is not None else REPO_ROOT / "diffusion/samples" / f"diffusion_{stem}_samples.p"
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


def generate(
    checkpoint: str | Path,
    file_name: str | None = None,
    *,
    sample_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    num_samples: int = 100,
    h12: int | None = None,
    nflux: int | None = None,
    w0: float | None = None,
    gs_inv: float | None = None,
    batch_size: int = 64,
    seed: int = 90261001,
    device: str = "auto",
    sampler: str | None = "confidence",
    temperature: float | None = 0.4,
    guidance_scale: float | None = 1.0,
    only_sample: bool = True,
    show_progress: bool = True,
) -> Path:
    """Generate and save samples, optionally evaluating them in a CPU process.

    This is the entry point used by ``diffusion/sample.ipynb``. All settings
    are explicit arguments; no notebook globals are required. Sampling and
    checkpoint arguments have the meanings documented in
    :func:`sample_for_evaluation`. Defaults match the notebook's confidence
    sampling recipe, with 100 samples per geometry and guidance 1.0.

    Args:
        checkpoint: Checkpoint path, or a filename in ``diffusion/checkpoints``.
        file_name: Output stem without an extension. Defaults to
            ``diffusion_{checkpoint_stem}_samples``.
        sample_dir: Output directory, defaulting to ``diffusion/samples``.
            Relative paths are resolved from the current working directory.
        results_dir: Evaluation output directory. Defaults to the
            ``evaluations`` subdirectory of ``sample_dir``.
        only_sample: If False, run ``diffusion.src.evaluation.evaluate`` after saving,
            using the current Python interpreter in a separate process.

    Returns:
        Path: Saved sample file, with its settings in a sibling JSON sidecar.

    Raises:
        FileExistsError: If the samples or sidecar already exist.
        subprocess.CalledProcessError: If optional evaluation fails. The saved
            samples remain available for evaluation later.
    """
    checkpoint = Path(checkpoint)
    sample_dir = Path(sample_dir) if sample_dir is not None else REPO_ROOT / "diffusion/samples"
    output = sample_dir.resolve() / f"{file_name or ('diffusion_' + checkpoint.stem + '_samples')}.p"
    path = sample_and_save(
        checkpoint=checkpoint, output=output, num_samples=num_samples,
        h12=h12, nflux=nflux, w0=w0, gs_inv=gs_inv,
        batch_size=batch_size, seed=seed, device=device,
        sampler=sampler, temperature=temperature,
        guidance_scale=guidance_scale, show_progress=show_progress,
    )
    metadata = json.loads(path.with_suffix(".json").read_text())
    print(f"Saved {len(metadata['geometry_ids'])} geometries to {path}")
    print(f"Sampling metadata: {path.with_suffix('.json')}")
    if not only_sample:
        results_dir = Path(results_dir) if results_dir is not None else sample_dir / "evaluations"
        command = [
            sys.executable, "-u", "-m", "diffusion.src.evaluation.evaluate",
            "--mode", metadata["mode"], "--samples_file", str(path),
            "--output_dir", str(results_dir.resolve()),
        ]
        subprocess.run(command, check=True, cwd=REPO_ROOT)
    return path


def main() -> None:
    """Parse command-line arguments and run :func:`sample_and_save`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--h12", type=int)
    parser.add_argument("--nflux", type=int)
    parser.add_argument("--w0", type=float)
    parser.add_argument("--gs-inv", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=90261001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sampler", choices=SAMPLERS)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--guidance-scale", type=float)
    args = vars(parser.parse_args())
    mode = args.pop("mode")
    print(f"Saved samples to {sample_and_save(mode, **args)}")


if __name__ == "__main__":
    main()
