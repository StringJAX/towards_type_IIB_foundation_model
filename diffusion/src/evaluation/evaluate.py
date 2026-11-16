"""Evaluate saved physical diffusion samples using the diffusion physics module.

Run from the repository root with ``python -m diffusion.src.evaluation.evaluate``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np

from .physics import REPO_ROOT, cpu_jax, get_geometry_model, load_geometry_log, validate_vacua
from .samples import TARGETS, load_samples, sample_hodge_pairs, sample_prompt_target


def evaluate_samples(
    mode: str | None = None, *, samples_file: str | Path | None = None,
    output_dir: str | Path | None = None, tol: float = 1e-10,
    use_newton: bool = False, verbose: bool = False,
) -> Path:
    """Validate every saved sample on its prompted geometry.

    ``samples_file`` is an absolute path or a filename inside diffusion/samples.
    The mode, geometry panel, and physical target are read from its JSON
    sidecar. Supplying ``mode`` additionally checks that it matches the file.
    With only a mode, the legacy filename diffusion_{mode}_samples.p is used.
    Initializers are used exactly as saved, including Im(tau) in gs_inv mode.

    Saves unique-valid observable distributions to {stem}_valid.p for
    conditional modes, or per-geometry counts for unconditional mode. The
    neighboring {stem}_summary.json records counts, physical targets, settings,
    and timing. Existing output files are protected against overwrites.
    No training datasets or neural-network checkpoints are loaded.
    """
    if mode is not None and mode not in TARGETS:
        raise ValueError(f"Unknown evaluation mode {mode!r}")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
    if samples_file is None:
        if mode is None:
            raise ValueError("Supply samples_file or mode")
        samples_file = f"diffusion_{mode}_samples.p"
    path = Path(samples_file).with_suffix(".p")
    if not path.is_absolute():
        path = REPO_ROOT / "diffusion/samples" / path
    destination = Path(output_dir) if output_dir is not None else REPO_ROOT / "diffusion/evaluations"
    result_path = destination / f"{path.stem}_valid.p"
    summary_path = destination / f"{path.stem}_summary.json"
    for target in (result_path, summary_path):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
    samples, metadata = load_samples(path, mode)
    mode = metadata["mode"]
    prompt_target = sample_prompt_target(mode, metadata)
    geometry_log = load_geometry_log()
    jax = cpu_jax()
    started = time.time()
    distributions, rows = {}, []
    for (h12, h11), sample in zip(sample_hodge_pairs(metadata), samples):
        geometry_key = f"{h12} {h11}"
        print(f"Geometry {geometry_key}: {len(sample['flux'])} samples", flush=True)
        model = get_geometry_model(h12, h11, geometry_log=geometry_log)
        result = validate_vacua(**sample, h12=h12, h11=h11, model=model,
                                tol=tol, use_newton=use_newton, verbose=verbose)
        counts = {"geometry": geometry_key, **result["counts"]}
        rows.append(counts)
        print(f"Valid: {counts['valid']}; unique: {counts['valid_unique']}", flush=True)
        if mode != "unconditional":
            dist = np.empty(0, dtype=np.float64)
            if len(result["flux"]):
                with jax.default_device(jax.devices("cpu")[0]):
                    if mode == "nflux":
                        dist = np.asarray(model.tadpole(result["flux"]))
                    elif mode == "w0":
                        dist = np.abs(np.asarray(model.superpotential_gauge_invariant(
                            result["z"], result["tau"], result["flux"],
                        )))
                    else:
                        dist = result["tau"].imag
            target_key = str(int(prompt_target)) if mode == "nflux" else str(prompt_target)
            distributions[f"{geometry_key} {target_key}"] = {
                "dist": dist, "validity": counts["valid_unique"],
            }
        del model
        jax.clear_caches()
    total = sum(row["num_samples"] for row in rows)
    valid = sum(row["valid"] for row in rows)
    unique = sum(row["valid_unique"] for row in rows)
    summary = {
        "mode": mode, "samples_file": str(path.resolve()),
        "prompt_target": prompt_target, "tol": tol,
        "solver": "newton" if use_newton else "scipy_root",
        "sample_metadata": metadata, "geometries": rows,
        "total_samples": total, "valid": valid, "valid_unique": unique,
        "valid_rate": valid / total if total else 0.0,
        "valid_unique_rate": unique / total if total else 0.0,
        "elapsed_seconds": time.time() - started,
    }
    destination.mkdir(parents=True, exist_ok=True)
    with result_path.open("xb") as handle:
        pickle.dump(distributions if mode != "unconditional" else rows, handle, protocol=4)
    with summary_path.open("x") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Valid unique: {unique}/{total}")
    print(f"Saved {result_path}\nSaved {summary_path}")
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=TARGETS)
    parser.add_argument("--samples_file", help="Filename in diffusion/samples, or absolute path")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--use_newton", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    evaluate_samples(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
