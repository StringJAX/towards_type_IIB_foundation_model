"""Evaluate saved diffusion samples with the existing transformer physics utilities."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT: Path = Path(__file__).resolve().parents[1]
HODGE_PAIRS: tuple[tuple[int, int], ...] = (
    (2, 120), (2, 128), (2, 132), (2, 144), (2, 272),
    (3, 99), (3, 107), (3, 115), (3, 119), (3, 243),
    (4, 98), (4, 118), (4, 122), (4, 130), (4, 142),
    (5, 89), (5, 149), (5, 153), (5, 165), (5, 213),
)
# These configs select exactly the novelty-reference datasets used by
# transformer/evaluate.ipynb. No neural-network checkpoint is loaded.
REFERENCE_RUNS: dict[str, int] = {"nflux": 38, "w0": 39, "gs_inv": 46, "unconditional": 38}
TARGETS: dict[str, float | None] = {"nflux": 40, "w0": 1.1, "gs_inv": 4.0, "unconditional": None}


def load_samples(path: Path, mode: str, discr: float | None = None) -> tuple[Any, float, dict[str, Any]]:
    """Validate the full geometry panel and recover its coordinate scale.

    The optional JSON sidecar ``path.with_suffix(".json")`` written by
    :func:`diffusion.src.pipeline.sample_and_save` is checked against
    ``mode``, :data:`HODGE_PAIRS` and ``discr``.

    Args:
        path (Path): Pickled panel: one ``[flux, moduli, tau]`` entry per
            geometry in :data:`HODGE_PAIRS` order, with widths
            ``4 * (h12 + 1)``, ``2 * h12`` and 1 (``gs_inv``) or 2.
        mode (str): Evaluation mode, a key of :data:`REFERENCE_RUNS`.
        discr (float | None): Scale of the integer coordinates; ``None`` takes
            the sidecar's value, else 10.0. Defaults to ``None``.

    Returns:
        tuple[Any, float, dict[str, Any]]: The unpickled panel, the scale
            ``discr`` and the sidecar metadata (empty without a sidecar).

    Raises:
        ValueError: If the sidecar's mode or geometry order disagrees,
            ``discr`` is not positive and finite or disagrees with the
            sidecar, ``4 * discr`` is not an integer in mode ``gs_inv``, or an
            entry is missing, malformed, of the wrong shape or not finite.
        FileNotFoundError: If ``path`` does not exist.
    """
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if "mode" in metadata and metadata["mode"] != mode:
        raise ValueError(f"Requested {mode}, but sample metadata records {metadata['mode']}")
    if "hodge_pairs_h12_h11" in metadata:
        if metadata["hodge_pairs_h12_h11"] != [list(pair) for pair in HODGE_PAIRS]:
            raise ValueError("Sample geometry order does not match the evaluation panel")
    if discr is None:
        discr = float(metadata.get("discr", 10.0))
    if not math.isfinite(discr) or discr <= 0:
        raise ValueError("discr must be positive and finite")
    if "discr" in metadata and not math.isclose(discr, float(metadata["discr"])):
        raise ValueError("discr disagrees with the sample metadata")
    if mode == "gs_inv" and not math.isclose(4 * discr, round(4 * discr)):
        raise ValueError("discr must represent the gs_inv=4 prompt exactly")
    with path.open("rb") as handle:
        samples = pickle.load(handle)
    if len(samples) != len(HODGE_PAIRS):
        raise ValueError("Expected one sample entry for each of the 20 geometries")
    for pair, parts in zip(HODGE_PAIRS, samples):
        if parts is None:
            raise ValueError(f"Missing diffusion samples for geometry {pair}")
        if len(parts) != 3:
            raise ValueError(f"Expected [flux, moduli, tau] for geometry {pair}")
        widths = (4 * (pair[0] + 1), 2 * pair[0], 1 if mode == "gs_inv" else 2)
        count = len(parts[0])
        for part, width in zip(parts, widths):
            values = np.asarray(part)
            if values.shape != (count, width) or not np.isfinite(values).all():
                raise ValueError(f"Invalid sample shape or coordinates for geometry {pair}")
    return samples, discr, metadata


def evaluate_samples(
    mode: str, *, samples_file: str | Path | None = None,
    output_dir: str | Path | None = None, discr: float | None = None,
    tol: float = 1e-10, use_newton: bool = False, verbose: bool = False,
) -> Path:
    """Validate each sample on its prompted geometry, as in transformer evaluation.

    Conditional distributions retain the transformer's {prompt: {dist, validity}}
    structure. A JSON summary also records counts and novelty per geometry.

    The process environment variable ``JAX_PLATFORMS`` is set to ``cpu`` and
    float64 is enabled before ``jaxvacua`` and ``transformer/evaluation_scripts``
    are imported. Novelty is measured against the training data of the
    reference run in :data:`REFERENCE_RUNS`. For ``w0`` and ``gs_inv`` the
    prompt key uses the target scaled by ``discr`` and rounded.

    Args:
        mode (str): Key of :data:`REFERENCE_RUNS`.
        samples_file (str | Path | None): Sample file; its suffix is set to
            ``.p`` if different, and a relative path is taken inside
            ``diffusion/samples``. Defaults to ``diffusion_{mode}_samples.p``.
        output_dir (str | Path | None): Output directory. Defaults to
            ``diffusion/evaluations``.
        discr (float | None): Coordinate scale, as for :func:`load_samples`.
            Defaults to ``None``.
        tol (float): Positive, finite solver tolerance passed to
            ``validate_vacua``. Defaults to ``1e-10``.
        use_newton (bool): Pass ``use_scipy=False`` to ``validate_vacua``
            (recorded as ``newton``) instead of ``True`` (``scipy_root``).
            Defaults to ``False``.
        verbose (bool): Passed to ``print_console_output``. Defaults to
            ``False``.

    Returns:
        Path: The result pickle ``{stem}_valid.p``, holding the per-prompt
            distributions or, for ``unconditional``, the per-geometry counts; the
            summary is written next to it as ``{stem}_summary.json``.

    Raises:
        ValueError: If ``mode`` is unknown, ``tol`` is not positive and
            finite, or the samples fail :func:`load_samples`.
        FileExistsError: If an output file already exists; this is checked
            before loading the samples.
        FileNotFoundError: If the samples file, the reference run's config, one
            of its training-data files or ``geometry_log_degree_10.p`` is missing.
    """
    if mode not in REFERENCE_RUNS:
        raise ValueError(f"Unknown evaluation mode {mode!r}")
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
    path = Path(samples_file) if samples_file is not None else Path(f"diffusion_{mode}_samples.p")
    if path.suffix != ".p":
        path = path.with_suffix(".p")
    if not path.is_absolute():
        path = REPO_ROOT / "diffusion/samples" / path
    destination = Path(output_dir) if output_dir is not None else REPO_ROOT / "diffusion/evaluations"
    result_path = destination / f"{path.stem}_valid.p"
    summary_path = destination / f"{path.stem}_summary.json"
    for target in (result_path, summary_path):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
    samples, discr, metadata = load_samples(path, mode, discr)

    # CPU and float64 must be configured before physics utilities initialize JAX.
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jaxvacua as jvc

    transformer_dir = REPO_ROOT / "transformer"
    if str(transformer_dir) not in sys.path:
        sys.path.insert(0, str(transformer_dir))
    from evaluation_scripts.evaluation_utils import convert_samples_to_data, validate_vacua
    from evaluation_scripts.output_utils import print_console_output

    run = REFERENCE_RUNS[mode]
    config_path = transformer_dir / f"logs/run_{run}/run_{run}.json"
    config = json.loads(config_path.read_text())
    training_suffix = "_".join(Path(config["train_data_flux"]).stem.split("_")[2:])
    data_dir = transformer_dir / "data/data_multiple"
    training_paths = {
        h12: data_dir / f"train_data_raw_{training_suffix}_{h12}.npy"
        for h12 in (2, 3, 4, 5)
    }
    for training_path in training_paths.values():
        if not training_path.is_file():
            raise FileNotFoundError(training_path)
    with (data_dir / "geometry_log_degree_10.p").open("rb") as handle:
        geometry_log = pickle.load(handle)

    started = time.time()
    distributions, rows = {}, []
    training_data = set()
    previous_h12 = None
    for (h12, h11), parts in zip(HODGE_PAIRS, samples):
        if h12 != previous_h12:
            training_data = set(map(tuple, np.load(training_paths[h12], mmap_mode="r")))
            previous_h12 = h12
        geometry_key = f"{h12} {h11}"
        print(f"\nPrompted geometry: [{h12} {h11}]; samples: {len(parts[0])}", flush=True)
        model = jvc.FluxVacuaFinder(
            h12=h12, model_ID=geometry_log[geometry_key]["model_ID"], maximum_degree=10,
        )
        aux = TARGETS[mode]
        if mode in {"w0", "gs_inv"}:
            aux = int(round(aux * discr))
        fluxes, moduli, tau = convert_samples_to_data(
            [jnp.asarray(part) for part in parts], discr,
            gs_inv=aux if mode == "gs_inv" else None,
        )
        out = {"geometry": geometry_key, "failed_tadpole": 0, "valid": 0, "valid_unique": 0}
        if len(fluxes):
            fluxes, moduli, tau, _, out = validate_vacua(
                fluxes, moduli, tau, jnp.arange(len(fluxes)), model,
                out, tol, not use_newton, False, False, False,
            )
        out["num_in_train_data"] = (
            sum(tuple(row) in training_data for row in np.asarray(fluxes))
            if fluxes is not None else 0
        )
        out["num_samples"] = len(parts[0])
        print_console_output(out, aux, verbose)
        rows.append({key: value if isinstance(value, str) else int(value) for key, value in out.items()})
        if mode != "unconditional":
            dist = np.empty(0, dtype=np.float64)
            if fluxes is not None and len(fluxes):
                if mode == "nflux":
                    dist = np.asarray(model.tadpole(fluxes))
                elif mode == "w0":
                    dist = np.asarray(jnp.abs(model.superpotential_gauge_invariant(moduli, tau, fluxes)))
                else:
                    dist = np.asarray(jnp.imag(tau))
            distributions[f"{geometry_key} {aux}"] = {"dist": dist, "validity": int(out["valid_unique"])}
        jax.clear_caches()

    total = sum(row["num_samples"] for row in rows)
    unique = sum(row["valid_unique"] for row in rows)
    novel = unique - sum(row["num_in_train_data"] for row in rows)
    summary = {
        "mode": mode, "samples_file": str(path.resolve()), "discr": discr,
        "physical_target": TARGETS[mode], "tol": tol,
        "solver": "newton" if use_newton else "scipy_root",
        "reference_run": run, "novelty_reference": "transformer training data",
        "sample_metadata": metadata, "geometries": rows,
        "total_samples": total, "valid_unique": unique, "valid_unique_novel": novel,
        "valid_unique_rate": unique / total if total else 0.0,
        "valid_unique_novel_rate": novel / total if total else 0.0,
        "elapsed_seconds": time.time() - started,
    }
    destination.mkdir(parents=True, exist_ok=True)
    with result_path.open("xb") as handle:
        pickle.dump(distributions if mode != "unconditional" else rows, handle, protocol=4)
    with summary_path.open("x") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Valid unique: {unique}/{total}; novel: {novel}/{total}")
    print(f"Saved {result_path}\nSaved {summary_path}")
    return result_path


def main() -> None:
    """Parse command-line arguments and run :func:`evaluate_samples`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=REFERENCE_RUNS, required=True)
    parser.add_argument("--samples_file", help="Filename/stem inside diffusion/samples, or an absolute path")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--discr", type=float, help="Defaults to the sampling JSON sidecar's value, or 10")
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--use_newton", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    evaluate_samples(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
