"""Reproducible validation check on flux-only reference data.

Example: python -m diffusion.src.evaluation.validate_dataset --num-samples 500
The selected raw files contain no geometry labels or moduli. Each row is tried
on candidate geometries with matching h12, starting from geometry-log means.
An unresolved row is not proof that its flux has no vacuum.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .physics import REPO_ROOT, cpu_jax, get_geometry_model, load_geometry_log, validate_vacua
from .samples import HODGE_PAIRS


def validate_dataset(*, num_samples: int = 500, seed: int = 90261001,
                     data_dir: str | Path | None = None,
                     output: str | Path | None = None, tol: float = 1e-10) -> Path:
    """Select balanced rows across h12=2..5 and test all candidate geometries.

    Uses the same validate_vacua function as generated-sample evaluation. The
    JSON report records every source row, successful geometry and measured
    F-term residual. Counts refer to input rows, before flux deduplication.
    """
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 4:
        raise ValueError("num_samples must be an integer >= 4")
    data_dir = Path(data_dir) if data_dir is not None else REPO_ROOT / "transformer/data/data_multiple"
    output = Path(output) if output is not None else REPO_ROOT / f"diffusion/evaluations/dataset_validation_{num_samples}.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    log = load_geometry_log(data_dir / "geometry_log_degree_10.p")
    jax = cpu_jax()
    rng = np.random.default_rng(seed)
    rows, attempts = [], []
    started = time.time()
    for number, h12 in enumerate((2, 3, 4, 5)):
        count = num_samples // 4 + (number < num_samples % 4)
        source = data_dir / f"train_data_raw_all_h12_mod_new_{h12}.npy"
        raw = np.load(source, mmap_mode="r")
        selected = np.sort(rng.choice(len(raw), size=count, replace=False))
        flux = np.asarray(raw[selected])
        matched = np.zeros(count, dtype=bool)
        records = [{"source": str(source.resolve()), "row_index": int(index), "h12": h12,
                    "validated": False, "h11": None, "residual": None} for index in selected]
        for _, h11 in (pair for pair in HODGE_PAIRS if pair[0] == h12):
            pending = np.flatnonzero(~matched)
            if not len(pending):
                break
            mean = np.asarray(log[f"{h12} {h11}"]["moduli_mean"], dtype=np.complex128)
            print(f"Checking h12={h12}, h11={h11}: {len(pending)} unresolved rows", flush=True)
            model = get_geometry_model(h12, h11, geometry_log=log)
            result = validate_vacua(
                flux[pending], np.tile(mean[:-1], (len(pending), 1)), np.full(len(pending), mean[-1]),
                h12=h12, h11=h11, model=model, tol=tol,
            )
            successes = np.flatnonzero(result["valid_mask"])
            for local in successes:
                index = int(pending[local])
                matched[index] = True
                records[index].update(validated=True, h11=h11, residual=float(result["residuals"][local]))
            attempts.append({"h12": h12, "h11": h11, **result["counts"]})
            print(f"Validated {int(matched.sum())}/{count} selected h12={h12} rows so far", flush=True)
            del model
            jax.clear_caches()
        rows.extend(records)
    validated = sum(row["validated"] for row in rows)
    report = {
        "num_datapoints": len(rows), "validated": validated, "unresolved": len(rows) - validated,
        "seed": seed, "tol": tol, "solver": "scipy_root", "jax_version": jax.__version__,
        "initializer": "per-geometry moduli_mean from geometry_log_degree_10.p",
        "geometry_selection": "try each candidate with matching h12 until validated",
        "note": "Source rows contain only fluxes. Unresolved rows may need better initializers; they are not certified invalid.",
        "elapsed_seconds": time.time() - started, "attempts": attempts, "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Validated {validated}/{len(rows)} reference datapoints; saved {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=90261001)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tol", type=float, default=1e-10)
    validate_dataset(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
