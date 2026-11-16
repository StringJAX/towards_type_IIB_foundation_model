"""Diffusion vacuum validation using JAXVacua and SciPy directly.

No transformer sampling, conversion, or evaluation code is imported here.
Inputs are integer fluxes and complex physical initializers. JAX is loaded
lazily so reading samples and sampling with PyTorch do not initialize it.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import pickle
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GEOMETRY_LOG = REPO_ROOT / "transformer/data/data_multiple/geometry_log_degree_10.p"


def cpu_jax():
    """Enable float64 and return JAX; solver calls explicitly use the CPU."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    jax.config.update("jax_enable_x64", True)
    return jax


def load_geometry_log(path: str | Path = DEFAULT_GEOMETRY_LOG) -> dict:
    """Read the trusted geometry-ID and mean-initializer data file."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def get_geometry_model(h12: int, h11: int, *, geometry_log: dict | None = None,
                       maximum_degree: int = 10):
    """Construct the exact JAXVacua geometry identified by its Hodge pair."""
    log = load_geometry_log() if geometry_log is None else geometry_log
    key = f"{h12} {h11}"
    if key not in log:
        raise ValueError(f"Unknown geometry {key}")
    jax = cpu_jax()
    import jaxvacua
    with jax.default_device(jax.devices("cpu")[0]):
        model = jaxvacua.FluxVacuaFinder(
            h12=h12, model_ID=int(log[key]["model_ID"]), maximum_degree=maximum_degree,
        )
    actual = (int(model.lcs_tree.h12), int(model.lcs_tree.h11))
    if actual != (h12, h11):
        raise ValueError(f"Requested geometry {(h12, h11)}, but JAXVacua loaded {actual}")
    return model


def complex_to_real(z: np.ndarray, tau: complex) -> np.ndarray:
    """Interleave Re/Im(z), followed by Re/Im(tau), for the root solver."""
    return np.concatenate((np.column_stack((z.real, z.imag)).ravel(), [tau.real, tau.imag]))


def validate_vacua(
    flux: np.ndarray, z: np.ndarray, tau: np.ndarray, *,
    h12: int, h11: int, model: Any = None, tol: float = 1e-10,
    use_newton: bool = False, verbose: bool = False,
) -> dict[str, Any]:
    """Solve F-terms, canonicalize, check physical domains, and deduplicate.

    The sequential checks are: finite inputs; 0 < N_flux <= Q; an F-term
    residual max(abs(DW_x)) < tol; mapping with JAXVacua's map_to_fd;
    Im(tau)>0, abs(Re(tau))<=1/2, abs(tau)>=1; positive Kähler-cone
    hyperplane products; abs(Re(z))<=1/2; and a residual check after mapping.
    Domain boundaries allow 1e-8 numerical tolerance, matching map_to_fd.
    Fluxes are deduplicated only after validation, preserving the first row.

    SciPy uses analytic dDW_x and hybr (tol=1e-3*tol). A solution is accepted
    by its measured residual, including an already-solved starting point,
    rather than relying on SciPy's termination flag. ``use_newton`` selects
    JAXVacua's Newton solver instead. Both run in float64 on the CPU.

    Returns unique valid ``flux``, ``z``, ``tau``, and their original row
    ``indices``; an input-length ``valid_mask`` before deduplication; input-
    length ``residuals`` (infinity where unsolved); and per-stage ``counts``.
    Empty batches and batches with no valid solutions return empty arrays.
    """
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
    flux = np.asarray(flux)
    z = np.asarray(z, dtype=np.complex128)
    tau = np.asarray(tau, dtype=np.complex128)
    if flux.ndim != 2 or flux.shape[1] != 4 * (h12 + 1):
        raise ValueError("Flux shape disagrees with h12")
    count = len(flux)
    if z.shape != (count, h12) or tau.shape != (count,):
        raise ValueError("Initializer shapes disagree with flux and h12")
    if not np.isfinite(flux).all() or not np.equal(flux, np.rint(flux)).all() or np.any(np.abs(flux.astype(float)) >= 2**63):
        raise ValueError("Fluxes must be finite int64-representable integers")
    flux = flux.astype(np.int64)
    counts = {"num_samples": count, "failed_initializer": 0, "failed_tadpole": 0,
              "failed_to_solve": 0, "failed_tau_FD": 0, "failed_LCS": 0,
              "failed_m_FD": 0, "failed_mapped_residual": 0, "valid": 0, "valid_unique": 0}
    valid_mask = np.zeros(count, dtype=bool)
    residuals = np.full(count, np.inf)
    kept_flux, kept_z, kept_tau, kept_indices = [], [], [], []
    if count:
        jax = cpu_jax()
        from scipy.optimize import root
        model = get_geometry_model(h12, h11) if model is None else model
        with jax.default_device(jax.devices("cpu")[0]):
            tadpoles = np.asarray(model.tadpole(flux)).reshape(-1)
            bound = float(model.Q())
            hyperplanes = np.asarray(model.lcs_tree.hyperplanes)
            for index, (f, zi, ti) in enumerate(zip(flux, z, tau)):
                if not np.isfinite(zi).all() or not np.isfinite(ti):
                    counts["failed_initializer"] += 1
                    continue
                if not (0 < tadpoles[index] <= bound):
                    counts["failed_tadpole"] += 1
                    continue
                x0 = complex_to_real(zi, ti)
                initial_residual = float(np.max(np.abs(model.DW_x(x0, f))))
                if np.isfinite(initial_residual) and initial_residual < tol:
                    x = x0
                elif use_newton:
                    zn, tn, _ = model.newton_method_flux_vacua(
                        zi, ti, f, mode="SUSY", solver_mode="complex",
                        max_iters=50, tol=tol, step_size_Newton=1.0,
                    )
                    x = complex_to_real(np.asarray(zn), complex(tn))
                else:
                    solution = root(model.DW_x, x0, args=(f,), jac=model.dDW_x,
                                    method="hybr", tol=tol * 1e-3)
                    x = solution.x
                residual = float(np.max(np.abs(model.DW_x(x, f)))) if np.isfinite(x).all() else np.inf
                residuals[index] = residual if np.isfinite(residual) else np.inf
                if residuals[index] >= tol:
                    counts["failed_to_solve"] += 1
                    continue
                zs, ts = x[:-2:2] + 1j * x[1:-2:2], complex(x[-2], x[-1])
                if ts.imag <= 0:
                    counts["failed_tau_FD"] += 1
                    continue
                zm, tm, fm = model.map_to_fd(zs, ts, f)
                zm, tm, fm = np.asarray(zm), complex(tm), np.asarray(fm)
                boundary = 1e-8
                if not np.isfinite(tm) or tm.imag <= 0 or abs(tm.real) > 0.5 + boundary or abs(tm) < 1 - boundary:
                    counts["failed_tau_FD"] += 1
                    continue
                if not np.isfinite(zm).all() or not np.all(hyperplanes @ zm.imag > 0):
                    counts["failed_LCS"] += 1
                    continue
                if np.any(np.abs(zm.real) > 0.5 + boundary):
                    counts["failed_m_FD"] += 1
                    continue
                mapped_residual = float(np.max(np.abs(model.DW_x(complex_to_real(zm, tm), fm))))
                if not np.isfinite(mapped_residual) or mapped_residual >= tol:
                    counts["failed_mapped_residual"] += 1
                    continue
                residuals[index] = mapped_residual
                valid_mask[index] = True
                kept_flux.append(fm.astype(np.int64))
                kept_z.append(zm)
                kept_tau.append(tm)
                kept_indices.append(index)
                if verbose:
                    print(f"{h12} {h11}: row {index} valid; residual={mapped_residual:.3g}", flush=True)
    valid_flux = np.asarray(kept_flux, dtype=np.int64).reshape(-1, 4 * (h12 + 1))
    _, unique = np.unique(valid_flux, axis=0, return_index=True)
    unique.sort()
    counts["valid"] = int(valid_mask.sum())
    counts["valid_unique"] = len(unique)
    return {
        "flux": valid_flux[unique],
        "z": np.asarray(kept_z, dtype=np.complex128).reshape(-1, h12)[unique],
        "tau": np.asarray(kept_tau, dtype=np.complex128)[unique],
        "indices": np.asarray(kept_indices, dtype=np.int64)[unique],
        "valid_mask": valid_mask, "residuals": residuals, "counts": counts,
    }
