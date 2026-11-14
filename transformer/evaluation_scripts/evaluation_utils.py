"""Model loading, sample processing and validity checks for generated flux vacua.

Generated token sequences are parsed into fluxes, complex-structure moduli and the
axio-dilaton (:func:`process_samples`, :func:`convert_samples_to_data`).
:func:`validate_vacua` then applies the validity checks: the D3 tadpole bound, solving
the F-term conditions with jaxvacua (``scipy.optimize.root`` or Newton's method),
mapping to the fundamental domain, the cuts on the axio-dilaton, the Kähler cone and the
axions, and the removal of duplicate fluxes. Novelty with respect to the training data is
checked by the callers (e.g. ``evaluate.py``), not in this module.
"""

if __package__ and "." in __package__:
    from ..jax_transformer.transformer_model import DecoderModel, EncoderDecoderModel
    from ..jax_transformer.utils import load_model
    from ..jax_transformer.data_loader import DictTokenizer
    from ..data_scripts.tokenizer_utils import token_to_data
else:
    from jax_transformer.transformer_model import DecoderModel, EncoderDecoderModel
    from jax_transformer.utils import load_model
    from jax_transformer.data_loader import DictTokenizer
    from data_scripts.tokenizer_utils import token_to_data
from flax.training import train_state
import optax
from scipy.optimize import root
import jax
import jax.numpy as jnp
import json
import numpy as np
from tqdm import tqdm
import os
from collections.abc import Sequence
from typing import Any


def load_config(RUN_ID: int, cwd: str) -> dict[str, Any]:
    r"""Loads config of trained model.

    Args:
        RUN_ID (int): Specifies the trained model.
        cwd (str): Path to directory containing the folder 'logs'.

    Returns:
        config (dict): Model config.

    Raises:
        AssertionError: If config['optimizer'] contains unsupported optimizer.
    """
    with open(cwd + f"/logs/run_{RUN_ID}/run_{RUN_ID}.json") as f:
        config = json.load(f)
    assert config["optimizer"] in [
        "adam",
        "adamw",
    ], f"Currently only adam and adamw are possible as optimizers, but got {config['optimizer']}."
    return config


def load_model_from_config(
    RUN_ID: int,
    config: dict[str, Any],
    load_ckpt: str,
    cwd: str,
    override_opt: bool = False,
    GRPO_ID: str | None = None,
) -> tuple[train_state.TrainState, DecoderModel | EncoderDecoderModel, DictTokenizer]:
    r"""Loads trained model from respective config.

    Args:
        RUN_ID (int): Specifies the trained model.
        config (dict): Model config.
        load_ckpt (str): Name of checkpoint to be loaded.
        cwd (str): Path to directory containing the folder 'checkpoints'.
        override_opt (bool): Needed for compatibility with older checkpoints. Defaults to False.
        GRPO_ID (str | None): ID of the GRPO run. Used only if ``override_opt`` is True and
            ``RUN_ID > 35``, to read the learning rate from
            'logs/run_{RUN_ID}/GRPO_{GRPO_ID}/log.json' under ``cwd``. Defaults to None.

    Returns:
        state (train_state.TrainState): Loaded train state.
        model (nn.Module): Loaded model.
        tokenizer (data_loader.DictTokenizer): Tokenizer used during training.
    """
    models = {"decoder": DecoderModel, "encoder-decoder": EncoderDecoderModel}
    optimizers = {"adam": optax.adam, "adamw": optax.adamw}
    schedules = {"cosine": optax.warmup_cosine_decay_schedule}

    tokenizer = DictTokenizer(config["token_dict"])
    optimizer = optimizers[config["optimizer"]]
    if config["lr_schedule"] is not None:
        schedule = schedules[config["lr_schedule"]]
        decay_steps = config["decay_steps"]
        lr_schdl = schedule(
            init_value=0.0,
            peak_value=config["lr"],
            warmup_steps=config["warmup_steps"],
            decay_steps=decay_steps,
        )
        opt = optimizer(learning_rate=lr_schdl, b1=0.9, b2=0.98, eps=1e-9)
    else:
        opt = optimizer(learning_rate=config["lr"], b1=0.9, b2=0.98, eps=1e-9)

    # Needed for compatibility with older checkpoints
    if override_opt:
        if RUN_ID <= 35:
            opt = optax.adamw(
                learning_rate=1e-5,
                weight_decay=0.00,
            )
        else:
            with open(cwd + f"/logs/run_{RUN_ID}/GRPO_{GRPO_ID}/log.json") as f:
                grpo_config = json.load(f)
            lr = grpo_config["lr"]
            opt = optax.adam(learning_rate=lr)
            if RUN_ID >= 46 or (RUN_ID == 45 and int(GRPO_ID) == 3):
                opt = optax.chain(optax.clip_by_global_norm(1.0), opt)

    # Setting args for model
    args_model = {
        "num_layers": config["num_layers"],
        "embed_dim": config["embed_dim"],
        "num_heads": config["num_heads"],
        "vocab_size": tokenizer.vocab_size,
        "dropout": config["dropout"],
    }
    if config["model"] == "encoder-decoder":
        max_len_in = config["max_len_in"]
        max_len_out = config["max_len_out"]
        update = {"max_len_in": max_len_in, "max_len_out": max_len_out}
        args_model.update(update)
    elif config["model"] == "decoder":
        max_len = config["max_len"]
        update = {"max_len": max_len}
        args_model.update(update)

    # Loading model
    model = models[config["model"]](**args_model)
    params_key = jax.random.key(42)
    if config["model"] == "decoder":
        inp = jnp.ones((1, model.max_len), dtype=jnp.int32)
        params = model.init({"params": params_key}, input_ids=inp, training=False)[
            "params"
        ]
    elif config["model"] == "encoder-decoder":
        inp = jnp.ones((1, model.max_len_in), dtype=jnp.int32)
        out = jnp.ones((1, model.max_len_out), dtype=jnp.int32)
        mask_enc = jnp.ones((1, model.max_len_in), dtype=jnp.int32)
        mask_dec = jnp.ones((1, model.max_len_out), dtype=jnp.int32)
        params = model.init(
            {"params": params_key},
            input_ids=inp,
            output_ids=out,
            mask_enc=mask_enc,
            mask_dec=mask_dec,
            training=False,
        )["params"]
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opt)
    state = load_model(state, cwd + f"/checkpoints/run_{RUN_ID}/" + load_ckpt)
    return state, model, tokenizer


def to_old_a_matrix(model: Any) -> Any:
    r"""Map a-matrix of jaxvacua model to previously used convention. Needed to verify old models.

    Sets ``a_matrix[i][j] = intnums[i][j][j] / 2`` on ``model.lcs_tree``; the model is
    modified in place.

    Args:
        model (jvc.FluxVacuaFinder): Jaxvacua model specifying geometry.

    Returns:
        model (jvc.FluxVacuaFinder): Jaxvacua model with old a-matrix.
    """
    a_matrix = np.zeros((model.h12, model.h12))
    for I1 in range(model.h12):
        for I2 in range(model.h12):
            a_matrix[I1][I2] = model.lcs_tree.intnums[I1][I2][I2] / 2.0
    model.lcs_tree.a_matrix = a_matrix
    return model


def solve(
    fluxes: jax.Array,
    moduli0: jax.Array,
    tau0: jax.Array,
    model: Any,
    tol: float,
    use_scipy: bool = True,
    verbose: bool = False,
    debug: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Solving the F-term conditions for given input and model.

    Args:
        fluxes (jnp.ndarray([batch_size, 4*(model.h12+1)])): Batch of fluxes.
        moduli0 (jnp.ndarray([batch_size, model.h12])): Batch of moduli used as initial guesses.
        tau0 (jnp.ndarray([batch_size])): Batch of axio-dilaton used as initial guesses.
        model (jvc.FluxVacuaFinder): Jaxvacua model specifying geometry.
        tol (float): Tolerance on the F-term residual. With scipy, root() is called with
            ``tol=1e-3 * tol`` and a point counts as solved if root() reports success and
            ``max(abs(model.DW_x))`` at the solution is below ``tol``. With Newton's method,
            a point counts as solved if the residual returned by
            ``model.newton_method_flux_vacua`` is below ``tol``.
        use_scipy (bool): Use scipy.optimize.root() (method 'hybr') to find solutions;
            otherwise use jaxvacua's Newton method. Defaults to True.
        verbose (bool): Show solving progress. Defaults to False.
        debug (bool): Log if terminated. Defaults to False.

    Returns:
        fluxes (jnp.ndarray([solved_size, 4*(model.h12+1)])): Fluxes with found solution.
        moduli (jnp.ndarray([solved_size, model.h12])): Moduli at solution.
        tau (jnp.ndarray([solved_size])): Axio-dilaton at solution.
        idx_solved (jnp.ndarray([batch_size])): Boolean mask over input batch, indicating position of solved fluxes.

    Raises:
        NotImplementedError: If ``moduli0`` is one-dimensional and ``use_scipy`` is True.
        AssertionError: If ``moduli0`` has neither one nor two axes.
    """
    if debug:
        print("Solving...")

    if moduli0.ndim == 1:
        if use_scipy:
            raise NotImplementedError
        else:
            solver = jax.vmap(
                lambda fluxes0: model.newton_method_flux_vacua(
                    moduli0,
                    tau0,
                    fluxes0,
                    mode="SUSY",
                    solver_mode="complex",
                    tol=tol,
                    step_size_Newton=1.0,
                )
            )
            moduli, tau, res = solver(fluxes)
    elif moduli0.ndim == 2:
        if use_scipy:
            moduli0_np = np.asarray(moduli0)
            tau0_np = np.asarray(tau0)
            fluxes_np = np.asarray(fluxes)
            batch_size = len(moduli0_np)
            moduli0_np = np.stack([moduli0_np.real, moduli0_np.imag], axis=-1).reshape(
                batch_size, -1
            )
            tau0_np = np.stack([tau0_np.real, tau0_np.imag], axis=-1)
            x0_np = np.concatenate([moduli0_np, tau0_np], axis=-1)
            x = np.empty_like(x0_np)
            idx_solved = np.empty(batch_size, dtype=bool)
            if verbose:
                for i in tqdm(range(len(fluxes_np))):
                    flux_iter = fluxes_np[i]
                    x0_iter = x0_np[i]
                    sol = root(
                        model.DW_x,
                        x0_iter,
                        args=(flux_iter,),
                        jac=model.dDW_x,
                        method="hybr",
                        tol=tol * 1e-3,
                    )
                    res = np.max(np.abs(model.DW_x(sol.x, flux_iter)))
                    idx_solved[i] = sol.success and (res < tol)
                    x[i] = sol.x
            else:
                for i, (flux_iter, x0_iter) in enumerate(zip(fluxes_np, x0_np)):
                    sol = root(
                        model.DW_x,
                        x0_iter,
                        args=(flux_iter,),
                        jac=model.dDW_x,
                        method="hybr",
                        tol=tol * 1e-3,
                    )
                    res = np.max(np.abs(model.DW_x(sol.x, flux_iter)))
                    idx_solved[i] = sol.success and (res < tol)
                    x[i] = sol.x
            moduli = x[:, 0:-2:2] + 1.0j * x[:, 1:-2:2]
            tau = x[:, -2] + 1.0j * x[:, -1]
        else:
            with jax.default_device(jax.devices("cpu")[0]):
                solver = lambda moduli0, tau0, fluxes0: model.newton_method_flux_vacua(
                    moduli0,
                    tau0,
                    fluxes0,
                    mode="SUSY",
                    solver_mode="complex",
                    max_iters=50,
                    tol=tol,
                    step_size_Newton=1.0,
                )
                moduli, tau, res = solver(moduli0, tau0, fluxes)
                idx_solved = jnp.abs(res) < tol
    else:
        raise AssertionError(
            f"moduli0 must be have one or two axis but got shape {moduli0}."
        )
    # Filter for solved
    fluxes, moduli, tau = (
        fluxes[idx_solved],
        moduli[idx_solved],
        tau[idx_solved],
    )
    if debug:
        print("Solving terminated.")
    return (
        jnp.asarray(fluxes),
        jnp.asarray(moduli),
        jnp.asarray(tau),
        jnp.asarray(idx_solved),
    )


def map_to_FD(
    fluxes: jax.Array,
    moduli: jax.Array,
    tau: jax.Array,
    model: Any,
    debug: bool,
    debug_deep: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Maps fluxes, moduli and axio-dilaton to fundamental domain of SL(2,Z) and monodromy transformations. If the mapping is not possible, the original data points are returned.

    Args:
        fluxes (jnp.ndarray([batch_size, 4*(model.h12+1)])): Batch of fluxes.
        moduli (jnp.ndarray([batch_size, model.h12])): Batch of moduli.
        tau (jnp.ndarray([batch_size])): Batch of axio-dilaton.
        model (jvc.FluxVacuaFinder): Jaxvacua model specifying geometry.
        debug (bool): Log if terminated.
        debug_deep (bool): Print progress.

    Returns:
        fluxes (jnp.ndarray([batch_size, 4*(model.h12+1)])): Fluxes within fundamental domain.
        moduli (jnp.ndarray([batch_size, model.h12])): Moduli within fundamental domain.
        tau (jnp.ndarray([batch_size])): Axio-dilaton within fundamental domain.
    """
    moduli_FD_lst = []
    tau_FD_lst = []
    fluxes_FD_lst = []
    if debug:
        print("Mapping to FD...")
    if debug_deep:
        for i in tqdm(range(len(fluxes))):
            moduli_iter, tau_iter, flux_iter = moduli[i], tau[i], fluxes[i]
            moduli_FD, tau_FD, flux_FD = model.map_to_fd(
                moduli_iter, tau_iter, flux_iter
            )
            tau_FD_lst.append(np.asarray(tau_FD))
            moduli_FD_lst.append(np.asarray(moduli_FD)[None, :])
            fluxes_FD_lst.append(np.asarray(flux_FD[None, :]))
    else:
        for moduli_iter, tau_iter, flux_iter in zip(moduli, tau, fluxes):
            moduli_FD, tau_FD, flux_FD = model.map_to_fd(
                moduli_iter, tau_iter, flux_iter
            )
            tau_FD_lst.append(np.asarray(tau_FD))
            moduli_FD_lst.append(np.asarray(moduli_FD)[None, :])
            fluxes_FD_lst.append(np.asarray(flux_FD[None, :]))

    fluxes = np.concatenate(fluxes_FD_lst, axis=0)
    moduli = np.concatenate(moduli_FD_lst, axis=0)
    tau = np.array(tau_FD_lst)
    if debug:
        print("Mapping terminated.")
    return jnp.asarray(fluxes), jnp.asarray(moduli), jnp.asarray(tau)


def check_tadpole(
    model: Any, fluxes: jax.Array, debug: bool = False
) -> tuple[jax.Array, jax.Array]:
    r"""Checks tadpole bound for set of fluxes.

    A flux vector is kept if ``0 < N_flux <= Q``, with ``N_flux = model.tadpole(fluxes)``
    and ``Q = model.Q()``.

    Args:
        model (jvc.FluxVacuaFinder): Jaxvacua model specifying geometry.
        fluxes (jnp.ndarray([batch_size, 4*(model.h12+1)])): Batch of fluxes.
        debug (bool): Log if terminated. Defaults to False.

    Returns:
        fluxes (jnp.ndarray([valid_size, 4*(model.h12+1)])): Fluxes within tadpole bound.
        idx_valid (jnp.ndarray([batch_size])): Boolean mask over input batch, indicating position of valid fluxes.
    """
    if debug:
        print("Checking tadpole...")
    Q3 = model.Q()
    nflux = model.tadpole(fluxes)
    idx_valid = (nflux > 0) & (nflux <= Q3)
    fluxes = fluxes[idx_valid]
    if debug:
        print("Check terminated.")
    return fluxes, idx_valid


def check_FD(
    fluxes: jax.Array,
    moduli: jax.Array,
    tau: jax.Array,
    model: Any = None,
    debug: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Checks if input in fundamental domain of SL(2,Z).

    A point is kept if ``Im(tau) > 0.83`` and ``|Re(tau)| < 0.5``.

    Args:
        fluxes (jnp.ndarray([batch_size, 4*(model.h12+1)])): Batch of fluxes.
        moduli (jnp.ndarray([batch_size, model.h12])): Batch of moduli.
        tau (jnp.ndarray([batch_size])): Batch of axio-dilaton.
        debug (bool): Log if terminated. Defaults to False.
        model (jvc.FluxVacuaFinder): Needed for consistency (will be changed). Defaults to None.

    Returns:
        fluxes (jnp.ndarray([valid_size, 4*(model.h12+1)])): Fluxes within fundamental domain.
        moduli (jnp.ndarray([valid_size, model.h12])): Moduli within fundamental domain.
        tau (jnp.ndarray([valid_size])): Axio-dilaton within fundamental domain.
        idx_valid (jnp.ndarray([batch_size])): Boolean mask over input batch, indicating position of valid fluxes.
    """
    if debug:
        print("Checking tau FD...")
    idx_tau = (tau.imag > 0.83) & (jnp.abs(tau.real) < 0.5)
    fluxes, moduli, tau = fluxes[idx_tau], moduli[idx_tau], tau[idx_tau]
    if debug:
        print("Check terminated.")
    return fluxes, moduli, tau, idx_tau


def check_LCS(
    fluxes: jax.Array,
    moduli: jax.Array,
    tau: jax.Array,
    model: Any,
    debug: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Checks if input in LCS region.

    A point is kept if ``Im(z)`` lies strictly inside the Kähler cone given by
    ``model.lcs_tree.hyperplanes``, i.e. every entry of ``hyperplanes @ Im(z)`` is positive.

    Args:
        fluxes (jnp.ndarray([batch_size, 4*(model.h12+1)])): Batch of fluxes.
        moduli (jnp.ndarray([batch_size, model.h12])): Batch of moduli.
        tau (jnp.ndarray([batch_size])): Batch of axio-dilaton.
        model (jvc.FluxVacuaFinder): Jaxvacua model specifying geometry.
        debug (bool): Log if terminated. Defaults to False.

    Returns:
        fluxes (jnp.ndarray([valid_size, 4*(model.h12+1)])): Fluxes within LCS region.
        moduli (jnp.ndarray([valid_size, model.h12])): Moduli within LCS region.
        tau (jnp.ndarray([valid_size])): Axio-dilaton within LCS region.
        idx_cone (jnp.ndarray([batch_size])): Boolean mask over input batch, indicating position of valid fluxes.
    """
    if debug:
        print("Checking LCS region...")
    idx_cone = jnp.all(moduli.imag @ model.lcs_tree.hyperplanes.T > 0, axis=1)
    moduli, tau, fluxes = moduli[idx_cone], tau[idx_cone], fluxes[idx_cone]
    if debug:
        print("Check terminated.")
    return fluxes, moduli, tau, idx_cone


def check_monodromy_FD(
    fluxes: jax.Array,
    moduli: jax.Array,
    tau: jax.Array,
    model: Any = None,
    debug: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Checks if input in fundamental domain of monodromy transformations.

    A point is kept if ``|Re(z^i)| < 0.5`` for every modulus ``z^i``.

    Args:
        fluxes (jnp.ndarray([batch_size, 4*(model.h12+1)])): Batch of fluxes.
        moduli (jnp.ndarray([batch_size, model.h12])): Batch of moduli.
        tau (jnp.ndarray([batch_size])): Batch of axio-dilaton.
        debug (bool): Log if terminated. Defaults to False.
        model (jvc.FluxVacuaFinder): Needed for consistency (will be changed). Defaults to None.

    Returns:
        fluxes (jnp.ndarray([valid_size, 4*(model.h12+1)])): Fluxes within fundamental domain.
        moduli (jnp.ndarray([valid_size, model.h12])): Moduli within fundamental domain.
        tau (jnp.ndarray([valid_size])): Axio-dilaton within fundamental domain.
        idx_domain (jnp.ndarray([batch_size])): Boolean mask over input batch, indicating position of valid fluxes.
    """
    if debug:
        print("Checking monodromy FD...")
    idx_domain = jnp.all(jnp.abs(moduli.real) < 0.5, axis=-1)
    moduli, tau, fluxes = (
        moduli[idx_domain],
        tau[idx_domain],
        fluxes[idx_domain],
    )
    if debug:
        print("Check terminated.")
    return fluxes, moduli, tau, idx_domain


def get_unique(
    fluxes: jax.Array,
    moduli: jax.Array,
    tau: jax.Array,
    idx_samples: jax.Array,
    debug: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Return numerically unique flux vacua.

    Uniqueness is decided by the flux vector alone. For repeated fluxes the first
    occurrence is kept, and the output is sorted lexicographically by fluxes (as returned
    by ``jnp.unique``).

    Args:
        fluxes (jnp.ndarray([batch_size, 4*(model.h12+1)])): Batch of fluxes.
        moduli (jnp.ndarray([batch_size, model.h12])): Batch of moduli.
        tau (jnp.ndarray([batch_size])): Batch of axio-dilaton.
        idx_samples (jnp.ndarray([batch_size])): Integer positions of the inputs in the original sample array.
        debug (bool): Log if terminated. Defaults to False.

    Returns:
        fluxes (jnp.ndarray([unique_size, 4*(model.h12+1)])): Unique fluxes.
        moduli (jnp.ndarray([unique_size, model.h12])): Unique moduli.
        tau (jnp.ndarray([unique_size])): Unique axio-dilaton
        idx_samples (jnp.ndarray([unique_size])): Positions of the unique vacua in the original sample array.
    """
    if debug:
        print("Filtering unique...")
    fluxes, idx_unique = jnp.unique(fluxes, axis=0, return_index=True)
    idx_samples = idx_samples[idx_unique]
    moduli, tau = moduli[idx_unique], tau[idx_unique]
    if debug:
        print("Filtering terminated.")
    return fluxes, moduli, tau, idx_samples


def validate_vacua(
    fluxes: jax.Array,
    moduli: jax.Array,
    tau: jax.Array,
    idx_samples: jax.Array,
    model: Any,
    out: dict[str, Any],
    tol: float,
    use_scipy: bool,
    plot_trajectory: bool = False,
    debug: bool = False,
    debug_deep: bool = False,
) -> tuple[
    jax.Array | None, jax.Array | None, jax.Array | None, jax.Array | None, dict[str, Any]
]:
    r"""Applies all validity checks to a batch of generated flux vacua.

    The steps run in this order, each on the survivors of the previous one: the tadpole
    bound (:func:`check_tadpole`); solving the F-term conditions with the generated moduli
    and axio-dilaton as initial guesses (:func:`solve`); mapping to the fundamental domain
    (:func:`map_to_FD`); :func:`check_FD`, :func:`check_LCS` and :func:`check_monodromy_FD`;
    and the removal of duplicate fluxes (:func:`get_unique`). The counts are written to
    ``out`` under ``"failed_tadpole"``, ``"failed_to_solve"``, ``"failed_tau_FD"``,
    ``"failed_LCS"``, ``"failed_m_FD"``, ``"valid"`` (number passing all checks) and
    ``"valid_unique"`` (number of distinct fluxes among them). Keys of steps that are not
    reached are left unchanged, except ``"valid"`` and ``"valid_unique"``, which are then
    set to 0.

    Args:
        fluxes (jnp.ndarray([batch_size, 4*(model.h12+1)])): Batch of generated fluxes.
        moduli (jnp.ndarray([batch_size, model.h12])): Generated moduli, used as initial guesses.
        tau (jnp.ndarray([batch_size])): Generated axio-dilaton, used as initial guesses.
        idx_samples (jnp.ndarray([batch_size])): Integer positions of the inputs in the original sample array.
        model (jvc.FluxVacuaFinder): Jaxvacua model specifying geometry.
        out (dict): Dictionary for the console output; updated in place and returned.
        tol (float): Tolerance passed to :func:`solve`.
        use_scipy (bool): Use scipy.optimize.root() instead of jaxvacua's Newton method.
        plot_trajectory (bool): If True and some points are solved, save the initial moduli of
            the solved points (``z0``), their solutions before the mapping to the fundamental
            domain (``solver_raw_z``) and a mask of those passing the three checks after
            the mapping (``solver_is_valid``) to 'data/sol_trajectories/{h12}_{h11}_trajectory.npz' in the
            current working directory. Defaults to False.
        debug (bool): Log the start and end of each step. Defaults to False.
        debug_deep (bool): Show progress bars while solving and mapping. Defaults to False.

    Returns:
        tuple: ``(fluxes, moduli, tau, idx_samples, out)`` for the unique valid vacua: fluxes,
            moduli and axio-dilaton at the solution, mapped to the fundamental domain, their
            positions in the original sample array, and the updated ``out``. The first four
            entries are None if no vacuum passes all checks.
    """
    # Check tadpole
    fluxes, idx_valid_tadpole = check_tadpole(model, fluxes, debug)
    idx_samples = idx_samples[idx_valid_tadpole]
    out["failed_tadpole"] = jnp.sum(~idx_valid_tadpole)

    if jnp.any(idx_valid_tadpole):
        # Solve F-term conditions
        moduli, tau = moduli[idx_valid_tadpole], tau[idx_valid_tadpole]
        z0 = moduli
        tau0 = tau
        fluxes, moduli, tau, idx_solved = solve(
            fluxes, z0, tau0, model, tol, use_scipy, debug_deep, debug
        )
        out["failed_to_solve"] = jnp.sum(~idx_solved)

        if plot_trajectory:
            plot_trajectory_dict = {"z0": z0[idx_solved], "solver_raw_z": moduli, "solver_is_valid": np.zeros(len(moduli), dtype=bool)}

        # If solutions were found, check conditions
        if jnp.any(idx_solved):
            idx_samples = idx_samples[idx_solved]
            # Map to FD
            fluxes, moduli, tau = map_to_FD(
                fluxes, moduli, tau, model, debug, debug_deep
            )

            # Perform checks
            check_fcts = [check_FD, check_LCS, check_monodromy_FD]
            check_failed = []

            if plot_trajectory:
                idx_check_lst = []

            for fct in check_fcts:
                fluxes, moduli, tau, idx_check = fct(fluxes, moduli, tau, model, debug)
                idx_samples = idx_samples[idx_check]
                check_failed.append(jnp.sum(~idx_check))
                if plot_trajectory:
                    idx_check_lst.append(idx_check)

            if plot_trajectory:
                valid_indices = np.arange(len(plot_trajectory_dict["solver_is_valid"]))
                for idx_check in idx_check_lst:
                    valid_indices = valid_indices[np.asarray(idx_check, dtype=bool)]
                plot_trajectory_dict["solver_is_valid"][valid_indices] = True
                np.savez(os.getcwd() + f"/data/sol_trajectories/{model.lcs_tree.h12}_{model.lcs_tree.h11}_trajectory.npz", z0=plot_trajectory_dict["z0"], solver_raw_z=plot_trajectory_dict["solver_raw_z"], solver_is_valid=plot_trajectory_dict["solver_is_valid"])

            # Update console output
            out["valid"] = len(fluxes)
            check_keys = ["tau_FD", "LCS", "m_FD"]
            for key, fail_rate in zip(check_keys, check_failed):
                out["failed_" + key] = fail_rate

            # Filter for unique
            fluxes, moduli, tau, idx_samples = get_unique(
                fluxes, moduli, tau, idx_samples, debug
            )
            out["valid_unique"] = len(fluxes)

            if len(fluxes) == 0:
                return None, None, None, None, out
            else:
                return fluxes, moduli, tau, idx_samples, out
        else:
            out["valid_unique"] = 0
            out["valid"] = 0
            return None, None, None, None, out
    else:
        out["valid_unique"] = 0
        out["valid"] = 0
        return None, None, None, None, out


def convert_samples_to_data(
    samples: list, discr: float, gs_inv: float | None = None
) -> tuple:
    r"""Converts processed samples to fluxes, complex moduli and axio-dilaton.

    The generated moduli and axio-dilaton are discretised; their entries are divided by
    ``discr``.

    Args:
        samples (list): ``[fluxes, moduli, tau]`` for one prompt, as returned per prompt by
            :func:`process_samples`. ``fluxes`` has shape [num_samples, 4*(h12+1)];
            ``moduli`` has shape [num_samples, 2*h12], with the real parts in the first h12
            columns and the imaginary parts in the last h12; ``tau`` has shape
            [num_samples, 2], holding Re(tau) and Im(tau), or [num_samples, 1], holding only
            Re(tau), if ``gs_inv`` is given.
        discr (float): Discretisation factor.
        gs_inv (float | None): Discretised value of the 1/g_s prompt (``discr`` times 1/g_s).
            If given, Im(tau) is set to ``gs_inv / discr`` and only the first column of
            ``tau`` is used. Defaults to None.

    Returns:
        tuple: ``(fluxes, moduli, tau)``: the fluxes, unchanged, with shape
            [num_samples, 4*(h12+1)]; the complex moduli, shape [num_samples, h12]; and the
            complex axio-dilaton, shape [num_samples].
    """
    fluxes = samples[0]
    moduli = samples[1]
    moduli_real, moduli_imag = jnp.split(moduli, 2, axis=-1)
    moduli = (moduli_real + 1.0j * moduli_imag) / discr
    tau = samples[2]
    if gs_inv is not None:
        tau = (tau[:, 0] + 1.0j * gs_inv) / discr
    else:
        tau = (tau[:, 0] + 1.0j * tau[:, 1]) / discr
    return fluxes, moduli, tau


def process_sequence(
    sequence: np.ndarray | Sequence[str],
    tokenizer: DictTokenizer,
    conds: int | np.ndarray,
    split_tokens: Sequence[str] | np.ndarray | None = None,
) -> list[np.ndarray] | np.ndarray | None:
    r"""Processes single token sequence.

    Without ``split_tokens``, all tokens after ``<BOS>`` are converted to integers. With
    ``split_tokens``, the part between ``<BOS>`` and ``<EOS>`` is split at the split tokens
    and each part is converted separately.

    Args:
        sequence (np.ndarray([seq_len])): One-dimensional array containing sequence of tokens as strings.
        tokenizer (data_loader.DictTokenizer): Tokenizer.
        conds (int or np.ndarray([num_conds])): Correct lengths of each part of sequence
            (a single length if ``split_tokens`` is None).
        split_tokens (list[str]): Split tokens indicating where to split the sequence, each
            expected exactly once. Defaults to None.

    Returns:
        list[np.ndarray] | np.ndarray | None: With ``split_tokens``, a list with one
            integer array per part; otherwise one integer array. None if the sequence is
            invalid: no unique ``<BOS>`` or a part of the wrong length; with ``split_tokens``
            also a missing or repeated split token or ``<EOS>``, an invalid number in a part, or
            any other exception while parsing.
    """
    sequence = np.array(sequence)
    r"""Processes single token sequence.
                                
        Args:
            sequence (np.ndarray([seq_len])): One-dimensional array containing sequence of tokens as strings.
            tokenizer (utils.DictTokenizer): Tokenizer.
            conds (int or np.ndarray([num_conds])): Correct lengths of each part of sequence.
            split_tokens (np.ndarray([num_conds-1])): Split tokens indicating where to split the sequence. Defaults to None.

        Returns:
            data (list): List of different parts of sequence. Returns None if sequence invalid.
        """
    try:
        bos_idx = np.argwhere(sequence == "<BOS>").item()
    except:
        return None
    if split_tokens is not None:
        try:
            split_idx = np.array(
                [
                    np.argwhere(sequence == split_token).item()
                    for split_token in split_tokens
                ]
            )

            split_idx = np.insert(split_idx, 0, bos_idx)
            split_idx = np.append(split_idx, np.argwhere(sequence == "<EOS>").item())
            data = []
            for i in range(len(split_idx) - 1):
                data_iter, valid_mask = token_to_data(
                    sequence[None, split_idx[i] + 1 : split_idx[i + 1]],
                    tokenizer,
                    return_invalid=True,
                )
                if valid_mask.item():
                    data.append(data_iter.flatten())
                else:
                    return None
            lens = [len(data_iter) for data_iter in data]
            if np.all(lens == conds):
                return data
            else:
                return None
        except:
            return None
    else:
        data = token_to_data(sequence[None, bos_idx + 1 :], tokenizer).flatten()
        if len(data) == conds:
            return data
        else:
            return None



@jax.jit
def process_sample_batch(
    tokens: jax.Array,
    token_value: jax.Array,
    is_separator_token: jax.Array,
    PLUS_ID: int,
    MINUS_ID: int,
    EOS_ID: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Parses a batch of token-ID sequences into signed integers with one ``lax.scan``.

    A number is a sign token (``+`` or ``-``) followed by one or more digit tokens; the
    digits are combined in base 1000 (hard-coded). A number is emitted at the position of
    the token that ends it (a sign, a separator or ``<EOS>``), provided it has a sign and
    at least one digit.

    Args:
        tokens (jnp.ndarray([batch_size, seq_len])): Integer token IDs.
        token_value (jnp.ndarray([vocab_size])): Digit value of each token ID, -1 for
            tokens that are not digits.
        is_separator_token (jnp.ndarray([vocab_size])): Boolean mask of the separator token IDs.
        PLUS_ID (int): Token ID of ``+``.
        MINUS_ID (int): Token ID of ``-``.
        EOS_ID (int): Token ID of ``<EOS>``.

    Returns:
        tuple: ``(emitted_values, emit_mask, segment, scan_valid)``, where
            ``emitted_values`` (jnp.ndarray([batch_size, seq_len])) holds the signed integer
            emitted at each position and 0 elsewhere; ``emit_mask``
            (jnp.ndarray([batch_size, seq_len])) marks the positions where a number is emitted;
            ``segment`` (jnp.ndarray([batch_size, seq_len])) counts the separator tokens strictly
            before each position, so a number emitted at a separator belongs to the part that
            the separator closes; and ``scan_valid`` (jnp.ndarray([batch_size])) is False for
            sequences with a digit outside a signed number, or with a sign followed by another
            sign, a separator or ``<EOS>`` before any digit.
    """
    values = token_value[tokens]
    is_number = values >= 0
    is_plus  = tokens == PLUS_ID
    is_minus = tokens == MINUS_ID
    is_sign  = is_plus | is_minus
    is_eos = tokens == EOS_ID
    is_sep = is_separator_token[tokens]
    segment = jnp.cumsum(is_sep, -1) - is_sep.astype(jnp.int32)
    xs = (
        values.T,
        is_plus.T,
        is_minus.T,
        is_number.T,
        is_sign.T,
        is_sep.T,
        is_eos.T,
    )

    def step(carry, x):
        acc, sign, has_number, has_sign, valid = carry

        value, is_plus, is_minus, is_number, is_sign, is_sep, is_eos = x

        boundary = is_sign | is_sep | is_eos

        # -----------------------------------------
        # Emit previous number
        # -----------------------------------------

        emit = boundary & has_number & has_sign

        emitted_value = jnp.where(
            emit,
            sign * acc,
            0,
        )

        # -----------------------------------------
        # Validate
        # -----------------------------------------

        # Numeric chunks are only allowed after a sign.
        valid &= ~(is_number & ~has_sign)

        # A sign shouldn't terminate an empty number.
        #
        # You may or may not want this depending on BOS etc.
        valid &= ~(is_sign & has_sign & ~has_number)

        # Separator/EOS shouldn't terminate an empty signed number.
        valid &= ~((is_sep | is_eos) & has_sign & ~has_number)

        # -----------------------------------------
        # Finish old number
        # -----------------------------------------

        acc = jnp.where(boundary, 0, acc)

        has_number = jnp.where(
            boundary,
            False,
            has_number,
        )

        has_sign = jnp.where(
            boundary,
            False,
            has_sign,
        )

        # -----------------------------------------
        # Start new number
        # -----------------------------------------

        sign = jnp.where(is_plus, 1, sign)
        sign = jnp.where(is_minus, -1, sign)

        has_sign |= is_sign

        # -----------------------------------------
        # Accumulate
        # -----------------------------------------

        acc = jnp.where(
            is_number,
            acc * 1000 + value,
            acc,
        )

        has_number |= is_number

        return (
            acc,
            sign,
            has_number,
            has_sign,
            valid,
        ), (
            emitted_value,
            emit,
        )

    B = values.shape[0]
    initial = (
        jnp.zeros(B, dtype=jnp.int64),  # acc
        jnp.ones(B, dtype=jnp.int64),   # sign
        jnp.zeros(B, dtype=jnp.bool_),  # has_number
        jnp.zeros(B, dtype=jnp.bool_),  # has_sign
        jnp.ones(B, dtype=jnp.bool_),   # structurally valid
    )


    final_carry, (emitted_values, emit_mask) = jax.lax.scan(
        step,
        initial,
        xs,
    )
    _, _, _, _, scan_valid = final_carry

    emitted_values = emitted_values.T
    emit_mask = emit_mask.T


    return emitted_values, emit_mask, segment, scan_valid


def process_samples(
    samples_lst: Sequence[Any],
    tokenizer: DictTokenizer,
    conds: np.ndarray,
    split_tokens: Sequence[str] | np.ndarray | None = None,
    return_valid_idx: bool = False,
) -> (
    list[jax.Array]
    | list[list[jax.Array]]
    | tuple[list[list[jax.Array]], list[jax.Array]]
):
    r"""Processes generated samples for one or multiple prompts.

    With ``split_tokens``, the samples of each prompt are parsed in one batch by
    :func:`process_sample_batch`. A sample is kept if it is structurally valid, contains
    exactly ``len(cond) - 1`` separator tokens and each part contains exactly the expected
    number of integers. Without ``split_tokens``, each sequence is parsed by
    :func:`process_sequence`.

    Args:
        samples_lst (list): List of sampled sequences for each prompt. With ``split_tokens``,
            one array of integer token IDs of shape [num_samples, seq_len] per prompt (as
            returned by ``jax_transformer.utils.sample``); without, an iterable of
            token-string sequences per prompt.
        tokenizer (data_loader.DictTokenizer): Tokenizer.
        conds (np.ndarray([num_prompts, num_conds])): Correct lengths of each part of sequence for each prompt.
        split_tokens (np.ndarray([num_conds-1])): Split tokens indicating where to split the sequence. Defaults to None.
        return_valid_idx (bool): Whether to additionally return indices of valid samples.
            Only used if ``split_tokens`` is given. Defaults to False.

    Returns:
        list | tuple: Processed samples for each prompt. With ``split_tokens``, a list over
            prompts of lists with one array of shape [num_valid, cond[k]] per part; if
            ``return_valid_idx`` is True, the tuple ``(samples_valid, valid_idx_per_prompt)``,
            where ``valid_idx_per_prompt`` holds the row indices of the valid samples of each
            prompt. Without ``split_tokens``, a list over prompts of arrays of shape
            [num_valid, conds[i]].
    """
    if split_tokens is None:
        samples_valid = []
        for i in range(len(samples_lst)):
            fluxes_valid = [
                out[None, :]
                for out in (
                    process_sequence(sequence, tokenizer, conds[i])
                    for sequence in samples_lst[i]
                )
                if out is not None
            ]
            samples_valid.append(jnp.concatenate(fluxes_valid, axis=0))
        return samples_valid
    else:
        samples_valid = []
        valid_idx_per_prompt = []
        token_value = jnp.full(tokenizer.vocab_size, -1, dtype=int)
        token_value = token_value.at[-tokenizer.base:].set(jnp.arange(tokenizer.base))
        SEP_IDS = [tokenizer.dict[sep] for sep in split_tokens]
        is_separator_token = jnp.zeros(tokenizer.vocab_size, dtype=jnp.bool_)
        is_separator_token = is_separator_token.at[
            jnp.array(SEP_IDS)
        ].set(True)

        import sys
        jnp.set_printoptions(threshold=sys.maxsize)

        
        for samples, cond in zip(samples_lst, conds):
            emitted_values, emit_mask, segment, scan_valid = process_sample_batch(samples, token_value, is_separator_token, tokenizer.dict["+"], tokenizer.dict["-"], tokenizer.dict["<EOS>"])

            B, L = emitted_values.shape

            valid = scan_valid.copy()
            parts = []

            for seg_id, expected_len in enumerate(cond):
                expected_len = int(expected_len)

                # Which emitted numbers belong to this segment?
                # shape: [B, L]
                belongs = emit_mask & (segment == seg_id)

                # How many integers did each sample generate for this segment?
                # shape: [B]
                actual_len = jnp.sum(belongs, axis=-1)

                # This sample is valid only if this segment has exactly
                # the expected number of integers.
                valid &= actual_len == expected_len

                # Position of each emitted number within this segment.
                #
                # Example:
                #
                # belongs = [F, F, T, F, T, T, F]
                # pos     = [-1,-1, 0, 0, 1, 2, 2]
                #
                pos = jnp.cumsum(belongs, axis=-1) - 1

                # Allocate fixed-size output for this segment.
                part = jnp.zeros(
                    (B, expected_len),
                    dtype=emitted_values.dtype,
                )

                # Scatter emitted values into their position.
                batch_idx = jnp.arange(B)[:, None]

                safe_pos = jnp.clip(
                    pos,
                    0,
                    expected_len - 1,
                )

                contribution = jnp.where(
                    belongs & (pos < expected_len),
                    emitted_values,
                    0,
                )

                part = part.at[
                    batch_idx,
                    safe_pos,
                ].add(contribution)

                parts.append(part)

             # Require exactly the right number of separators.
            num_separators = jnp.sum(
                is_separator_token[samples],
                axis=-1,
            )

            valid &= num_separators == len(cond) - 1

            # Data-dependent operation: keep this outside @jax.jit
            valid_idx = jnp.where(valid)[0]

            parts = [
                part[valid]
                for part in parts
            ]
            samples_valid.append(parts)
            valid_idx_per_prompt.append(valid_idx)
        if return_valid_idx:
            return samples_valid, valid_idx_per_prompt
        else:
            return samples_valid

