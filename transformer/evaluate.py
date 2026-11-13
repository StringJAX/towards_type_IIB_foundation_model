"""Sample and validate transformer vacua from Python or the command line."""

import argparse
import logging
import os
from pathlib import Path
import pickle
import time

import logging
logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm
import jaxvacua as jvc  # type: ignore

if __package__:
    from .jax_transformer.utils import sample
    from .evaluation_scripts.evaluation_utils import (
        load_config, load_model_from_config, process_samples,
        convert_samples_to_data, validate_vacua, to_old_a_matrix,
    )
    from .evaluation_scripts.output_utils import print_console_output
    from .data_scripts.tokenizer_utils import int_to_token
else:
    from jax_transformer.utils import sample
    from evaluation_scripts.evaluation_utils import (
        load_config, load_model_from_config, process_samples,
        convert_samples_to_data, validate_vacua, to_old_a_matrix,
    )
    from evaluation_scripts.output_utils import print_console_output
    from data_scripts.tokenizer_utils import int_to_token


def evaluate(
    RUN_ID, load_ckpt, *, single=False, nflux=None, w0=None, gs_inv=None,
    num_samples=1000, tol=1e-10, sampling_temp=0.6, discr=10.0,
    only_sample=False, samples_file=None, only_evaluate=False,
    verbose=False, use_newton=False, cwd=None,
):
    """Sample a checkpoint and optionally validate the generated vacua.

    Arguments match the command-line options. ``cwd`` is the directory holding
    logs, checkpoints and data; it defaults to this file's directory.

    With ``only_sample=True``, return the processed samples in geometry-prompt
    order (each entry is [fluxes, discretized moduli, discretized tau], or None).
    Optionally save them as data/samples/<samples_file>.p. ``only_evaluate``
    loads that file instead of sampling.

    Otherwise return a list of dictionaries, one per prompt/validation geometry,
    with prompt, geometry, fluxes, moduli, tau, and sample_indices. Arrays contain
    the unique validated solutions, including solver-refined complex moduli and
    tau; arrays are None when no solutions survive. sample_indices refer to the
    processed samples for that prompt. A sample can validate on multiple geometries.
    """
    cwd = str(Path(cwd) if cwd is not None else Path(__file__).resolve().parent)
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
    assert (
        RUN_ID is not None and load_ckpt is not None
    ), "--RUN_ID and --load_ckpt always have to be set."
    assert (
        np.sum([nflux is not None, w0 is not None, gs_inv is not None]) == 1 or np.sum([nflux is not None, w0 is not None, gs_inv is not None]) == 0
    ), "Invalid option set. Needs to be --nflux --gs_inv OR --w0."

    if only_evaluate:
        assert (
            samples_file is not None
        ), "--only_evaluate is set but --samples_file is None."
    
    if only_sample and only_evaluate:
        raise ValueError("only_sample and only_evaluate cannot both be True.")

    is_mod = np.sum([nflux is not None, w0 is not None, gs_inv is not None]) == 0
    is_nflux = nflux is not None
    is_w0 = w0 is not None
    is_gs_inv = gs_inv is not None

    if is_nflux:
        assert isinstance(nflux, int), "nflux needs to be an integer."

    ###################
    #   DEFINITIONS   #
    ###################
    geometry_log_path = cwd + "/data/data_multiple/geometry_log_degree_10.p"

    use_scipy = not use_newton


    # Ensure compatibility with older checkpoints
    if "GRPO" in load_ckpt:
        override_opt = True
        GRPO_ID = load_ckpt.split("/")[0].split("_")[-1]
    else:
        override_opt = False
        GRPO_ID = None
    config = load_config(RUN_ID, cwd)
    state, model, tokenizer = load_model_from_config(
        RUN_ID, config, load_ckpt, cwd, override_opt=override_opt, GRPO_ID=GRPO_ID
    )
    max_len = np.min([80, model.max_len_out])

    # Set prompts
    all_hodge_pairs = np.array(
        [
            [[2, 120], [2, 128], [2, 132], [2, 144], [2, 272]],
            [[3, 99], [3, 107], [3, 115], [3, 119], [3, 243]],
            [[4, 98], [4, 118], [4, 122], [4, 130], [4, 142]],
            [[5, 89], [5, 149], [5, 153], [5, 165], [5, 213]],
        ]
    )
    if RUN_ID == 45 or RUN_ID == 44:
        hodge_pairs = all_hodge_pairs[3,:]
    else:
        hodge_pairs = all_hodge_pairs

    prompts = all_hodge_pairs.reshape(-1, 2)

    # Load data
    train_data_file = "_".join(
        config["train_data_flux"].split("/")[-1].split(".")[0].split("_")[2:]
    )
    h12_lst = all_hodge_pairs[:, 0, 0]
    raw_data_paths = {
        h12: cwd + f"/data/data_multiple/train_data_raw_{train_data_file}_{h12}.npy"
        for h12 in h12_lst
    }
    with open(geometry_log_path, "rb") as f:
        geometry_log = pickle.load(f)

    if is_w0:
        prompt_with_w0 = []
        aux_prompts = np.array([w0])
        for prompt in prompts:
            prompt = np.repeat(prompt[None, :], len(aux_prompts), axis=0)
            prompt = np.concatenate(
                [prompt, np.round(discr * aux_prompts)[:, None].astype(np.int16)],
                axis=-1,
            )
            prompt_with_w0.append(prompt)
        prompts = np.concatenate(prompt_with_w0, axis=0)

    if is_nflux:
        prompt_with_nflux = []
        aux_prompts_len = []
        for prompt in prompts:
            geometry_key = f"{prompt[0]} {prompt[1]}"
            h12 = prompt[0]
            model_ID = geometry_log[geometry_key]["model_ID"]
            jvc_model = jvc.FluxVacuaFinder(h12=h12, model_ID=model_ID, maximum_degree=10)
            aux_prompts = np.array([nflux])
            aux_prompts_len.append(len(aux_prompts))
            prompt = np.repeat(prompt[None, :], len(aux_prompts), axis=0)
            prompt = np.concatenate([prompt, aux_prompts[:, None]], axis=-1)
            prompt_with_nflux.append(prompt)
        prompts = np.concatenate(prompt_with_nflux, axis=0)

    if is_gs_inv:
        prompt_with_gs_inv = []
        aux_prompts = np.array([gs_inv])
        for prompt in prompts:
            prompt = np.repeat(prompt[None, :], len(aux_prompts), axis=0)
            prompt = np.concatenate(
                [prompt, np.round(discr * aux_prompts)[:, None].astype(np.int16)],
                axis=-1,
            )
            prompt_with_gs_inv.append(prompt)
        prompts = np.concatenate(prompt_with_gs_inv, axis=0)


    prompts_tok = []
    for prompt in prompts:
        prompt_tok, _ = int_to_token(
            np.atleast_2d(prompt), config["base"], do_padding=False
        )
        prompts_tok.append(prompt_tok[:, 1:].flatten())

    # Define conditions

    if is_w0 or is_gs_inv:
        mult = len(aux_prompts)
    elif is_nflux:
        mult = aux_prompts_len
    else:
        mult = 1
    num_per_row = all_hodge_pairs.shape[-2]
    conds = 4 * (np.repeat(np.array(h12_lst), num_per_row) + 1)
    conds = np.repeat(conds, mult)[:, None]
    if is_gs_inv:
        t_conds = 1 * np.ones(mult * len(h12_lst) * num_per_row)[:, None]
    elif is_nflux:
        t_conds = 2 * np.ones(np.sum(mult))[:, None]
    else:
        t_conds = 2 * np.ones(mult * len(h12_lst) * num_per_row)[:, None]
    z_conds = np.repeat(2 * np.array(h12_lst), num_per_row)
    z_conds = np.repeat(z_conds, mult)[:, None]
    conds = np.concatenate([conds, z_conds, t_conds], axis=-1)
    split_tokens = ["<M_START>", "<T_START>"]

    ###################
    #     Sampling    #
    ###################

    if only_evaluate:
        assert not only_sample, "Both only_evaluate and only_sample were set to True."
        if os.path.isfile(samples_file):
            with open(samples_file, "rb") as handle:
                samples = pickle.load(handle)
        else:
            raise FileNotFoundError(
                f"only_evaluate was set to True but couldn't find samples at {samples_file}"
            )
    else:
        # Maximum length of samples
        samples_lst = []
        print(f"Sampling...")
        time0 = time.time()
        for i, prompt in enumerate(tqdm(prompts_tok)):
            samples = sample(
                state=state,
                model_type=config["model"],
                num_samples=num_samples,
                max_len=max_len,
                tokenizer=tokenizer,
                prompt=prompt,
                sampling_temp=sampling_temp,
            )
            samples_lst.append(samples)
        # WARNING: This does not return unique values

        print(f"Processing samples...")
        jax.config.update("jax_enable_x64", True)
        samples = process_samples(samples_lst, tokenizer, conds, split_tokens)
        time1 = time.time()
        print(
            f"Sampling time in seconds (including processing): {np.round(time1-time0, 1)}"
        )
    if only_sample:
        return samples

    # DEBUG: CLEAR CACHES
    jax.clear_caches()
    ###################
    #    Evaluation   #
    ###################

    valid_samples = []
    mean_validity = []
    print(f"Validating...")
    time0 = time.time()
    # Needed since N_flux prompts have different sizes
    if is_nflux:
        start_idx = 0
    for k in range(len(all_hodge_pairs)):
        h12 = all_hodge_pairs[k][0][0]
        # Load original training data
        fluxes_train = jnp.load(
            raw_data_paths[h12],
            mmap_mode="r",
        )
        training_data = set(map(tuple, np.array(fluxes_train)))
        for i in range(len(all_hodge_pairs[k])):

            # Print header for geometry
            print("##################################")
            print(f"Prompted geometry: {all_hodge_pairs[k][i]}")

            # Check all geometries with the same h12 for each prompt
            if is_mod:
                num_valid_total = 0
                fluxes_prev = set()
                prompt_samples = samples[k * 5 + i]
                print(f"Num samples: {len(prompt_samples[0]) if prompt_samples is not None else 0}")
                log_dict = {}

            for j in range(len(all_hodge_pairs[k])):
                if (single or is_nflux or is_w0 or is_gs_inv) and not np.all(
                    all_hodge_pairs[k][j] == all_hodge_pairs[k][i]
                ):
                    continue
                # Prepare dict for console output
                out = {}
                geometry_key = f"{all_hodge_pairs[k][j][0]} {all_hodge_pairs[k][j][1]}"
                out["geometry"] = geometry_key
                # Prepare model
                model_ID = geometry_log[geometry_key]["model_ID"]
                model = jvc.FluxVacuaFinder(h12=h12, model_ID=model_ID, maximum_degree=10)
                # Needed for compatibility with older checkpoints
                if RUN_ID <= 35:
                    model = to_old_a_matrix(model)

                if is_w0 or is_gs_inv:
                    len_aux_iter = len(aux_prompts)
                elif is_nflux:
                    len_aux_iter = len(aux_prompts)
                else:
                    len_aux_iter = 1
                for l in range(len_aux_iter):
                    if is_nflux and (l + 1) % 100 == 0:
                        jax.clear_caches()
                    if is_mod:
                        idx = k * 5 + i
                        samples_iter = samples[idx]
                    elif is_nflux:
                        idx = start_idx + l
                        samples_iter = samples[idx]
                    else:
                        if (
                            gs_inv
                            and (l + 1) % 10 == 0
                            and all_hodge_pairs[k][j][0] == 5
                        ):
                            jax.clear_caches()
                        idx = (k * 5 + i) * len(aux_prompts) + l
                        samples_iter = samples[idx]
                    if is_gs_inv:
                        gs_inv_prompt = int(prompts[idx, -1])
                    else:
                        gs_inv_prompt = None
                    if samples_iter is not None:
                        fluxes, moduli, tau = convert_samples_to_data(
                            samples_iter, discr, gs_inv_prompt
                        )
                    else:
                        fluxes = None

                    if fluxes is not None:
                        idx_samples = jnp.arange(len(fluxes))
                        fluxes, moduli, tau, idx_samples, out = validate_vacua(
                            fluxes,
                            moduli,
                            tau,
                            idx_samples,
                            model,
                            out,
                            tol,
                            use_scipy,
                            False,
                            False,
                            False,
                        )

                    else:
                        out["valid"] = 0
                        out["valid_unique"] = 0
                    mean_validity.append(out["valid_unique"])
                    if fluxes is not None:
                        # Check if in training data
                        fluxes_np = np.array(fluxes)
                        num_in_train_data = sum(
                            tuple(row) in training_data for row in fluxes_np
                        )
                        out["num_in_train_data"] = num_in_train_data

                        # Update genuinely new solutions
                        if is_mod and not single:
                            num_new = 0
                            for idx_sample in np.array(idx_samples):
                                if idx_sample not in fluxes_prev:
                                    fluxes_prev.add(idx_sample)
                                    num_new += 1
                            num_valid_total += num_new

                    valid_samples.append({
                        "prompt": tuple(int(value) for value in prompts[idx]),
                        "geometry": tuple(int(value) for value in all_hodge_pairs[k][j]),
                        "fluxes": fluxes,
                        "moduli": moduli if fluxes is not None else None,
                        "tau": tau if fluxes is not None else None,
                        "sample_indices": idx_samples if fluxes is not None else None,
                    })

                    # Print console output
                    if is_nflux or is_w0 or is_gs_inv:
                        aux = prompts[idx][-1]
                    else:
                        aux = None
                    print_console_output(out, aux, verbose)
                if is_nflux:
                    start_idx = idx + 1

                if is_nflux or is_w0 or is_gs_inv:
                    jax.clear_caches()
            if is_mod and not single:
                print("------------------------")
                print(f"Total valid: {num_valid_total}")
                jax.block_until_ready(fluxes_prev)
                del fluxes_prev
            jax.clear_caches()
        jax.block_until_ready(fluxes_train)
        del fluxes_train
    time1 = time.time()
    print(f"Evaluation time in seconds: {np.round(time1-time0, 1)}")

    return valid_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--RUN_ID", type=int)
    parser.add_argument("--load_ckpt", type=str)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--nflux", type=int)
    parser.add_argument("--w0", type=float)
    parser.add_argument("--gs_inv", type=float)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--sampling_temp", type=float, default=0.6)
    parser.add_argument("--discr", type=float, default=10.0)
    parser.add_argument("--only_sample", action="store_true")
    parser.add_argument("--samples_file", type=str)
    parser.add_argument("--only_evaluate", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use_newton", action="store_true")
    return evaluate(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
