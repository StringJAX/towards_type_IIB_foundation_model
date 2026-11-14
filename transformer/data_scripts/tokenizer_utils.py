"""Conversion between integer data and token sequences.

An integer is written as a sign token ("+" or "-"; zero is "+") followed by its
digits in a given base (default 1000), most significant digit first, each digit
as a decimal string. Rows of integers are concatenated, wrapped in special tokens
such as "<BOS>", "<EOS>", "<PAD>", "<TAB>", "<M_START>" and "<T_START>".
Sequences of sign and digit tokens are decoded back with :func:`token_to_data`.
"""

from typing import Any

import numpy as np
import warnings

def to_base(integer: int | np.integer, base: int) -> np.ndarray:
    r"""Maps integers onto base.

    The number of digits is computed in floating point with ``np.emath.logn``;
    for base 10 it drops the leading digit of 10**3 and 10**6. For base
    1000 no failure was found for 1000**k, k = 1, ..., 6.

    Args:
        integer (int): Integer to be mapped.
        base (int): Base of the digit expansion.

    Returns:
        out (np.ndarray([2+floor(log_base(abs(integer)))])): Sequence of tokens representing integer in base:
            the sign token, then the digits as strings. Zero gives ["+", "0"].

    """
    if integer == 0:
        return np.array(["+", "0"])
    sign_to_token = {0: "+", 1: "+", -1: "-"}
    sign = sign_to_token[np.sign(integer)]
    abs = np.abs(integer)
    out_len = np.floor(np.emath.logn(base, abs)) + 1
    pows = base ** np.flip(np.arange(out_len))
    out = ((abs // pows) % base).astype(int)
    out = np.concatenate([[sign], out.astype(str)])
    return out


def add_padding(tokens: np.ndarray, max_len: int, add_eos: bool = True) -> np.ndarray:
    r"""Adds padding to token sequence.

    Args:
        tokens (np.ndarray([seq_len])): Sequence of tokens.
        max_len (int): Maximum length of token sequences, excluding "<EOS>";
            max_len - seq_len "<PAD>" tokens are appended.
        add_eos (bool): Whether to insert "<EOS>" between tokens and padding.
            Defaults to True.

    Returns:
        out (np.ndarray([max_len + 1]) or np.ndarray([max_len])): Token sequence with added padding
            tokens on the right; length max_len + 1 with "<EOS>", max_len without.
    """
    padding = np.full(max_len - len(tokens), "<PAD>")
    if add_eos:
        out = np.concatenate([tokens, np.array(["<EOS>"]), padding])
    else:
        out = np.concatenate([tokens, padding])
    return out


def int_to_token(
    data: np.ndarray, base: int = 1000, do_padding: bool = True,
    max_len_padding: int | None = None, add_eos: bool = True,
) -> tuple[np.ndarray, int | np.integer | None]:
    r"""Maps rows of integers to token sequences starting with "<BOS>".

    Args:
        data (np.ndarray([num_rows, num_ints])): Integers; each row becomes one sequence.
        base (int): Base of the digit expansion. Defaults to 1000.
        do_padding (bool): Whether to pad each row with :func:`add_padding`.
            Without padding, all rows must give the same number of tokens.
            Defaults to True.
        max_len_padding (int or None): Padded length passed to :func:`add_padding`.
            None uses the longest row. Defaults to None.
        add_eos (bool): Whether padding inserts "<EOS>". Defaults to True.

    Returns:
        tokens (np.ndarray([num_rows, seq_len])): "<BOS>" followed by the tokens of each row
            (and, with padding, "<EOS>" if add_eos and "<PAD>" tokens).
        max_len_padding (int or None): The padded length used; with do_padding=False,
            the input value is returned unchanged.
    """
    tokens = [
        np.concatenate([to_base(integer, base) for integer in row]) for row in data
    ]
    if do_padding:
        if max_len_padding is None:
            max_len_padding = np.max(np.array([len(seq) for seq in tokens]))
        tokens = np.array([add_padding(seq, max_len_padding, add_eos) for seq in tokens])
    bos = np.repeat("<BOS>", len(tokens))[:, None]
    tokens = np.concatenate([bos, tokens], axis=-1)
    return tokens, max_len_padding


def data_to_token(
    data: np.ndarray, split_idx: int, max_len_padding: int | None = None
) -> tuple[np.ndarray, int | np.integer]:
    r"""Maps rows of prompt and flux integers to encoder-decoder token sequences.

    Each row becomes the prompt tokens (no "<BOS>", no padding), "<TAB>", then
    "<BOS>", the flux tokens, "<EOS>" and "<PAD>" tokens. Uses base 1000.

    Args:
        data (np.ndarray([num_rows, num_ints])): Integers; columns before split_idx
            form the prompt, the remaining columns the fluxes.
        split_idx (int): Index of the first flux column.
        max_len_padding (int or None): Padded length of the flux tokens. None uses
            the longest row. Defaults to None.

    Returns:
        data (np.ndarray([num_rows, seq_len])): Token sequences.
        max_len_padding (int): The padded length used for the flux tokens.
    """
    prompts_tok, _ = int_to_token(data[:, :split_idx], do_padding=False)
    prompts_tok = prompts_tok[:,1:]
    fluxes_tok, max_len_padding = int_to_token(data[:, split_idx:], max_len_padding=max_len_padding)
    tab_tok = np.repeat(np.array(["<TAB>"])[None, :], len(data), axis=0)
    data = np.concatenate([prompts_tok, tab_tok, fluxes_tok], axis=-1)
    return data, max_len_padding

def validate_token_seq(tokens: np.ndarray) -> bool:
    r"""Validates single token sequence.

    A sequence is valid if it starts with a sign token ("+" or "-"), ends with a
    digit, contains only sign and digit tokens, and every sign is followed by a digit.

    Args:
        tokens (np.ndarray([seq_len])): Tokens of one sequence, without special tokens.

    Returns:
        bool: True if valid sequence, False if invalid.
    """
    if tokens[0] not in ["-", "+"]:
        return False
    if not tokens[-1].isdigit():
        return False
    for i in range(1, len(tokens)):
        if tokens[i - 1] not in ["-", "+"] and not tokens[i-1].isdigit():
            return False
        if tokens[i - 1] in ["-", "+"] and not tokens[i].isdigit():
            return False
    return True


def token_seq_to_data(tokens: np.ndarray, base: int) -> np.ndarray:
    r"""Maps single token sequence to respective integer sequence.

    Args:
        tokens (np.ndarray([seq_len])): Valid tokens of one sequence (see :func:`validate_token_seq`).
        base (int): Base of the digit expansion.

    Returns:
        out (np.ndarray([num_ints])): Integer sequence, one entry per sign token.

    """
    int_lst = []
    integer = []
    str_to_sgn = {"+": +1, "-": -1}
    for i in range(len(tokens)):
        if tokens[i] in ["-", "+"]:
            sign = str_to_sgn[tokens[i]]
        else:
            integer.append(int(tokens[i]))
        if i == len(tokens) - 1:
            integer = np.array(integer)
            pow = base ** np.flip(np.arange(len(integer)))
            int_lst.append(np.sum(sign * integer * pow))
            integer = []
        elif not tokens[i + 1].isdigit():
            integer = np.array(integer)
            pow = base ** np.flip(np.arange(len(integer)))
            int_lst.append(np.sum(sign * integer * pow))
            integer = []
    out = np.hstack(int_lst)
    return out


def token_to_data(
    tokens: np.ndarray, tokenizer: Any, return_invalid: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    r"""Maps tokens to integer sequences.

    Warns if any sequence is invalid (see :func:`validate_token_seq`). Without
    return_invalid, invalid sequences are dropped; with it, they are kept as rows
    of None. All rows that are stacked must have equal length.

    Args:
        tokens (np.ndarray([data_size, max_len])): Tokens, without special tokens.
        tokenizer (Any): Tokenizer providing the integer base as ``tokenizer.base``
            (e.g. :class:`jax_transformer.data_loader.DictTokenizer`).
        return_invalid (bool): Whether to keep invalid rows and also return the
            validity mask. Defaults to False.

    Returns:
        data (np.ndarray([num_rows, num_ints])): Integer sequences; num_rows is the number of
            valid sequences, or data_size with return_invalid.
        valid_mask (np.ndarray([data_size])): Only with return_invalid. True for valid sequences.
    """
    base = tokenizer.base
    valid_mask = np.array([validate_token_seq(seq) for seq in tokens])
    if np.any(~valid_mask):
        warnings.warn(
            f"Invalid sequence(s) detected in IntegerTokenizer.token_ids_to_data: Dropping {(~valid_mask).sum()} sequence(s)."
        )
    if return_invalid:
        tokens_valid = np.where(valid_mask[:,None], tokens, None)
        data = []

        data = np.vstack([ seq if seq[0] == None else token_seq_to_data(seq, base) for seq in tokens_valid])
        return data, valid_mask
    else:
        tokens_valid = tokens[valid_mask]
        data = np.vstack([token_seq_to_data(seq, base) for seq in tokens_valid])
        return data


def int_nums_to_token(data: np.ndarray, base: int = 1000) -> np.ndarray:
    r"""Maps rows of integers to one flat token sequence, each row prefixed by "<INT_START>".

    Args:
        data (np.ndarray([num_rows, num_ints])): Integers; all rows must give the same number of tokens.
        base (int): Base of the digit expansion. Defaults to 1000.

    Returns:
        np.ndarray([num_rows * (1 + num_tokens_per_row)]): Flattened token sequence.
    """
    tokens = [
        np.concatenate([to_base(integer, base) for integer in row]) for row in data
    ]
    bos = np.repeat("<INT_START>", len(tokens))[:, None]
    tokens = np.concatenate([bos, tokens], axis=-1)
    return tokens.flatten()

def a_matrix_to_token(data: np.ndarray, base: int = 1000) -> np.ndarray:
    r"""Maps A-matrix entries to a token sequence starting with "<A_START>".

    Args:
        data (np.ndarray([num_ints])): One-dimensional array of integer entries.
        base (int): Base of the digit expansion. Defaults to 1000.

    Returns:
        np.ndarray([seq_len]): "<A_START>" followed by the tokens of each entry.
    """
    tokens =  np.concatenate([to_base(integer, base) for integer in data])
    bos = np.array(["<A_START>"])
    tokens = np.concatenate([bos, tokens], axis=0)
    return tokens

def b_vector_to_token(data: np.ndarray, base: int = 1000) -> np.ndarray:
    r"""Maps b-vector entries to a token sequence starting with "<B_START>".

    Args:
        data (np.ndarray([num_ints])): One-dimensional array of integer entries.
        base (int): Base of the digit expansion. Defaults to 1000.

    Returns:
        np.ndarray([seq_len]): "<B_START>" followed by the tokens of each entry.
    """
    tokens =  np.concatenate([to_base(integer, base) for integer in data])
    bos = np.array(["<B_START>"])
    tokens = np.concatenate([bos, tokens], axis=0)
    return tokens

def hodge_numbers_to_token(data: np.ndarray, base: int = 1000) -> np.ndarray:
    r"""Maps Hodge numbers to a token sequence starting with "<H_START>".

    Args:
        data (np.ndarray([num_ints])): One-dimensional array of Hodge numbers.
        base (int): Base of the digit expansion. Defaults to 1000.

    Returns:
        np.ndarray([seq_len]): "<H_START>" followed by the tokens of each number.
    """
    tokens =  np.concatenate([to_base(integer, base) for integer in data])
    bos = np.array(["<H_START>"])
    tokens = np.concatenate([bos, tokens], axis=0)
    return tokens

def chi_to_token(data: int | np.integer, base: int = 1000) -> np.ndarray:
    r"""Maps the Euler characteristic to a token sequence starting with "<C_START>".

    Args:
        data (int): Euler characteristic.
        base (int): Base of the digit expansion. Defaults to 1000.

    Returns:
        np.ndarray([seq_len]): "<C_START>" followed by the tokens of data.
    """
    tokens = to_base(data, base)
    bos = np.array(["<C_START>"])
    tokens = np.concatenate([bos, tokens], axis=0)
    return tokens

def top_data_to_token(
    data: np.ndarray, prompt_dic: dict[str, Any], max_len_padding: int | None = None
) -> tuple[np.ndarray, int | np.integer]:
    r"""Maps rows of a geometry key and fluxes to encoder-decoder token sequences.

    Each row becomes the prompt tokens from prompt_dic, "<TAB>", then "<BOS>", the
    flux tokens (base 1000), "<EOS>" and "<PAD>" tokens.

    Args:
        data (np.ndarray([num_rows, 2 + num_fluxes])): The first two columns form the
            key into prompt_dic; the remaining columns are the fluxes.
        prompt_dic (dict[str, Any]): Maps ``str([a, b])`` of the first two columns
            (as Python ints, e.g. "[2, 272]") to a prompt token sequence; all
            prompt sequences must have equal length.
        max_len_padding (int or None): Padded length of the flux tokens. None uses
            the longest row. Defaults to None.

    Returns:
        tok (np.ndarray([num_rows, seq_len])): Token sequences.
        max_len_padding (int): The padded length used for the flux tokens.
    """
    fluxes = data[:,2:]
    prompts = data[:,:2]
    prompts = np.array([prompt_dic[str(pair.astype(int).tolist())] for pair in prompts])
    fluxes_tok, max_len_padding = int_to_token(fluxes, max_len_padding=max_len_padding)
    tab_tok = np.repeat(np.array(["<TAB>"])[None, :], len(fluxes_tok), axis=0)
    tok = np.concatenate([prompts, tab_tok, fluxes_tok], axis=-1)
    return tok, max_len_padding

def int_and_mod_to_token(
    fluxes: np.ndarray, moduli: np.ndarray, tau: np.ndarray, base: int = 1000,
    max_len_padding: int | None = None,
) -> tuple[np.ndarray, int | np.integer]:
    r"""Maps fluxes, moduli and axio-dilaton to decoder token sequences.

    Each row becomes "<BOS>", the flux tokens, "<M_START>", the moduli tokens,
    "<T_START>", the tau tokens, "<EOS>" and "<PAD>" tokens. The inverse
    conversion in evaluation_scripts.evaluation_utils.convert_samples_to_data
    reads moduli as real parts followed by imaginary parts, and tau as (real,
    imaginary) unless gs_inv is given, both divided by a discretisation factor.

    Args:
        fluxes (np.ndarray([num_rows, num_fluxes])): Integer fluxes.
        moduli (np.ndarray([num_rows, num_moduli])): Integer-valued moduli data.
        tau (np.ndarray([num_rows, num_tau])): Integer-valued axio-dilaton data.
        base (int): Base of the digit expansion. Defaults to 1000.
        max_len_padding (int or None): Padded length passed to :func:`add_padding`.
            None uses the longest row. Defaults to None.

    Returns:
        tokens (np.ndarray([num_rows, max_len_padding + 2])): Token sequences.
        max_len_padding (int): The padded length used.
    """
    tokens = []

    for i in range(len(fluxes)):
        flux_tok = np.concatenate([to_base(integer, base) for integer in fluxes[i]])
        mod_tok = np.concatenate([to_base(integer, base) for integer in moduli[i]])
        tau_tok = np.concatenate([to_base(integer, base) for integer in tau[i]])
        tok = np.concatenate([flux_tok, np.array(["<M_START>"]), mod_tok, np.array(["<T_START>"]), tau_tok])
        tokens.append(tok)
    if max_len_padding is None:
        max_len_padding = np.max(np.array([len(seq) for seq in tokens]))
    tokens = np.array([add_padding(seq, max_len_padding) for seq in tokens])
    bos = np.repeat("<BOS>", len(tokens))[:, None]
    tokens = np.concatenate([bos, tokens], axis=-1)
    return tokens, max_len_padding


def mod_data_to_token(
    prompts: np.ndarray, fluxes: np.ndarray, moduli: np.ndarray, tau: np.ndarray,
    base: int = 1000, max_len_padding_in: int | None = None,
    max_len_padding_out: int | None = None,
) -> tuple[np.ndarray, int | np.integer, int | np.integer]:
    r"""Maps integer prompts and fluxes, moduli and tau to encoder-decoder token sequences.

    Each row becomes the prompt tokens padded with "<PAD>" (no "<BOS>" or
    "<EOS>"), "<TAB>", then the output of :func:`int_and_mod_to_token`.

    Args:
        prompts (np.ndarray([num_rows, num_prompt_ints])): Integer prompts.
        fluxes (np.ndarray([num_rows, num_fluxes])): Integer fluxes.
        moduli (np.ndarray([num_rows, num_moduli])): Integer-valued moduli data.
        tau (np.ndarray([num_rows, num_tau])): Integer-valued axio-dilaton data.
        base (int): Base of the digit expansion. Defaults to 1000.
        max_len_padding_in (int or None): Padded length of the prompt tokens. None
            uses the longest prompt. Defaults to None.
        max_len_padding_out (int or None): Padded length of the output tokens. None
            uses the longest output. Defaults to None.

    Returns:
        data (np.ndarray([num_rows, seq_len])): Token sequences.
        max_len_padding_in (int): The padded prompt length used.
        max_len_padding_out (int): The padded output length used.
    """
    prompts_tok, max_len_padding_in = int_to_token(prompts, base=base, max_len_padding=max_len_padding_in, add_eos=False)
    prompts_tok = prompts_tok[:,1:]
    out_tok, max_len_padding_out = int_and_mod_to_token(fluxes, moduli, tau, base, max_len_padding_out)
    tab_tok = np.repeat(np.array(["<TAB>"])[None, :], len(prompts), axis=0)
    data = np.concatenate([prompts_tok, tab_tok, out_tok], axis=-1)
    return data, max_len_padding_in, max_len_padding_out


def topo_and_mod_data_to_token(
    prompt: np.ndarray, fluxes: np.ndarray, moduli: np.ndarray, tau: np.ndarray,
    base: int = 1000, max_len_padding_in: int | None = None,
    max_len_padding_out: int | None = None,
) -> tuple[np.ndarray, int | None, int | np.integer]:
    r"""Maps one prompt token sequence and fluxes, moduli and tau to encoder-decoder token sequences.

    Each row becomes the shared prompt tokens, "<TAB>", then the output of
    :func:`int_and_mod_to_token`.

    Args:
        prompt (np.ndarray([prompt_len])): Prompt tokens, shared by all rows.
        fluxes (np.ndarray([num_rows, num_fluxes])): Integer fluxes.
        moduli (np.ndarray([num_rows, num_moduli])): Integer-valued moduli data.
        tau (np.ndarray([num_rows, num_tau])): Integer-valued axio-dilaton data.
        base (int): Base of the digit expansion. Defaults to 1000.
        max_len_padding_in (int or None): If given, the prompt is padded with
            "<PAD>" (no "<EOS>") to this length; None leaves it unpadded.
            Defaults to None.
        max_len_padding_out (int or None): Padded length of the output tokens. None
            uses the longest output. Defaults to None.

    Returns:
        data (np.ndarray([num_rows, seq_len])): Token sequences.
        max_len_padding_in (int or None): The input value, unchanged.
        max_len_padding_out (int): The padded output length used.
    """
    if max_len_padding_in is not None:
        prompt = add_padding(prompt, max_len_padding_in, False)
    prompts_tok = np.repeat(prompt[None,:], len(fluxes), axis=0)
    out_tok, max_len_padding_out = int_and_mod_to_token(fluxes, moduli, tau, base, max_len_padding_out)
    tab_tok = np.repeat(np.array(["<TAB>"])[None, :], len(prompts_tok), axis=0)
    data = np.concatenate([prompts_tok, tab_tok, out_tok], axis=-1)
    return data, max_len_padding_in, max_len_padding_out