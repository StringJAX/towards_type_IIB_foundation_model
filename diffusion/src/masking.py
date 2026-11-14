"""Mask-probability schedules shared by training and sampling."""

from __future__ import annotations

import math

import torch


MASK_SCHEDULES: tuple[str, ...] = ("sine", "linear")
TIME_CONDITIONING_MODES: tuple[str, ...] = ("discrete", "continuous", "none")


def canonical_mask_schedule(value: object = "sine") -> str:
    """Return a validated mask-schedule name.

    ``sine`` preserves the original sine-squared schedule. ``linear`` uses
    ``m_t = (t + 1) / T``, so uniformly sampled discrete timesteps approximate
    a uniform mask probability on ``(0, 1]``.

    Args:
        value (object): Schedule name. It is converted to a string, stripped,
            lower-cased and has hyphens replaced by underscores; the aliases
            ``sine_squared`` and ``sin2`` map to ``sine`` and ``uniform`` maps
            to ``linear``. Defaults to ``"sine"``.

    Returns:
        str: ``"sine"`` or ``"linear"``.

    Raises:
        ValueError: If ``value`` names no supported schedule.
    """

    schedule = str(value).strip().lower().replace("-", "_")
    aliases = {
        "sine": "sine",
        "sine_squared": "sine",
        "sin2": "sine",
        "linear": "linear",
        "uniform": "linear",
    }
    try:
        return aliases[schedule]
    except KeyError as error:
        raise ValueError(
            "mask_schedule must be one of: sine, linear"
        ) from error


def canonical_time_conditioning(value: object = "discrete") -> str:
    """Return a validated denoiser time-conditioning mode.

    Args:
        value (object): Mode name, normalised as in
            :func:`canonical_mask_schedule`. Aliases: ``timestep`` for
            ``discrete``; ``mask_probability`` and ``rho`` for ``continuous``;
            ``no_time`` and ``disabled`` for ``none``. Defaults to
            ``"discrete"``.

    Returns:
        str: ``"discrete"``, ``"continuous"`` or ``"none"``.

    Raises:
        ValueError: If ``value`` names no supported mode.
    """

    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "discrete": "discrete",
        "timestep": "discrete",
        "continuous": "continuous",
        "mask_probability": "continuous",
        "rho": "continuous",
        "none": "none",
        "no_time": "none",
        "disabled": "none",
    }
    try:
        return aliases[mode]
    except KeyError as error:
        raise ValueError(
            "time_conditioning must be one of: discrete, continuous, none"
        ) from error


def timestep_from_mask_fraction(
    masked_count: torch.Tensor,
    active_count: torch.Tensor | int,
    num_timesteps: int,
    mask_schedule: str = "sine",
) -> torch.Tensor:
    """Invert a mask schedule at the current fraction of masked coordinates.

    With ``rho = clamp(masked_count / max(active_count, 1), 0, 1)`` and
    ``T = num_timesteps``, the timestep is ``round(T * rho - 1)`` for the
    linear schedule and ``round(T * (2 / pi) * arcsin(sqrt(rho)) - 1)`` for
    the sine schedule, clamped to ``[0, T - 1]``.

    Args:
        masked_count (torch.Tensor): Number of masked active coordinates per row.
        active_count (torch.Tensor | int): Number of active coordinates per row.
        num_timesteps (int): Number of discrete timesteps ``T``.
        mask_schedule (str): Schedule name accepted by
            :func:`canonical_mask_schedule`. Defaults to ``"sine"``.

    Returns:
        torch.Tensor: Integer (``long``) timesteps with the broadcast shape of
            ``masked_count`` and ``active_count``.

    Raises:
        ValueError: If ``num_timesteps`` is not positive or the schedule is
            unknown.
    """

    if int(num_timesteps) <= 0:
        raise ValueError("num_timesteps must be positive")
    schedule = canonical_mask_schedule(mask_schedule)
    masked = torch.as_tensor(masked_count).float()
    active = torch.as_tensor(
        active_count, device=masked.device
    ).float().clamp_min(1.0)
    rho = (masked / active).clamp(0.0, 1.0)
    if schedule == "linear":
        continuous = int(num_timesteps) * rho - 1.0
    else:
        continuous = (
            int(num_timesteps)
            * (2.0 / math.pi)
            * torch.asin(torch.sqrt(rho))
            - 1.0
        )
    return continuous.round().clamp(0, int(num_timesteps) - 1).long()


def time_condition_from_mask_fraction(
    masked_count: torch.Tensor,
    active_count: torch.Tensor | int,
    *,
    time_conditioning: str = "discrete",
    num_timesteps: int | None = 200,
    mask_schedule: str = "sine",
) -> torch.Tensor | None:
    """Map the current masked fraction to the denoiser's time input.

    Args:
        masked_count (torch.Tensor): Number of masked active coordinates per row.
        active_count (torch.Tensor | int): Number of active coordinates per row;
            values below 1 are treated as 1.
        time_conditioning (str): Mode accepted by
            :func:`canonical_time_conditioning`. Defaults to ``"discrete"``.
        num_timesteps (int | None): Number of discrete timesteps; required in
            discrete mode. Defaults to 200.
        mask_schedule (str): Schedule used in discrete mode. Defaults to
            ``"sine"``.

    Returns:
        torch.Tensor | None: ``None`` in mode ``none``; the masked fraction
            clamped to ``[0, 1]`` (float) in mode ``continuous``; the timestep from
            :func:`timestep_from_mask_fraction` (``long``) in mode ``discrete``.

    Raises:
        ValueError: If the mode is unknown, or if discrete mode is requested
            with an unknown schedule or with ``num_timesteps`` set to ``None``
            or not positive.
    """

    mode = canonical_time_conditioning(time_conditioning)
    masked = torch.as_tensor(masked_count).float()
    active = torch.as_tensor(
        active_count, device=masked.device
    ).float().clamp_min(1.0)
    if mode == "none":
        return None
    if mode == "continuous":
        return (masked / active).clamp(0.0, 1.0)
    if num_timesteps is None:
        raise ValueError("Discrete time conditioning requires num_timesteps")
    return timestep_from_mask_fraction(
        masked,
        active,
        int(num_timesteps),
        mask_schedule,
    )


__all__ = [
    "MASK_SCHEDULES",
    "TIME_CONDITIONING_MODES",
    "canonical_mask_schedule",
    "canonical_time_conditioning",
    "time_condition_from_mask_fraction",
    "timestep_from_mask_fraction",
]
