"""Bound evaluation sampling memory while preserving complete prompt results."""

from __future__ import annotations

from typing import Any, Callable

import torch


def concatenate_sample_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate CPU sample tensors and weight scalar means by batch size.

    Nested dictionaries are combined key by key. Tensors with at least one
    dimension are concatenated along dimension 0; zero-dimensional tensors are
    averaged with weights equal to each batch's number of ``tokens`` rows.

    Args:
        batches (list[dict[str, Any]]): Non-empty list of sampler outputs with
            identical keys, each containing a ``tokens`` entry.

    Returns:
        dict[str, Any]: One output dictionary covering all batches.
    """

    sizes = [len(batch["tokens"]) for batch in batches]

    def combine(values: list[Any]) -> Any:
        """Combine one entry across all batches, recursing into dictionaries."""
        if isinstance(values[0], dict):
            return {key: combine([value[key] for value in values]) for key in values[0]}
        if values[0].ndim == 0:
            return sum(value * size for value, size in zip(values, sizes)) / sum(sizes)
        return torch.cat(values, dim=0)

    return combine(batches)


def sample_flux_in_batches(
    sampler: Any,
    *,
    num_samples: int,
    sampling_batch_size: int,
    seed: int,
    progress: Callable[[int, int], None] | None = None,
    **sample_kwargs: Any,
) -> dict[str, Any]:
    """Sample one prompt in bounded chunks, returning all outputs on CPU.

    Chunk zero keeps the prompt seed. Later chunks use a large seed stride so
    adjacent prompt seeds do not reuse the same random stream. Reproducibility
    requires both the seed and batch size to remain fixed. Progress counts
    completed sampling coordinates across all chunks.

    Args:
        sampler (Any): Object with a ``sample_flux(batch_size=..., seed=...,
            progress=..., **sample_kwargs)`` method returning a dictionary of
            tensors, such as :class:`StructuredFluxSampler`.
        num_samples (int): Total number of samples for this prompt.
        sampling_batch_size (int): Maximum number of samples per
            ``sample_flux`` call.
        seed (int): Prompt seed; chunk ``k`` uses
            ``(seed + k * 1_000_000_007) mod 2**64``.
        progress (Callable[[int, int], None] | None): Optional callback
            receiving ``(completed, total)`` summed over all chunks.
            Defaults to ``None``.
        **sample_kwargs (Any): Further keyword arguments for ``sample_flux``.

    Returns:
        dict[str, Any]: The outputs of all chunks on CPU, combined by
            :func:`concatenate_sample_batches`.

    Raises:
        ValueError: If ``num_samples`` or ``sampling_batch_size`` is not a
            positive integer.
    """

    for name, value in (("num_samples", num_samples), ("sampling_batch_size", sampling_batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    num_batches = (num_samples + sampling_batch_size - 1) // sampling_batch_size
    batches = []

    def to_cpu(value: Any) -> Any:
        """Detach a tensor, or each tensor of a nested dictionary, onto the CPU."""
        if isinstance(value, dict):
            return {key: to_cpu(item) for key, item in value.items()}
        return value.detach().cpu()

    for batch_index, start in enumerate(range(0, num_samples, sampling_batch_size)):
        def update(completed: int, total: int) -> None:
            """Forward this chunk's progress, offset by the completed chunks."""
            if progress is not None:
                progress(batch_index * total + completed, num_batches * total)

        generated = sampler.sample_flux(
            batch_size=min(sampling_batch_size, num_samples - start),
            seed=(seed + batch_index * 1_000_000_007) % (2**64),
            progress=update if progress is not None else None,
            **sample_kwargs,
        )
        batches.append(to_cpu(generated))
        del generated
    return concatenate_sample_batches(batches)
