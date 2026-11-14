"""Geometry-conditioned causal flux/moduli transformer architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn

from .layers import (
    AdaLNTransformerBlock,
    ContinuousTimeMLP,
    QueryCrossAttentionBlock,
    TimestepMLP,
)
from .observable_conditioning import (
    MultiObservableTokenConditioner,
    ObservableConditioningContext,
    ObservableTokenConditioner,
    SUPPORTED_CONDITIONAL_OBSERVABLES,
    canonical_observable_name,
)


@dataclass(frozen=True)
class FluxVacuumTransformerConfig:
    """Architecture settings of :class:`FluxVacuumTransformer`.

    Attributes:
        input_vocab_size (int): Value tokens plus ``PAD`` and ``MASK``; must
            equal ``output_vocab_size + 2``.
        output_vocab_size (int): Number of value tokens.
        pad_token_id (int): Must equal ``output_vocab_size``.
        mask_token_id (int): Must equal ``output_vocab_size + 1``.
        num_geometries (int): Rows of the per-geometry normalisation tables.
        geometry_feature_mean (tuple[float, float, float]): Finite means of
            the ``(h11, h12, tadpole_bound)`` features.
        geometry_feature_std (tuple[float, float, float]): Standard deviations
            of those features; finite and positive.
        max_h12 (int): Largest ``h12``.
        max_flux_dim (int): Padded flux length; must equal
            ``4 * (max_h12 + 1)``.
        model_dim (int): Model width.
        attention_heads (int): Number of attention heads.
        mlp_ratio (int): MLP hidden width in units of ``model_dim``.
        dropout (float): Dropout probability.
        shared_transformer_layers (int): Shared encoder blocks; at least 1.
        flux_policy_layers (int): Flux-branch blocks; at least 1.
        moduli_decoder_layers (int): Moduli cross-attention blocks.
        moduli_coordinate_margin_std (float): Non-negative margin, in units of
            the per-coordinate standard deviation, added outside the stored
            quantiles to form the initializer clamping bounds.
        observable_conditioning_enabled (bool): Whether to build an observable
            conditioner.
        observable_token_width (int): Positive hidden width of the observable
            encoder.
        conditional_observable (str): Single conditioned observable, used when
            ``conditional_observables`` is empty.
        conditional_observables (tuple[str, ...]): Ordered, unique conditioned
            observables.
        conditional_transform_delta (float): Positive offset of the
            ``W0_abs`` transform.
        time_conditioning (str): ``"discrete"``, ``"continuous"`` or ``"none"``.
    """

    input_vocab_size: int
    output_vocab_size: int
    pad_token_id: int
    mask_token_id: int
    num_geometries: int = 20
    geometry_feature_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    geometry_feature_std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_h12: int = 5
    max_flux_dim: int = 24
    model_dim: int = 256
    attention_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0
    shared_transformer_layers: int = 3
    flux_policy_layers: int = 2
    moduli_decoder_layers: int = 2
    moduli_coordinate_margin_std: float = 1.0
    observable_conditioning_enabled: bool = False
    observable_token_width: int = 128
    conditional_observable: str = "N_flux"
    conditional_observables: tuple[str, ...] = ()
    conditional_transform_delta: float = 1.0e-8
    time_conditioning: str = "discrete"

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises:
            ValueError: If any constraint stated in the class docstring is
                violated, or an observable name is not exactly one of
                :data:`SUPPORTED_CONDITIONAL_OBSERVABLES`.
        """
        if self.shared_transformer_layers <= 0 or self.flux_policy_layers <= 0:
            raise ValueError("Both shared and flux-policy branches require at least one block")
        if self.max_flux_dim != 4 * (self.max_h12 + 1):
            raise ValueError("max_flux_dim must equal 4 * (max_h12 + 1)")
        if self.output_vocab_size + 2 != self.input_vocab_size:
            raise ValueError("Input vocabulary must be value tokens plus PAD and MASK")
        if self.pad_token_id != self.output_vocab_size:
            raise ValueError("PAD must immediately follow the physical value tokens")
        if self.mask_token_id != self.output_vocab_size + 1:
            raise ValueError("MASK must immediately follow PAD")
        if len(self.geometry_feature_mean) != 3 or len(self.geometry_feature_std) != 3:
            raise ValueError("Geometry normalization must contain h11, h12, and tadpole")
        if not all(math.isfinite(float(value)) for value in self.geometry_feature_mean):
            raise ValueError("Geometry feature means must be finite")
        if not all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in self.geometry_feature_std
        ):
            raise ValueError("Geometry feature standard deviations must be finite and positive")
        if not math.isfinite(self.moduli_coordinate_margin_std) or self.moduli_coordinate_margin_std < 0.0:
            raise ValueError("Moduli coordinate margin must be finite and non-negative")
        if self.observable_token_width <= 0:
            raise ValueError("observable_token_width must be positive")
        if self.conditional_observable not in SUPPORTED_CONDITIONAL_OBSERVABLES:
            raise ValueError(
                "conditional_observable must be one of "
                f"{SUPPORTED_CONDITIONAL_OBSERVABLES}"
            )
        if self.conditional_observables:
            if len(set(self.conditional_observables)) != len(
                self.conditional_observables
            ):
                raise ValueError("conditional_observables must be unique")
            if any(
                observable not in SUPPORTED_CONDITIONAL_OBSERVABLES
                for observable in self.conditional_observables
            ):
                raise ValueError(
                    "conditional_observables entries must be among "
                    f"{SUPPORTED_CONDITIONAL_OBSERVABLES}"
                )
        if (
            not math.isfinite(float(self.conditional_transform_delta))
            or float(self.conditional_transform_delta) <= 0.0
        ):
            raise ValueError("conditional_transform_delta must be finite and positive")
        if self.time_conditioning not in {"discrete", "continuous", "none"}:
            raise ValueError(
                "time_conditioning must be discrete, continuous, or none"
            )


    def to_dict(self) -> dict[str, Any]:
        """Return the settings as a dictionary (:func:`dataclasses.asdict`).

        Returns:
            dict[str, Any]: Field names mapped to values.
        """
        return asdict(self)

    @property
    def conditioned_observables(self) -> tuple[str, ...]:
        """Canonical ordered observables, including legacy single-token configs.

        Returns:
            tuple[str, ...]: ``conditional_observables`` if non-empty, otherwise
                ``(conditional_observable,)``.
        """

        if self.conditional_observables:
            return self.conditional_observables
        return (self.conditional_observable,)


def flux_active_mask(
    h12: torch.Tensor,
    max_flux_dim: int,
) -> torch.Tensor:
    """Return the structural mask for ``L = 4 * (h12 + 1)``.

    Args:
        h12 (torch.Tensor): ``h12`` per row, flattened to shape ``(batch,)``.
        max_flux_dim (int): Padded flux length.

    Returns:
        torch.Tensor: Boolean mask of shape ``(batch, max_flux_dim)``, ``True``
            on each row's first ``4 * (h12 + 1)`` positions.

    Raises:
        ValueError: If an ``h12`` value lies outside ``[1, 5]``.
    """

    h12 = h12.long().reshape(-1)
    if bool(((h12 < 1) | (h12 > 5)).any()):
        raise ValueError("h12 must lie in [1, 5]")
    position = torch.arange(max_flux_dim, device=h12.device).unsqueeze(0)
    width = (h12 + 1).unsqueeze(1)
    return position < 4 * width


def moduli_query_mask(h12: torch.Tensor) -> torch.Tensor:
    """Return which of the six moduli queries (five ``z`` slots, then ``tau``) are used.

    Args:
        h12 (torch.Tensor): ``h12`` per row, flattened to shape ``(batch,)``.

    Returns:
        torch.Tensor: Boolean mask of shape ``(batch, 6)``: the first ``h12``
            ``z`` slots and the ``tau`` slot are ``True``.
    """
    h12 = h12.long().reshape(-1)
    z_index = torch.arange(5, device=h12.device).unsqueeze(0)
    z_active = z_index < h12.unsqueeze(1)
    return torch.cat(
        (z_active, torch.ones((h12.shape[0], 1), dtype=torch.bool, device=h12.device)),
        dim=1,
    )


class FluxVacuumTransformer(nn.Module):
    """One model with a shared encoder and causally separated output branches."""

    def __init__(
        self,
        config: FluxVacuumTransformerConfig,
        *,
        moduli_statistics: dict[str, Any] | None = None,
    ) -> None:
        """Build the model.

        Args:
            config (FluxVacuumTransformerConfig): Architecture settings.
            moduli_statistics (dict[str, Any] | None): Per-geometry moduli
                normalisation statistics (see
                :func:`validate_moduli_statistics`), or ``None`` for neutral
                placeholders. Defaults to ``None``.

        Raises:
            ValueError: If the moduli statistics give non-finite values,
                non-positive standard deviations or reversed bounds.
        """
        super().__init__()
        self.config = config
        dim = config.model_dim
        self.value_embedding = nn.Embedding(
            config.input_vocab_size, dim, padding_idx=config.pad_token_id
        )
        self.coordinate_embedding = nn.Embedding(config.max_flux_dim, dim)
        self.geometry_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        if config.time_conditioning == "continuous":
            self.mask_probability_mlp = ContinuousTimeMLP(dim)
        elif config.time_conditioning == "discrete":
            # Preserve this module name and its state-dict keys for legacy
            # discrete-timestep checkpoints.
            self.timestep_mlp = TimestepMLP(dim)
        else:
            # Consume the same initialization draws as the continuous model so
            # paired runs with one seed start every shared parameter identically.
            # The temporary module is deliberately not registered or retained.
            _ = ContinuousTimeMLP(dim)
        geometry_mean = torch.tensor(config.geometry_feature_mean, dtype=torch.float32)
        geometry_std = torch.tensor(config.geometry_feature_std, dtype=torch.float32)
        self.register_buffer("h11_mean", geometry_mean[0].clone(), persistent=True)
        self.register_buffer("h12_mean", geometry_mean[1].clone(), persistent=True)
        self.register_buffer("tadpole_mean", geometry_mean[2].clone(), persistent=True)
        self.register_buffer("h11_std", geometry_std[0].clone(), persistent=True)
        self.register_buffer("h12_std", geometry_std[1].clone(), persistent=True)
        self.register_buffer("tadpole_std", geometry_std[2].clone(), persistent=True)

        block_kwargs = {
            "model_dim": dim,
            "num_heads": config.attention_heads,
            "mlp_ratio": config.mlp_ratio,
            "dropout": config.dropout,
        }
        self.shared_blocks = nn.ModuleList(
            [
                AdaLNTransformerBlock(**block_kwargs)
                for _ in range(config.shared_transformer_layers)
            ]
        )
        self.flux_blocks = nn.ModuleList(
            [
                AdaLNTransformerBlock(**block_kwargs)
                for _ in range(config.flux_policy_layers)
            ]
        )
        self.shared_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.flux_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.flux_head = nn.Linear(dim, config.output_vocab_size)
        if not config.observable_conditioning_enabled:
            self.observable_conditioner = None
        elif len(config.conditioned_observables) == 1:
            # Keep the legacy module/state-dict layout exactly unchanged.
            self.observable_conditioner = ObservableTokenConditioner(
                model_dim=dim,
                hidden_width=config.observable_token_width,
                num_geometries=config.num_geometries,
                observable=config.conditioned_observables[0],
                transform_delta=config.conditional_transform_delta,
            )
        else:
            self.observable_conditioner = MultiObservableTokenConditioner(
                model_dim=dim,
                hidden_width=config.observable_token_width,
                num_geometries=config.num_geometries,
                observables=config.conditioned_observables,
                transform_delta=config.conditional_transform_delta,
            )

        self.moduli_queries = nn.Parameter(torch.randn(6, dim) * 0.02)
        self.moduli_decoder = nn.ModuleList(
            [
                QueryCrossAttentionBlock(**block_kwargs)
                for _ in range(config.moduli_decoder_layers)
            ]
        )
        self.moduli_norm = nn.LayerNorm(dim)
        self.moduli_head = nn.Linear(dim, 2)
        means, standard_deviations, coordinate_min, coordinate_max = (
            self._normalization_tensors(moduli_statistics)
        )
        self.register_buffer("moduli_mean", means, persistent=True)
        self.register_buffer("moduli_std", standard_deviations, persistent=True)
        self.register_buffer("moduli_coordinate_min", coordinate_min, persistent=True)
        self.register_buffer("moduli_coordinate_max", coordinate_max, persistent=True)

    def _normalization_tensors(
        self, stats: dict[str, Any] | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the moduli normalisation and clamping tables.

        Rows are looked up as ``stats["geometries"][str(geometry_id)]``; a
        missing row keeps mean 0, standard deviation 1 and bounds ``[-10, 10]``.
        Otherwise the bounds are the stored quantiles widened by
        ``moduli_coordinate_margin_std`` standard deviations.

        Args:
            stats (dict[str, Any] | None): Moduli statistics, or ``None``.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Mean, standard deviation, lower and upper bound, each ``float32``
                of shape ``(num_geometries, 6, 2)``.

        Raises:
            ValueError: If a table is not finite, a standard deviation is not
                positive, or a lower bound exceeds its upper bound.
        """
        mean = torch.zeros(self.config.num_geometries, 6, 2, dtype=torch.float32)
        std = torch.ones_like(mean)
        lower = torch.full_like(mean, -10.0)
        upper = torch.full_like(mean, 10.0)
        if stats is None:
            return mean, std, lower, upper
        rows = stats.get("geometries", {})
        for geometry_id in range(self.config.num_geometries):
            row = rows.get(str(geometry_id))
            if row is None:
                # Keep neutral placeholders for unavailable normalization rows.
                continue
            mean[geometry_id] = torch.as_tensor(row["mean"], dtype=torch.float32)
            std[geometry_id] = torch.as_tensor(row["std"], dtype=torch.float32)
            quantile_low = torch.as_tensor(row["quantile_low"], dtype=torch.float32)
            quantile_high = torch.as_tensor(row["quantile_high"], dtype=torch.float32)
            margin = self.config.moduli_coordinate_margin_std * std[geometry_id]
            lower[geometry_id] = quantile_low - margin
            upper[geometry_id] = quantile_high + margin
        if not bool(
            torch.isfinite(mean).all()
            and torch.isfinite(std).all()
            and torch.isfinite(lower).all()
            and torch.isfinite(upper).all()
        ):
            raise ValueError("Moduli normalization must be finite")
        if bool((std <= 0.0).any()):
            raise ValueError("Moduli standard deviations must be positive")
        if bool((lower > upper).any()):
            raise ValueError("Moduli coordinate safety bounds are reversed")
        return mean, std, lower, upper


    def normalized_geometry_features(
        self,
        h11: torch.Tensor,
        h12: torch.Tensor,
    ) -> torch.Tensor:
        """Build normalised physical ``(h11, h12, tadpole_bound)`` features.

        The code sets ``tadpole_bound = h11 + h12 + 2``.

        Args:
            h11 (torch.Tensor): ``h11`` per row.
            h12 (torch.Tensor): ``h12`` per row.

        Returns:
            torch.Tensor: Standardised features of shape ``(batch, 3)``.

        Raises:
            ValueError: If ``h11`` and ``h12`` differ in size.
        """

        h11_f = h11.float().reshape(-1)
        h12_f = h12.float().reshape(-1)
        if h11_f.shape != h12_f.shape:
            raise ValueError("h11 and h12 must have the same batch shape")
        tadpole_bound = h11_f + h12_f + 2.0
        return torch.stack(
            (
                (h11_f - self.h11_mean) / self.h11_std,
                (h12_f - self.h12_mean) / self.h12_std,
                (tadpole_bound - self.tadpole_mean) / self.tadpole_std,
            ),
            dim=-1,
        )

    @property
    def geometry_feature_mean(self) -> torch.Tensor:
        """torch.Tensor: Stored feature means ``(h11, h12, tadpole_bound)``, shape ``(3,)``."""
        return torch.stack((self.h11_mean, self.h12_mean, self.tadpole_mean))

    @property
    def geometry_feature_std(self) -> torch.Tensor:
        """torch.Tensor: Stored feature standard deviations, shape ``(3,)``."""
        return torch.stack((self.h11_std, self.h12_std, self.tadpole_std))

    @property
    def time_conditioner(self) -> nn.Module | None:
        """nn.Module | None: The time embedding module, or ``None`` without time conditioning."""
        if self.config.time_conditioning == "continuous":
            return self.mask_probability_mlp
        if self.config.time_conditioning == "discrete":
            return self.timestep_mlp
        return None

    def shared_encode(
        self,
        tokens: torch.Tensor,
        active_mask: torch.Tensor,
        h11: torch.Tensor,
        h12: torch.Tensor,
        time_condition: torch.Tensor | None,
        *,
        observable_token: torch.Tensor | None = None,
        observable_active: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the shared encoder, optionally with prepended observable tokens.

        Args:
            tokens (torch.Tensor): Token IDs of shape ``(batch, max_flux_dim)``.
            active_mask (torch.Tensor): Mask of the same shape; must equal
                :func:`flux_active_mask` of ``h12``.
            h11 (torch.Tensor): ``h11`` per row, shape ``(batch,)``.
            h12 (torch.Tensor): ``h12`` per row, shape ``(batch,)``.
            time_condition (torch.Tensor | None): Timestep (discrete) or mask
                probability (continuous) per row; required unless time
                conditioning is ``none``, in which case it is ignored.
            observable_token (torch.Tensor | None): Observable tokens of shape
                ``(batch, model_dim)`` or ``(batch, k, model_dim)``, placed
                before the flux positions. Defaults to ``None``.
            observable_active (torch.Tensor | None): Their activity mask, of
                shape ``(batch,)`` or ``(batch, k)``; required with
                ``observable_token``. Defaults to ``None``.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Hidden states of
                shape ``(batch, k + max_flux_dim, model_dim)`` (``k = 0`` without
                observable tokens), the AdaLN condition of shape
                ``(batch, model_dim)``, and the boolean sequence mask of shape
                ``(batch, k + max_flux_dim)``.

        Raises:
            ValueError: If a shape is inconsistent, an ``h12`` value lies
                outside ``[1, 5]``, ``active_mask`` differs from the structural
                mask, a time condition is missing, or only one
                of ``observable_token`` and ``observable_active`` is given.
        """
        active = active_mask.bool()
        if tokens.shape != active.shape or tokens.shape[1] != self.config.max_flux_dim:
            raise ValueError("tokens and active_mask must have shape [batch, max_flux_dim]")
        structural_mask = flux_active_mask(h12, self.config.max_flux_dim)
        if not torch.equal(active, structural_mask):
            raise ValueError("active_mask is inconsistent with L = 4 * (h12 + 1)")
        if h11.numel() != tokens.shape[0] or h12.numel() != tokens.shape[0]:
            raise ValueError("h11 and h12 must contain one value per batch row")
        geometry_features = self.normalized_geometry_features(h11, h12)
        geometry_pool = self.geometry_mlp(geometry_features)
        time_conditioner = self.time_conditioner
        if time_conditioner is None:
            condition = geometry_pool
        else:
            if time_condition is None:
                raise ValueError(
                    "A time-conditioned model requires a time condition"
                )
            condition = geometry_pool + time_conditioner(time_condition)
        coordinate_ids = torch.arange(
            self.config.max_flux_dim, device=tokens.device
        ).unsqueeze(0).expand(tokens.shape[0], -1)
        hidden = (
            self.value_embedding(tokens.long())
            + self.coordinate_embedding(coordinate_ids)
        )
        hidden = hidden * active.unsqueeze(-1).to(hidden.dtype)
        if observable_token is not None:
            if observable_active is None:
                raise ValueError("An observable token requires an active mask")
            if observable_token.ndim == 2:
                observable_token = observable_token.unsqueeze(1)
            if observable_active.ndim == 1:
                observable_active = observable_active.unsqueeze(1)
            observable_active = observable_active.bool()
            if (
                observable_token.ndim != 3
                or observable_token.shape[0] != tokens.shape[0]
                or observable_token.shape[2] != self.config.model_dim
            ):
                raise ValueError(
                    "observable_token must have shape [batch, tokens, model_dim]"
                )
            expected_active = observable_token.shape[:2]
            if observable_active.shape != expected_active:
                raise ValueError(
                    "observable_active must have shape [batch, observable_tokens]"
                )
            hidden = torch.cat((observable_token, hidden), dim=1)
            active = torch.cat((observable_active, active), dim=1)
            hidden = hidden * active.unsqueeze(-1).to(hidden.dtype)
        elif observable_active is not None:
            raise ValueError("observable_active was provided without a token")
        for block in self.shared_blocks:
            hidden = block(hidden, condition, active)
        return hidden, condition, active

    def denoise(
        self,
        tokens: torch.Tensor,
        active_mask: torch.Tensor,
        h11: torch.Tensor,
        h12: torch.Tensor,
        time_condition: torch.Tensor | None,
        *,
        observable_context: ObservableConditioningContext | None = None,
    ) -> torch.Tensor:
        """Return flux-policy logits.

        Args:
            tokens (torch.Tensor): Token IDs of shape ``(batch, max_flux_dim)``.
            active_mask (torch.Tensor): Mask of the same shape.
            h11 (torch.Tensor): ``h11`` per row, shape ``(batch,)``.
            h12 (torch.Tensor): ``h12`` per row, shape ``(batch,)``.
            time_condition (torch.Tensor | None): Time input, as for
                :meth:`shared_encode`.
            observable_context (ObservableConditioningContext | None):
                Observable targets, or ``None``. Defaults to ``None``.

        Returns:
            torch.Tensor: Logits of shape
                ``(batch, max_flux_dim, output_vocab_size)``, zero at inactive
                positions.
        """
        hidden = self.flux_hidden(
            tokens,
            active_mask,
            h11,
            h12,
            time_condition,
            observable_context=observable_context,
        )
        logits = self.flux_head(hidden)
        return logits.masked_fill(~active_mask.bool().unsqueeze(-1), 0.0)

    def flux_hidden(
        self,
        tokens: torch.Tensor,
        active_mask: torch.Tensor,
        h11: torch.Tensor,
        h12: torch.Tensor,
        time_condition: torch.Tensor | None,
        *,
        observable_context: ObservableConditioningContext | None = None,
    ) -> torch.Tensor:
        """Return the normalised representation immediately before ``flux_head``.

        A selected physical observable enters as one explicit sequence token.
        Keeping the categorical head outside this method makes the feature
        boundary explicit without introducing a second flux-policy path.
        A model with an observable conditioner but no ``observable_context``
        uses zero, inactive observable tokens.

        Args:
            tokens (torch.Tensor): Token IDs of shape ``(batch, max_flux_dim)``.
            active_mask (torch.Tensor): Mask of the same shape.
            h11 (torch.Tensor): ``h11`` per row, shape ``(batch,)``.
            h12 (torch.Tensor): ``h12`` per row, shape ``(batch,)``.
            time_condition (torch.Tensor | None): Time input, as for
                :meth:`shared_encode`.
            observable_context (ObservableConditioningContext | None):
                Observable targets, or ``None``. Defaults to ``None``.

        Returns:
            torch.Tensor: Normalised flux representations of shape
                ``(batch, max_flux_dim, model_dim)``, zero at inactive positions.

        Raises:
            ValueError: If an active context is given to a model without an
                observable conditioner, or an input is inconsistent.
        """

        observable_token: torch.Tensor | None = None
        observable_active: torch.Tensor | None = None
        if observable_context is not None and self.observable_conditioner is None:
            if observable_context.any_active():
                raise ValueError("This model has no native observable conditioner")
        elif self.observable_conditioner is not None:
            token_count = len(self.config.conditioned_observables)
            if observable_context is None:
                observable_token = self.value_embedding.weight.new_zeros(
                    (tokens.shape[0], token_count, self.config.model_dim)
                )
                observable_active = torch.zeros(
                    (tokens.shape[0], token_count),
                    dtype=torch.bool,
                    device=tokens.device,
                )
            else:
                encoded = self.observable_conditioner(observable_context)
                if isinstance(encoded, tuple):
                    observable_token, observable_active = encoded
                else:
                    observable_token = encoded.unsqueeze(1)
                    selected = observable_context.select(
                        self.config.conditioned_observables[0]
                    )
                    observable_active = torch.as_tensor(
                        selected.active, device=tokens.device
                    ).bool().reshape(-1, 1)
        hidden, condition, transformer_mask = self.shared_encode(
            tokens,
            active_mask,
            h11,
            h12,
            time_condition,
            observable_token=observable_token,
            observable_active=observable_active,
        )
        for block in self.flux_blocks:
            hidden = block(hidden, condition, transformer_mask)
        if observable_token is not None:
            hidden = hidden[:, observable_token.shape[1] :]
        return self.flux_norm(hidden) * active_mask.unsqueeze(-1).to(hidden.dtype)

    def forward(
        self,
        tokens: torch.Tensor,
        active_mask: torch.Tensor,
        h11: torch.Tensor,
        h12: torch.Tensor,
        time_condition: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Return the flux logits of :meth:`denoise` without an observable context.

        Args:
            tokens (torch.Tensor): Token IDs of shape ``(batch, max_flux_dim)``.
            active_mask (torch.Tensor): Mask of the same shape.
            h11 (torch.Tensor): ``h11`` per row, shape ``(batch,)``.
            h12 (torch.Tensor): ``h12`` per row, shape ``(batch,)``.
            time_condition (torch.Tensor | None): Time input, as for
                :meth:`shared_encode`.

        Returns:
            dict[str, torch.Tensor]: ``{"logits": ...}`` with logits of shape
                ``(batch, max_flux_dim, output_vocab_size)``.
        """
        return {
            "logits": self.denoise(
                tokens, active_mask, h11, h12, time_condition
            )
        }

    def predict_initializer(
        self,
        clean_tokens: torch.Tensor,
        active_mask: torch.Tensor,
        h11: torch.Tensor,
        h12: torch.Tensor,
        normalization_geometry_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict solver initial values for the moduli ``z`` and ``tau``.

        The shared encoder reads the complete flux tokens at time input zero
        (none if time conditioning is ``none``) and without observable tokens;
        six queries (five ``z`` slots, then ``tau``) cross-attend to it. The
        de-standardised coordinates ``(c0, c1)`` of each query are clamped to
        the per-geometry bounds and mapped to ``z = c0 + i sinh(c1)`` and
        ``tau = c0 + i exp(c1)``.

        Args:
            clean_tokens (torch.Tensor): Unmasked token IDs of shape
                ``(batch, max_flux_dim)``.
            active_mask (torch.Tensor): Mask of the same shape.
            h11 (torch.Tensor): ``h11`` per row, shape ``(batch,)``.
            h12 (torch.Tensor): ``h12`` per row, shape ``(batch,)``.
            normalization_geometry_id (torch.Tensor): Row index into the
                normalisation tables, shape ``(batch,)``.

        Returns:
            dict[str, torch.Tensor]: ``standardized``, ``coordinates_unclamped``
                and ``coordinates`` of shape ``(batch, 6, 2)``;
                ``coordinate_clamped`` (bool, same shape) marking active
                coordinates changed by clamping; ``clamp_fraction``, the scalar
                fraction of active coordinates clamped; ``query_mask`` of shape
                ``(batch, 6)``; complex ``z`` of shape ``(batch, 5)`` and ``tau``
                of shape ``(batch,)``, zero on unused ``z`` slots.

        Raises:
            FloatingPointError: If the standardised coordinates or the complex
                values are not finite.
            ValueError: If the inputs are inconsistent (see
                :meth:`shared_encode`).
        """
        time_condition = (
            None
            if self.config.time_conditioning == "none"
            else torch.zeros_like(
                h12,
                dtype=(
                    torch.float32
                    if self.config.time_conditioning == "continuous"
                    else torch.long
                ),
            )
        )
        memory, _, _ = self.shared_encode(
            clean_tokens,
            active_mask,
            h11,
            h12,
            time_condition,
        )
        memory = self.shared_norm(memory)
        query_mask = moduli_query_mask(h12)
        queries = self.moduli_queries.unsqueeze(0).expand(clean_tokens.shape[0], -1, -1)
        queries = queries * query_mask.unsqueeze(-1).to(queries.dtype)
        for block in self.moduli_decoder:
            queries = block(queries, memory, active_mask, query_mask)
        standardized = self.moduli_head(self.moduli_norm(queries))
        standardized = standardized * query_mask.unsqueeze(-1).to(standardized.dtype)
        if not bool(torch.isfinite(standardized).all()):
            raise FloatingPointError("Initializer produced non-finite standardized coordinates")
        mean = self.moduli_mean[normalization_geometry_id.long()]
        std = self.moduli_std[normalization_geometry_id.long()]
        coordinates_unclamped = standardized * std + mean
        coordinate_min = self.moduli_coordinate_min[normalization_geometry_id.long()]
        coordinate_max = self.moduli_coordinate_max[normalization_geometry_id.long()]
        coordinates = torch.maximum(
            torch.minimum(coordinates_unclamped, coordinate_max), coordinate_min
        )
        coordinates = coordinates * query_mask.unsqueeze(-1).to(coordinates.dtype)
        clamped = (coordinates != coordinates_unclamped) & query_mask.unsqueeze(-1)
        z_values = torch.complex(
            coordinates[:, :5, 0], coordinates[:, :5, 1].sinh()
        )
        tau_value = torch.complex(
            coordinates[:, 5, 0], coordinates[:, 5, 1].exp()
        )
        complex_values = torch.cat((z_values, tau_value.unsqueeze(1)), dim=1)
        complex_values = complex_values * query_mask.to(complex_values.dtype)
        if not bool(
            torch.isfinite(complex_values.real).all()
            and torch.isfinite(complex_values.imag).all()
        ):
            raise FloatingPointError("Initializer inverse transform produced non-finite values")
        return {
            "standardized": standardized,
            "coordinates_unclamped": coordinates_unclamped,
            "coordinates": coordinates,
            "coordinate_clamped": clamped,
            "clamp_fraction": clamped.float().sum()
            / (2.0 * query_mask.float().sum()).clamp_min(1.0),
            "query_mask": query_mask,
            "z": complex_values[:, :5],
            "tau": complex_values[:, 5],
        }


