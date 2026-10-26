"""Learned discrete codec for local observations sent to the CTDE server."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.debug_c4 import log


class ObservationCodec(nn.Module):
    """Per-observation nearest-codebook quantizer using one shared learned codebook."""

    def __init__(self, observation_dim: int, bit_width: int):
        super().__init__()

        if bit_width < 1:
            raise ValueError("bit_width must be positive")

        self.observation_dim = observation_dim
        self.bit_width = bit_width
        self.codebook_size = 1 << bit_width

        self.codebook = nn.Parameter(
            torch.empty(self.codebook_size, observation_dim)
        )
        nn.init.zeros_(self.codebook)

        self._initialized = False

    def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        # A loaded codebook must not be overwritten by lazy initialization.
        if prefix + "codebook" in state_dict:
            self._initialized = True

    @torch.no_grad()
    def _initialize_from_observations(
        self,
        observations: torch.Tensor,
    ) -> None:
        flat = observations.reshape(-1, self.observation_dim)

        if flat.shape[0] == 0:
            return

        if flat.shape[0] >= self.codebook_size:
            indices = torch.linspace(
                0,
                flat.shape[0] - 1,
                self.codebook_size,
                device=flat.device,
            ).round().long()
            prototypes = flat[indices]
        else:
            indices = torch.randint(
                0,
                flat.shape[0],
                (self.codebook_size,),
                device=flat.device,
            )
            prototypes = flat[indices]

        self.codebook.copy_(prototypes)
        self._initialized = True

    def encode(
        self,
        observations: torch.Tensor,
    ):
        if observations.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected final dimension {self.observation_dim}, "
                f"got {observations.shape[-1]}"
            )

        if not self._initialized:
            self._initialize_from_observations(observations.detach())

        flat = observations.reshape(-1, self.observation_dim)

        distances = (
            flat.square().sum(-1, keepdim=True)
            + self.codebook.square().sum(-1).unsqueeze(0)
            - 2.0 * flat @ self.codebook.t()
        )

        indices = distances.argmin(dim=-1)

        quantized = self.codebook[indices].reshape_as(observations)
        with torch.no_grad():
            err = (quantized - observations).float()
            unique_codes = int(indices.unique().numel()) if indices.numel() else 0
            log("VQ encode shape=%s init=%s codebook_size=%d unique_codes=%d indices=%s mse=%.8g max_abs_err=%.8g obs_min=%.6g obs_max=%.6g codebook_min=%.6g codebook_max=%.6g",
                tuple(observations.shape), self._initialized, self.codebook_size,
                unique_codes, indices.detach().reshape(-1).cpu().tolist()[:32],
                float(err.square().mean().item()) if err.numel() else 0.0,
                float(err.abs().max().item()) if err.numel() else 0.0,
                float(observations.min().item()) if observations.numel() else 0.0,
                float(observations.max().item()) if observations.numel() else 0.0,
                float(self.codebook.min().item()), float(self.codebook.max().item()))


        straight_through = (
            observations
            + (quantized - observations).detach()
        )

        codebook_loss = (
            quantized - observations.detach()
        ).square().mean()

        commitment_loss = (
            quantized.detach() - observations
        ).square().mean()

        vq_loss = codebook_loss + 0.25 * commitment_loss

        return (
            straight_through,
            indices.reshape(observations.shape[:-1]),
            vq_loss,
        )

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() and (
            indices.min() < 0
            or indices.max() >= self.codebook_size
        ):
            raise ValueError(
                "VQ symbol is outside the configured codebook"
            )

        decoded = self.codebook[indices.long()]
        log("VQ decode indices_shape=%s decoded_shape=%s unique_codes=%d",
            tuple(indices.shape), tuple(decoded.shape),
            int(indices.unique().numel()) if indices.numel() else 0)
        return decoded

    @property
    def bits_per_observation(self) -> int:
        return self.bit_width