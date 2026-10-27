"""Joint-limit energy: quadratic penalty on parameters outside [lower, upper].

Residual per limited parameter ``r_i = theta_i - clamp(theta_i, lower_i, upper_i)`` (zero inside the
range), Jacobian 1 where the limit is active. Piecewise quadratic, so it slots into Gauss-Newton as is.
Infinite bounds disable a side. Core, torch-only.
"""
from __future__ import annotations

import torch
from torch import Tensor

from parafit.core.types import GNBlock, State
from parafit.energies.base import Energy, WeightLike, scale_block


class JointLimitEnergy(Energy):
    name = "joint_limits"

    def __init__(self, lower: Tensor, upper: Tensor, columns: slice | Tensor, weight: WeightLike = 1.0):
        """``lower`` / ``upper`` (K,) bounds of the parameters at ``columns`` (a slice or (K,) index tensor)."""
        self.lower, self.upper, self.columns, self.weight = lower, upper, columns, weight

    def linearize(self, model, params: Tensor, state: State) -> GNBlock:
        B, P = params.shape
        cols = self.columns
        if isinstance(cols, slice):
            cols = torch.arange(P, device=params.device)[cols]
        q = params[:, cols]
        lower, upper = self.lower.to(q), self.upper.to(q)
        r = q - torch.maximum(torch.minimum(q, upper), lower)                    # (B,K)
        active = (r != 0).to(q.dtype)
        A = params.new_zeros(B, P, P)
        g = params.new_zeros(B, P)
        A[:, cols, cols] = active
        g[:, cols] = r
        return scale_block(GNBlock(A=A, g=g, cost=0.5 * (r * r).sum(-1), batch_size=[B]), self.weight)
