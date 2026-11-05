# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-phase Kimi K3 AttnRes with the native attn_res_fwd tensor contract.

Ported from GitCode commit 6592017db. The native fused backend already
includes the output RMSNorm added by a0403dcc2; these Torch backends provide
an alternative algorithm and a single-pass reference without a DSL dependency.
"""

from typing import NamedTuple

import torch


class AttnResPhase1Stats(NamedTuple):
    inter_numerator: torch.Tensor
    inter_max: torch.Tensor
    inter_exp_sum: torch.Tensor


def prepare_attn_res_phase1(
    block_residual: torch.Tensor, effective_queries: torch.Tensor, epsilon: float
) -> AttnResPhase1Stats:
    """Compute FP32 historical statistics for all queries sharing a bank."""
    values = block_residual.float()
    if values.shape[1] == 0:
        shape = (effective_queries.shape[0], values.shape[0])
        return AttnResPhase1Stats(
            values.new_zeros((*shape, values.shape[-1])),
            values.new_full(shape, -torch.inf),
            values.new_zeros(shape),
        )
    inverse_rms = torch.rsqrt(values.square().mean(-1) + epsilon)
    logits = torch.matmul(values, effective_queries.T).permute(2, 0, 1) * inverse_rms.unsqueeze(0)
    maximum = logits.max(dim=-1).values
    exponentials = torch.exp(logits - maximum.unsqueeze(-1))
    numerator = torch.matmul(exponentials.permute(1, 0, 2), values).permute(1, 0, 2)
    return AttnResPhase1Stats(numerator, maximum, exponentials.sum(-1))


def merge_attn_res_partial(
    partial: torch.Tensor, query: torch.Tensor, stats: AttnResPhase1Stats, epsilon: float
) -> torch.Tensor:
    """Merge the current BF16-rounded partial using stable Online Softmax."""
    value = partial.float()
    logit = torch.matmul(value, query) * torch.rsqrt(value.square().mean(-1) + epsilon)
    maximum = torch.maximum(stats.inter_max, logit)
    historical_scale = torch.exp(stats.inter_max - maximum)
    partial_scale = torch.exp(logit - maximum)
    denominator = historical_scale * stats.inter_exp_sum + partial_scale
    numerator = historical_scale.unsqueeze(-1) * stats.inter_numerator + partial_scale.unsqueeze(-1) * value
    return (numerator / denominator.unsqueeze(-1)).to(partial.dtype)


class AttnResQueryPlan(torch.nn.Module):
    """Weight-only queries, prepared after loading and shared across forwards."""

    def __init__(self, specs):
        super().__init__()
        self.slots, queries = self._build_queries(specs)
        self._query_names: dict[tuple[int, float], str] = {}
        for index, (key, query) in enumerate(queries.items()):
            name = f"query_{index}"
            # Level-2 sleep restores named buffers, including derived weights.
            self.register_buffer(name, query, persistent=False)
            self._query_names[key] = name

    @property
    def queries(self) -> dict[tuple[int, float], torch.Tensor]:
        # Resolve the registered buffers so module device moves keep working.
        return {key: self.get_queries(key) for key in self._query_names}

    def get_queries(self, key: tuple[int, float]) -> torch.Tensor:
        return getattr(self, self._query_names[key])

    @staticmethod
    def _build_queries(specs):
        groups: dict[tuple[int, float], list[torch.Tensor]] = {}
        slots: list[tuple[tuple[int, float], int]] = []
        for projection, norm, valid_blocks in specs:
            key = (valid_blocks, norm.variance_epsilon)
            queries = groups.setdefault(key, [])
            slots.append((key, len(queries)))
            queries.append((projection.weight.squeeze(0).float() * norm.weight.float()).detach())
        return slots, {key: torch.stack(queries).contiguous() for key, queries in groups.items()}

    @torch.inference_mode()
    def refresh(self, specs) -> None:
        """Update derived weights without changing addresses captured by ACLGraph."""
        slots, queries = self._build_queries(specs)
        current = self.queries
        if slots != self.slots or any(query.shape != current[key].shape for key, query in queries.items()):
            raise ValueError("Cannot change the AttnRes query layout during weight reload")
        for key, query in queries.items():
            current[key].copy_(query)


def resolve_attn_res_mode(config) -> str:
    mode = getattr(config, "attn_res_mode", "fused")
    if mode not in ("fused", "native", "two_phase", "original"):
        raise ValueError(f"Unsupported Kimi K3 attn_res_mode: {mode!r}")
    return mode


class AttnResBackend:
    """Per-forward cache, callable with the native fused operator's signature.

    Historical rows stay fixed until a block boundary publishes the next row.
    All slots with that history share one Phase 1 call, including a PP stage
    starting inside a block. Statistics never survive a model forward.
    """

    def __init__(self, mode: str, plan: AttnResQueryPlan):
        self.mode = mode
        self.plan = plan
        self.cursor = 0
        self.stats_key: tuple[int, float] | None = None
        self.stats: AttnResPhase1Stats | None = None

    def _materialize(self, raw, blocks, projection, gamma, epsilon, valid):
        key, slot = self.plan.slots[self.cursor]
        self.cursor += 1
        if key != (valid, epsilon):
            raise ValueError("AttnRes query plan does not match the decoder residual points")
        if valid == 0:
            return raw
        if self.mode == "original":
            values = torch.cat((blocks[:, :valid], raw.unsqueeze(1)), dim=1).float()
            inverse_rms = torch.rsqrt(values.square().mean(-1) + epsilon)
            query = projection.squeeze(0).float() * gamma.float()
            logits = torch.matmul(values, query) * inverse_rms
            return torch.matmul(logits.softmax(-1).unsqueeze(1), values).squeeze(1).to(raw.dtype)
        queries = self.plan.get_queries(key)
        if key != self.stats_key:
            self.stats = prepare_attn_res_phase1(blocks[:, :valid], queries, epsilon)
            self.stats_key = key
        assert self.stats is not None
        stats = AttnResPhase1Stats(*(value[slot] for value in self.stats))
        return merge_attn_res_partial(raw, queries[slot], stats, epsilon)

    def __call__(
        self,
        prefix_sum,
        addend,
        block_residual,
        proj_weight,
        norm_weight,
        norm_eps,
        num_valid_blocks,
        output_norm_weight=None,
        output_norm_eps=1e-5,
        block_write_idx=-1,
        return_materialized=False,
        mix=True,
        optimize_prefill=False,
    ):
        # The current model contract rounds every residual add before scoring.
        # Do not carry the old DSL path's unrounded FP32 partial between layers.
        raw = prefix_sum if addend is None else (prefix_sum.float() + addend.float()).to(prefix_sum.dtype)
        materialized = raw
        if mix:
            materialized = self._materialize(raw, block_residual, proj_weight, norm_weight, norm_eps, num_valid_blocks)
        if block_write_idx >= 0:
            block_residual[:, block_write_idx].copy_(raw)
        output = materialized
        if output_norm_weight is not None:
            value = materialized.float()
            output = (
                value
                * torch.rsqrt(value.square().mean(-1, keepdim=True) + output_norm_eps)
                * output_norm_weight.float()
            ).to(raw.dtype)
        return output, raw, materialized if return_materialized else output
