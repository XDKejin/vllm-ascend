# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU numerical and contract tests; no vLLM installation or NPU required."""

import ast
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
backend = SimpleNamespace(**runpy.run_path(str(ROOT / "vllm_ascend/ops/kimi_attn_res.py")))


def reference(rows, partial, query, epsilon):
    values = torch.cat((rows, partial.unsqueeze(1)), dim=1).double()
    logits = (values * query.double()).sum(-1) * torch.rsqrt(values.square().mean(-1) + epsilon)
    return (logits.softmax(-1).unsqueeze(-1) * values).sum(1).to(partial.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tokens,blocks", [(3, 0), (3, 4), (0, 2)])
@pytest.mark.parametrize("query_scale", [1.0, 100.0])
def test_online_softmax_matches_independent_reference(dtype, tokens, blocks, query_scale):
    torch.manual_seed(31)
    rows = torch.randn(tokens, blocks, 16).to(dtype)
    partial = torch.randn(tokens, 16).to(dtype)
    queries = torch.randn(5, 16) * query_scale
    stats = backend.prepare_attn_res_phase1(rows, queries, 1e-6)
    assert all(value.dtype == torch.float32 for value in stats)
    for idx, query in enumerate(queries):
        slot = backend.AttnResPhase1Stats(*(value[idx] for value in stats))
        output = backend.merge_attn_res_partial(partial, query, slot, 1e-6)
        torch.testing.assert_close(output, reference(rows, partial, query, 1e-6), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("mode", ["original", "two_phase"])
def test_backend_preserves_add_rounding_write_and_materialized_contract(mode):
    projection = SimpleNamespace(weight=torch.tensor([[0.3, -0.2, 0.1]]))
    norm = SimpleNamespace(weight=torch.tensor([1.0, 0.5, 2.0]), variance_epsilon=1e-6)
    plan = backend.AttnResQueryPlan([(projection, norm, 1)])
    op = backend.AttnResBackend(mode, plan)
    prefix = torch.tensor([[256.0, 1.0, -1.0]], dtype=torch.bfloat16)
    delta = torch.tensor([[1.0, 0.003, 0.003]], dtype=torch.bfloat16)
    bank = torch.tensor([[[1.0, -2.0, 3.0], [999.0, 999.0, 999.0]]], dtype=torch.bfloat16)
    before = bank.clone()
    output, raw, materialized = op(
        prefix,
        delta,
        bank,
        projection.weight,
        norm.weight,
        1e-6,
        1,
        output_norm_weight=norm.weight,
        output_norm_eps=1e-4,
        block_write_idx=1,
        return_materialized=True,
    )
    expected_raw = (prefix.float() + delta.float()).to(prefix.dtype)
    expected = reference(before[:, :1], expected_raw, projection.weight.squeeze(0) * norm.weight, 1e-6)
    normalized = expected.float() * torch.rsqrt(expected.float().square().mean(-1, keepdim=True) + 1e-4)
    torch.testing.assert_close(raw, expected_raw, atol=0, rtol=0)
    torch.testing.assert_close(materialized, expected)
    torch.testing.assert_close(output, (normalized * norm.weight).to(prefix.dtype))
    torch.testing.assert_close(bank[:, 1], raw)
    torch.testing.assert_close(bank[:, 0], before[:, 0])
    assert prefix[0, 0] == 256  # The caller's prefix is never updated in place.


def test_phase1_is_reused_and_never_cached_across_forwards(monkeypatch):
    projection = SimpleNamespace(weight=torch.ones(1, 4))
    norm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
    plan = backend.AttnResQueryPlan([(projection, norm, 1)] * 3 + [(projection, norm, 2)])
    # Functions loaded by runpy share a globals dict, independently of this namespace.
    prepare_globals = backend.AttnResBackend._materialize.__globals__
    prepare = MagicMock(wraps=prepare_globals["prepare_attn_res_phase1"])
    monkeypatch.setitem(prepare_globals, "prepare_attn_res_phase1", prepare)
    for _ in range(2):
        op = backend.AttnResBackend("two_phase", plan)
        bank = torch.randn(2, 3, 4)
        raw = torch.randn(2, 4)
        for valid in (1, 1, 1, 2):
            op(raw, None, bank, projection.weight, norm.weight, 1e-6, valid)
    assert prepare.call_count == 4


def test_query_plan_refresh_preserves_fp32_buffers():
    projection = SimpleNamespace(weight=torch.tensor([[0.3, 0.4]], dtype=torch.bfloat16))
    norm = SimpleNamespace(weight=torch.tensor([0.7, 0.8], dtype=torch.bfloat16), variance_epsilon=1e-6)
    specs = [(projection, norm, 1)]
    with torch.inference_mode():
        plan = backend.AttnResQueryPlan(specs)
    query = plan.queries[(1, 1e-6)]
    pointer = query.data_ptr()
    expected = projection.weight.float() * norm.weight.float()
    assert query.dtype == torch.float32
    torch.testing.assert_close(query, expected)
    projection.weight.mul_(2)
    plan.refresh(specs)
    assert plan.queries[(1, 1e-6)] is query
    assert query.data_ptr() == pointer
    assert any(buffer is query for buffer in plan.buffers())
    assert not plan.state_dict()
    torch.testing.assert_close(query, expected * 2)


@pytest.mark.parametrize("change", ["slots", "epsilon", "hidden_size"])
def test_query_plan_rejects_layout_changes_on_reload(change):
    projection = SimpleNamespace(weight=torch.ones(1, 4))
    norm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
    specs = [(projection, norm, 1)]
    plan = backend.AttnResQueryPlan(specs)
    query = plan.queries[(1, 1e-6)]
    expected = query.clone()
    if change == "slots":
        specs.append((projection, norm, 1))
    elif change == "epsilon":
        norm.variance_epsilon = 1e-5
    else:
        projection.weight = torch.zeros(1, 8)
        norm.weight = torch.ones(8)
    with pytest.raises(ValueError, match="query layout"):
        plan.refresh(specs)
    torch.testing.assert_close(query, expected)


def test_modes_preserve_native_default_and_reject_typos():
    assert backend.resolve_attn_res_mode(SimpleNamespace()) == "fused"
    for mode in ("native", "fused", "two_phase", "original"):
        assert backend.resolve_attn_res_mode(SimpleNamespace(attn_res_mode=mode)) == mode
    with pytest.raises(ValueError, match="attn_res_mode"):
        backend.resolve_attn_res_mode(SimpleNamespace(attn_res_mode="typo"))


def test_two_phase_forward_cache_supports_full_graph_capture():
    projection = SimpleNamespace(weight=torch.ones(1, 4))
    norm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
    plan = backend.AttnResQueryPlan([(projection, norm, 1)] * 2)

    def forward(partial, delta, bank):
        op = backend.AttnResBackend("two_phase", plan)
        output, raw, _ = op(partial, delta, bank, projection.weight, norm.weight, 1e-6, 1)
        return op(raw, output, bank, projection.weight, norm.weight, 1e-6, 1)[0]

    compiled = torch.compile(forward, backend="eager", fullgraph=True)
    for _ in range(2):
        partial, delta = torch.randn(2, 4), torch.randn(2, 4)
        bank = torch.randn(2, 1, 4)
        torch.testing.assert_close(compiled(partial, delta, bank), forward(partial, delta, bank))


@pytest.mark.parametrize("group", ["tp", "dp", "mc2"])
def test_hccl_options_keep_runtime_communication_default(group):
    # PR a0403dcc2 removed forced CCU for prefill. v0.30 already delegates
    # communication to HCCL for every role; preserve buffer and group settings.
    source = ast.parse((ROOT / "vllm_ascend/utils.py").read_text())
    node = next(node for node in source.body if getattr(node, "name", None) == "create_hccl_pg_options")
    namespace = {
        "torch_npu": MagicMock(),
        "get_hccl_config_for_pg_options": lambda name: None if name == "mc2" else {"hccl_buffer_size": 200},
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "utils.py", "exec"), namespace)
    options = namespace["create_hccl_pg_options"](group)
    assert "hccl_op_expansion_mode" not in options.hccl_config
    assert options.hccl_config["group_name"] == group
    if group != "mc2":
        assert options.hccl_config["hccl_buffer_size"] == 200
