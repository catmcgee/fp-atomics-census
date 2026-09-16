#!/usr/bin/env python3
"""Pure-Python validation for vLLM's native routed-expert observations.

The GPU implementation is vLLM 0.29's ``enable_return_routed_experts`` path.
It writes the router's selected ``topk_ids`` to a preallocated device tensor
inside the model forward, including CUDA-graph replay, and copies only after
the forward has completed.  This module deliberately has no torch or vLLM
import so its schema and validation can be tested on CPU.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence


SCHEMA = 1


class RoutingTelemetryError(ValueError):
    """The returned array cannot support an actual-routing claim."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest_token_ids(ids: Sequence[int]) -> str:
    if any(not isinstance(item, int) or isinstance(item, bool) for item in ids):
        raise RoutingTelemetryError("token IDs must be integers")
    return hashlib.sha256(canonical_json(list(ids))).hexdigest()


def _tolist(value: Any) -> list[Any]:
    converted = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(converted, list):
        raise RoutingTelemetryError("routing data must be a rank-3 array")
    return converted


def summarise_selected_experts(
    routing_data: Any,
    *,
    num_experts: int,
    valid_tokens: int,
) -> dict[str, Any]:
    """Summarise actual router IDs, trimming any static-capture padding.

    ``routing_data`` has vLLM's public shape
    ``[returned_tokens, layers, experts_per_token]``.  CUDA graph capture may
    reserve a larger internal tensor, so callers pass the number of real
    forward tokens for this request.  Rows after that count are never read or
    included in the digest/counts.
    """
    if not isinstance(num_experts, int) or isinstance(num_experts, bool) or num_experts <= 0:
        raise RoutingTelemetryError("num_experts must be a positive integer")
    if not isinstance(valid_tokens, int) or isinstance(valid_tokens, bool) or valid_tokens <= 0:
        raise RoutingTelemetryError("valid_tokens must be a positive integer")
    rows = _tolist(routing_data)
    if len(rows) < valid_tokens:
        raise RoutingTelemetryError(
            f"routing returned {len(rows)} rows, fewer than {valid_tokens} real forward tokens"
        )
    selected = rows[:valid_tokens]
    if not selected or not isinstance(selected[0], list) or not selected[0]:
        raise RoutingTelemetryError("routing data has no layer dimension")
    layers = len(selected[0])
    if not isinstance(selected[0][0], list) or not selected[0][0]:
        raise RoutingTelemetryError("routing data has no experts-per-token dimension")
    top_k = len(selected[0][0])
    counts = [[0 for _ in range(num_experts)] for _ in range(layers)]
    for token_index, token_layers in enumerate(selected):
        if not isinstance(token_layers, list) or len(token_layers) != layers:
            raise RoutingTelemetryError(f"token {token_index} has inconsistent layer count")
        for layer_index, ids in enumerate(token_layers):
            if not isinstance(ids, list) or len(ids) != top_k:
                raise RoutingTelemetryError(
                    f"token {token_index}, layer {layer_index} has inconsistent top-k"
                )
            for expert_id in ids:
                if not isinstance(expert_id, int) or isinstance(expert_id, bool):
                    raise RoutingTelemetryError("expert IDs must be integers")
                if expert_id < 0 or expert_id >= num_experts:
                    raise RoutingTelemetryError(
                        f"expert ID {expert_id} is outside [0, {num_experts})"
                    )
                counts[layer_index][expert_id] += 1
    return {
        "schema": SCHEMA,
        "source": "vllm_enable_return_routed_experts_actual_selected_ids",
        "reported_tokens": len(rows),
        "valid_tokens": valid_tokens,
        "capture_padding_trimmed": len(rows) - valid_tokens,
        "num_layers": layers,
        "experts_per_token": top_k,
        "selected_id_count": valid_tokens * layers * top_k,
        "per_layer_count_sums": [sum(layer_counts) for layer_counts in counts],
        "per_layer_distinct_experts": [
            sum(value > 0 for value in layer_counts) for layer_counts in counts
        ],
        "per_layer_expert_counts": counts,
        "selected_ids": selected,
        "selected_ids_sha256": hashlib.sha256(canonical_json(selected)).hexdigest(),
    }



def hook_evidence(records: Sequence[dict[str, Any]], *, require_hashes: bool) -> dict[str, Any]:
    """Normalise immutable shape-hook rows into comparable execution evidence.

    The hook records dispatch after vLLM resolves compilation and graph mode.
    We retain each non-warmup request row's actual consumed token IDs plus
    hidden/logit hashes.  Request IDs and warmup rows are intentionally omitted:
    they are process-local and cannot establish equivalence between engine
    instances.
    """
    forwards = [record for record in records if record.get("event") == "forward"]
    if not forwards:
        raise RoutingTelemetryError("shape hook produced no forward records")
    resolved = sorted({
        (str(record.get("resolved_compile")), str(record.get("resolved_cudagraph")))
        for record in forwards
    })
    dispatches = [record.get("dispatch") for record in forwards]
    if any(not isinstance(dispatch, dict) for dispatch in dispatches):
        raise RoutingTelemetryError("shape hook forward record lacks resolved dispatch")
    rows: list[dict[str, Any]] = []
    invalid: list[str] = []
    for record in forwards:
        for request in record.get("requests", []):
            if not isinstance(request, dict) or str(request.get("req", "")).startswith("_warmup"):
                continue
            if request.get("invalid") is not None:
                invalid.append(str(request["invalid"]))
                continue
            item = {
                "prompt_sha256": request.get("prompt_sha256"),
                "q": request.get("q"),
                "computed": request.get("computed"),
                "phase": request.get("phase"),
                "new_token_ids": request.get("new_token_ids"),
                "h": request.get("h"),
                "logits_h": request.get("logits_h"),
            }
            if not isinstance(item["prompt_sha256"], str) or not isinstance(item["new_token_ids"], list):
                invalid.append("missing prompt digest or consumed token IDs")
                continue
            if require_hashes and (not isinstance(item["h"], str) or not isinstance(item["logits_h"], str)):
                invalid.append("missing hidden-state or logits hash")
                continue
            rows.append(item)
    if invalid:
        raise RoutingTelemetryError("invalid shape-hook rows: " + "; ".join(invalid[:3]))
    if not rows:
        raise RoutingTelemetryError("shape hook produced no non-warmup request rows")
    graph_modes = sorted({str(dispatch.get("cudagraph_mode")) for dispatch in dispatches})
    return {
        "forward_records": len(forwards),
        "resolved_compile_cudagraph": [list(value) for value in resolved],
        "dispatch_modes": graph_modes,
        "rows": rows,
        "rows_sha256": hashlib.sha256(canonical_json(rows)).hexdigest(),
    }


def compare_hook_hashes(
    baseline: dict[str, Any], observed: dict[str, Any]
) -> dict[str, Any]:
    """Compare matched graph executions using hook input and tensor hashes."""
    if not isinstance(baseline, dict) or not isinstance(observed, dict):
        raise RoutingTelemetryError("both conditions need shape-hook evidence")
    left, right = baseline.get("rows"), observed.get("rows")
    if not isinstance(left, list) or not isinstance(right, list):
        raise RoutingTelemetryError("shape-hook evidence lacks rows")
    return {
        "same_row_count": len(left) == len(right),
        "same_rows_sha256": baseline.get("rows_sha256") == observed.get("rows_sha256"),
        "all_equal": len(left) == len(right) and baseline.get("rows_sha256") == observed.get("rows_sha256"),
    }


def validate_graph_dispatch(
    evidence: dict[str, Any], *, expect_graph: bool, expect_compile: str
) -> dict[str, Any]:
    """Fail closed unless hook dispatch proves the requested execution regime.

    ``FULL`` is required for the graph condition because the diagnostic uses
    several decode tokens; that is the replay evidence, rather than merely a
    graph-capable resolved configuration or a piecewise prefill dispatch.
    """
    modes = evidence.get("dispatch_modes") if isinstance(evidence, dict) else None
    if not isinstance(modes, list) or not modes:
        raise RoutingTelemetryError("shape-hook evidence has no dispatch modes")
    resolved = evidence.get("resolved_compile_cudagraph")
    if not isinstance(resolved, list) or not resolved:
        raise RoutingTelemetryError("shape-hook evidence has no resolved configuration")
    compile_modes = sorted({str(item[0]) for item in resolved if isinstance(item, list) and len(item) == 2})
    full_decode_replay = "FULL" in modes
    all_eager = all(mode == "NONE" for mode in modes)
    compile_matches = compile_modes == [expect_compile]
    valid = (full_decode_replay if expect_graph else all_eager) and compile_matches
    return {
        "dispatch_modes": modes,
        "resolved_compile_modes": compile_modes,
        "expected_compile_mode": expect_compile,
        "compile_matches": compile_matches,
        "expect_graph": expect_graph,
        "full_decode_replay": full_decode_replay,
        "all_eager": all_eager,
        "valid": valid,
    }

def _indexed(rows: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = row.get("prompt_index")
        if not isinstance(index, int) or isinstance(index, bool) or index in out:
            raise RoutingTelemetryError("samples need unique integer prompt_index values")
        out[index] = row
    return out


def compare_matched_samples(
    baseline: Sequence[dict[str, Any]],
    observed: Sequence[dict[str, Any]],
    *,
    compare_routing: bool,
) -> dict[str, Any]:
    """Compare same prompts across modes without declaring equality by fiat."""
    left, right = _indexed(baseline), _indexed(observed)
    if set(left) != set(right):
        raise RoutingTelemetryError("conditions do not cover the same prompt indexes")
    per_prompt = []
    for index in sorted(left):
        a, b = left[index], right[index]
        if a.get("prompt_token_ids_sha256") != b.get("prompt_token_ids_sha256"):
            raise RoutingTelemetryError(f"prompt token IDs differ for prompt {index}")
        comparison = {
            "prompt_index": index,
            "output_tokens_equal": a.get("output_token_ids") == b.get("output_token_ids"),
            "output_token_ids_sha256_equal": a.get("output_token_ids_sha256")
            == b.get("output_token_ids_sha256"),
        }
        if compare_routing:
            a_routing = a.get("routing")
            b_routing = b.get("routing")
            if not isinstance(a_routing, dict) or not isinstance(b_routing, dict):
                raise RoutingTelemetryError("routing comparison requires actual observations on both conditions")
            comparison.update(
                {
                    "selected_ids_sha256_equal": a_routing.get("selected_ids_sha256")
                    == b_routing.get("selected_ids_sha256"),
                    "per_layer_counts_equal": a_routing.get("per_layer_expert_counts")
                    == b_routing.get("per_layer_expert_counts"),
                }
            )
        per_prompt.append(comparison)
    return {"all_equal": all(all(value for key, value in row.items() if key != "prompt_index") for row in per_prompt), "per_prompt": per_prompt}


def validate_dynamic_inputs(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Require two distinct prompts to yield distinct output and routing evidence."""
    indexed = _indexed(rows)
    if len(indexed) < 2:
        raise RoutingTelemetryError("at least two dynamic prompts are required")
    values = [indexed[key] for key in sorted(indexed)]
    if any(not isinstance(row.get("routing"), dict) for row in values):
        raise RoutingTelemetryError("dynamic validation requires actual routing observations")
    output_hashes = {row.get("output_token_ids_sha256") for row in values}
    route_hashes = {row["routing"].get("selected_ids_sha256") for row in values}
    count_signatures = {
        hashlib.sha256(canonical_json(row["routing"].get("per_layer_expert_counts"))).hexdigest()
        for row in values
    }
    overlap_route_differences = 0
    decode_route_variants: dict[int, int] = {}
    decode_errors: list[str] = []
    for row in values:
        prompt_count = row.get("prompt_token_count")
        generated_count = row.get("generated_token_count")
        selected_ids = row["routing"].get("selected_ids")
        index = row["prompt_index"]
        if (
            not isinstance(prompt_count, int)
            or isinstance(prompt_count, bool)
            or not isinstance(generated_count, int)
            or isinstance(generated_count, bool)
            or not isinstance(selected_ids, list)
            or len(selected_ids) != prompt_count + generated_count - 1
        ):
            decode_errors.append(f"prompt {index}: routing length does not match prompt+decode")
            continue
        decode_rows = selected_ids[prompt_count:]
        variants = len({json.dumps(item, sort_keys=True) for item in decode_rows})
        decode_route_variants[index] = variants
        if len(decode_rows) < 2 or variants < 2:
            decode_errors.append(
                f"prompt {index}: decode graph replays do not contain two distinct routing rows"
            )
    for left, right in zip(values, values[1:]):
        left_ids = left["routing"].get("selected_ids")
        right_ids = right["routing"].get("selected_ids")
        if not isinstance(left_ids, list) or not isinstance(right_ids, list):
            raise RoutingTelemetryError("dynamic validation requires retained selected IDs")
        overlap_route_differences += sum(a != b for a, b in zip(left_ids, right_ids))
    return {
        "distinct_output_token_hashes": len(output_hashes),
        "distinct_selected_id_hashes": len(route_hashes),
        "distinct_count_vectors": len(count_signatures),
        "overlap_route_differences": overlap_route_differences,
        "decode_route_variants": decode_route_variants,
        "decode_errors": decode_errors,
        "valid": (
            len(output_hashes) > 1
            and len(route_hashes) > 1
            and len(count_signatures) > 1
            and overlap_route_differences > 0
            and not decode_errors
        ),
    }
