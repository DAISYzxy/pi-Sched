#!/usr/bin/env python3
"""
Compile dependency constraints into minimum-latency completion times and evaluate.

Default run:
  python src/execution_evaluate.py run

Evaluate an existing prediction file:
  python src/execution_evaluate.py evaluate --pred results/predictions.jsonl

Input dependency JSON is expected to be the output of acyclic_dependency_build.py.
Each benchmark request
must contain a positive integer ``time`` duration.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DEPENDENCY = SCRIPT_DIR.parent / "results" / "dependency.json"
DEFAULT_BENCHMARK = SCRIPT_DIR.parent / "data" / "sgd_micro_10_full.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "results" / "schedule.json"
DEFAULT_PRED_OUTPUT = SCRIPT_DIR.parent / "results" / "predictions.jsonl"
DEFAULT_EVAL_OUTPUT = SCRIPT_DIR.parent / "results" / "evaluation.json"


def infer_request_seq(agent_id: str) -> int:
    """
    Fallback: agent_000 -> 1, agent_001 -> 2, ...

    Prefer benchmark request_seq when a benchmark case is available.
    """
    match = re.search(r"(\d+)$", agent_id)
    if not match:
        raise ValueError(f"Cannot infer request_seq from agent_id: {agent_id}")
    return int(match.group(1)) + 1


def get_request_seq(agent_id: str, request_seq_map: Optional[dict[str, int]] = None) -> int:
    if request_seq_map and agent_id in request_seq_map:
        return int(request_seq_map[agent_id])
    return infer_request_seq(agent_id)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def dependency_cases(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return [case for case in data["cases"] if isinstance(case, dict)]
    if isinstance(data, dict) and isinstance(data.get("agents"), list):
        return [data]
    if isinstance(data, list):
        return [case for case in data if isinstance(case, dict)]
    raise ValueError("dependency input must be a case, a list of cases, or an object with a cases array")


def benchmark_requests(case: dict[str, Any]) -> list[dict[str, Any]]:
    public_input = case.get("public_input", {}) if isinstance(case.get("public_input"), dict) else {}
    hidden_input = case.get("input", {}) if isinstance(case.get("input"), dict) else {}
    requests = public_input.get("agent_requests") or hidden_input.get("agent_requests") or []
    return [req for req in requests if isinstance(req, dict)]


def request_seq_map_for_benchmark_case(case: Optional[dict[str, Any]]) -> dict[str, int]:
    if not case:
        return {}
    out: dict[str, int] = {}
    for req in benchmark_requests(case):
        agent_id = str(req.get("agent_id", ""))
        if not agent_id:
            continue
        try:
            out[agent_id] = int(req.get("request_seq", infer_request_seq(agent_id)))
        except (TypeError, ValueError):
            out[agent_id] = infer_request_seq(agent_id)
    return out


def node_weights_for_benchmark_case(case: Optional[dict[str, Any]]) -> dict[str, int]:
    """Return positive integer task durations keyed by agent id."""
    if not case:
        raise ValueError("benchmark case is required to load node time weights")

    weights: dict[str, int] = {}
    for req in benchmark_requests(case):
        agent_id = str(req.get("agent_id", "")).strip()
        if not agent_id:
            continue
        if agent_id in weights:
            raise ValueError(f"duplicate agent_id in benchmark requests: {agent_id}")
        value = req.get("time")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"benchmark request {agent_id} must have a positive integer time")
        weights[agent_id] = value

    if not weights:
        raise ValueError("benchmark case has no weighted agent requests")
    return weights


def load_benchmark_cases(path: Optional[Path]) -> list[dict[str, Any]]:
    if path is None:
        return []
    if not path.exists():
        return []
    return read_jsonl(path)


def benchmark_by_case_id(cases: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(case.get("case_id")): case for case in cases}


def edge_endpoint(edge: dict[str, Any], key: str) -> str:
    return str(edge.get(key, "")).strip()


def normalized_pair_records(case: dict[str, Any], field: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in case.get(field, []) or []:
        agents = item.get("agents", []) if isinstance(item, dict) else item
        if not isinstance(agents, (list, tuple)) or len(agents) < 2:
            continue
        left = str(agents[0]).strip()
        right = str(agents[1]).strip()
        if not left or not right or left == right:
            continue
        reasons = item.get("reasons", []) if isinstance(item, dict) else []
        records.append(
            {
                "field": field,
                "left": left,
                "right": right,
                "reasons": [str(reason) for reason in reasons] if isinstance(reasons, list) else [],
            }
        )
    return records


def agent_execution_rank(
    agent_id: str,
    request_seq_map: Optional[dict[str, int]] = None,
    agent_position: Optional[dict[str, int]] = None,
) -> int:
    if request_seq_map and agent_id in request_seq_map:
        return int(request_seq_map[agent_id])
    try:
        return infer_request_seq(agent_id)
    except ValueError:
        if agent_position and agent_id in agent_position:
            return int(agent_position[agent_id]) + 1
        raise


def schedule_sort_key(
    agent_id: str,
    request_seq_map: Optional[dict[str, int]],
    agent_position: dict[str, int],
) -> tuple[int, int, str]:
    return (agent_execution_rank(agent_id, request_seq_map, agent_position), agent_position.get(agent_id, 10**9), agent_id)


def topological_order(
    agents: list[str],
    preds: dict[str, set[str]],
    succs: dict[str, set[str]],
    request_seq_map: Optional[dict[str, int]],
    agent_position: dict[str, int],
) -> list[str]:
    indegree = {agent: len(preds.get(agent, set())) for agent in agents}
    ready = sorted(
        [agent for agent in agents if indegree.get(agent, 0) == 0],
        key=lambda agent: schedule_sort_key(agent, request_seq_map, agent_position),
    )
    ordered: list[str] = []

    while ready:
        agent = ready.pop(0)
        ordered.append(agent)
        for succ in sorted(succs.get(agent, set()), key=lambda item: schedule_sort_key(item, request_seq_map, agent_position)):
            indegree[succ] -= 1
            if indegree[succ] == 0:
                ready.append(succ)
        ready.sort(key=lambda item: schedule_sort_key(item, request_seq_map, agent_position))

    if len(ordered) != len(agents):
        raise ValueError("mandatory_edges contain a directed cycle")
    return ordered


def longest_path_weight(
    agents: list[str],
    succs: dict[str, set[str]],
    topo_order: list[str],
    node_weights: dict[str, int],
) -> int:
    if not agents:
        return 0
    distance = {agent: node_weights[agent] for agent in agents}
    for agent in topo_order:
        for succ in succs.get(agent, set()):
            distance[succ] = max(distance[succ], distance[agent] + node_weights[succ])
    return max(distance.values())


def longest_path_length(agents: list[str], succs: dict[str, set[str]], topo_order: list[str]) -> int:
    """Unweighted compatibility helper used by analysis scripts."""
    return longest_path_weight(agents, succs, topo_order, {agent: 1 for agent in agents})


def completion_time_bounds_for_latency(
    agents: list[str],
    preds: dict[str, set[str]],
    succs: dict[str, set[str]],
    topo_order: list[str],
    node_weights: dict[str, int],
    latency: int,
) -> tuple[dict[str, int], dict[str, int]]:
    earliest = {agent: node_weights[agent] for agent in agents}
    for agent in topo_order:
        for succ in succs.get(agent, set()):
            earliest[succ] = max(earliest[succ], earliest[agent] + node_weights[succ])

    latest = {agent: latency for agent in agents}
    for agent in reversed(topo_order):
        for pred in preds.get(agent, set()):
            latest[pred] = min(latest[pred], latest[agent] - node_weights[agent])
    return earliest, latest


def solve_completion_times_for_latency(
    agents: list[str],
    preds: dict[str, set[str]],
    succs: dict[str, set[str]],
    conflict_pairs: set[frozenset[str]],
    topo_order: list[str],
    request_seq_map: Optional[dict[str, int]],
    agent_position: dict[str, int],
    node_weights: dict[str, int],
    latency: int,
) -> Optional[dict[str, int]]:
    earliest, latest = completion_time_bounds_for_latency(
        agents,
        preds,
        succs,
        topo_order,
        node_weights,
        latency,
    )
    if any(earliest[agent] > latest[agent] for agent in agents):
        return None

    conflict_adj: dict[str, set[str]] = {agent: set() for agent in agents}
    for pair in conflict_pairs:
        if len(pair) != 2:
            continue
        left, right = tuple(pair)
        conflict_adj.setdefault(left, set()).add(right)
        conflict_adj.setdefault(right, set()).add(left)

    ordered_agents = sorted(agents, key=lambda agent: schedule_sort_key(agent, request_seq_map, agent_position))
    assignment: dict[str, int] = {}
    unassigned = set(agents)

    def domain(agent: str) -> list[int]:
        lower = earliest[agent]
        upper = latest[agent]
        for pred in preds.get(agent, set()):
            if pred in assignment:
                lower = max(lower, assignment[pred] + node_weights[agent])
        for succ in succs.get(agent, set()):
            if succ in assignment:
                upper = min(upper, assignment[succ] - node_weights[succ])
        if lower > upper:
            return []

        duration = node_weights[agent]

        def is_compatible(completion_time: int) -> bool:
            start_time = completion_time - duration
            for other in conflict_adj.get(agent, set()):
                if other not in assignment:
                    continue
                other_completion = assignment[other]
                other_start = other_completion - node_weights[other]
                if not (completion_time <= other_start or other_completion <= start_time):
                    return False
            return True

        return [
            completion_time
            for completion_time in range(lower, upper + 1)
            if is_compatible(completion_time)
        ]

    def choose_agent() -> tuple[Optional[str], Optional[list[int]]]:
        domains: dict[str, list[int]] = {}
        for agent in ordered_agents:
            if agent not in unassigned:
                continue
            values = domain(agent)
            if not values:
                return agent, values
            domains[agent] = values

        forced = [agent for agent, values in domains.items() if len(values) == 1]
        if forced:
            agent = forced[0]
            return agent, domains[agent]

        for agent in ordered_agents:
            if agent in unassigned:
                return agent, domains[agent]
        return None, None

    def has_forward_domain() -> bool:
        return all(domain(agent) for agent in unassigned)

    def backtrack() -> bool:
        if not unassigned:
            return True

        agent, values = choose_agent()
        if agent is None or values is None:
            return True
        if not values:
            return False

        unassigned.remove(agent)
        for value in values:
            assignment[agent] = value
            if has_forward_domain() and backtrack():
                return True
            assignment.pop(agent, None)
        unassigned.add(agent)
        return False

    if not backtrack():
        return None

    return dict(assignment)


def solve_min_latency_schedule(
    agents: list[str],
    preds: dict[str, set[str]],
    succs: dict[str, set[str]],
    conflict_pairs: set[frozenset[str]],
    request_seq_map: Optional[dict[str, int]],
    agent_position: dict[str, int],
    node_weights: dict[str, int],
) -> tuple[int, dict[str, int]]:
    topo = topological_order(agents, preds, succs, request_seq_map, agent_position)
    if not agents:
        return 0, {}

    missing_weights = sorted(set(agents) - set(node_weights))
    if missing_weights:
        raise ValueError(f"missing time weights for agents: {missing_weights}")
    invalid_weights = {
        agent: node_weights[agent]
        for agent in agents
        if isinstance(node_weights[agent], bool)
        or not isinstance(node_weights[agent], int)
        or node_weights[agent] <= 0
    }
    if invalid_weights:
        raise ValueError(f"node time weights must be positive integers: {invalid_weights}")

    lower_bound = longest_path_weight(agents, succs, topo, node_weights)
    upper_bound = sum(node_weights[agent] for agent in agents)
    for latency in range(lower_bound, upper_bound + 1):
        assignment = solve_completion_times_for_latency(
            agents,
            preds,
            succs,
            conflict_pairs,
            topo,
            request_seq_map,
            agent_position,
            node_weights,
            latency,
        )
        if assignment is not None:
            return latency, assignment

    raise ValueError("No feasible weighted schedule satisfies mandatory and non-parallel constraints")


def oriented_pair_edges(
    pair_records: list[dict[str, Any]],
    completion_time: dict[str, int],
    request_seq_map: Optional[dict[str, int]],
    agent_position: dict[str, int],
) -> list[dict[str, Any]]:
    edges: dict[tuple[str, str, str], set[str]] = {}
    for record in pair_records:
        left = record["left"]
        right = record["right"]
        if completion_time[left] < completion_time[right]:
            before, after = left, right
        elif completion_time[right] < completion_time[left]:
            before, after = right, left
        else:
            raise ValueError(f"non-parallel agents have equal completion times: {left}, {right}")
        key = (before, after, record["field"])
        edges.setdefault(key, set()).update(record.get("reasons", []))

    return [
        {"before": before, "after": after, "source": source, "reasons": sorted(reasons)}
        for (before, after, source), reasons in sorted(
            edges.items(),
            key=lambda item: (
                schedule_sort_key(item[0][0], request_seq_map, agent_position),
                schedule_sort_key(item[0][1], request_seq_map, agent_position),
                item[0][2],
            ),
        )
    ]


def final_dependency_edges(
    mandatory_edges: list[dict[str, Any]],
    oriented_edges: list[dict[str, Any]],
    request_seq_map: Optional[dict[str, int]],
    agent_position: dict[str, int],
) -> list[dict[str, Any]]:
    edge_reasons: dict[tuple[str, str], set[str]] = {}
    for edge in mandatory_edges:
        before = edge_endpoint(edge, "before")
        after = edge_endpoint(edge, "after")
        if not before or not after or before == after:
            continue
        reasons = edge.get("reasons", []) if isinstance(edge, dict) else []
        edge_reasons.setdefault((before, after), set()).update(str(reason) for reason in reasons)

    for edge in oriented_edges:
        before = edge_endpoint(edge, "before")
        after = edge_endpoint(edge, "after")
        if not before or not after or before == after:
            continue
        source = str(edge.get("source", "oriented_pair"))
        reasons = edge.get("reasons", []) if isinstance(edge, dict) else []
        if reasons:
            edge_reasons.setdefault((before, after), set()).update(str(reason) for reason in reasons)
        else:
            edge_reasons.setdefault((before, after), set()).add(f"oriented {source}")

    return [
        {"before": before, "after": after, "reasons": sorted(reasons)}
        for (before, after), reasons in sorted(
            edge_reasons.items(),
            key=lambda item: (
                schedule_sort_key(item[0][0], request_seq_map, agent_position),
                schedule_sort_key(item[0][1], request_seq_map, agent_position),
                item[0],
            ),
        )
    ]


def dependency_case_to_completion_time(
    case: dict[str, Any],
    node_weights: dict[str, int],
    request_seq_map: Optional[dict[str, int]] = None,
    *,
    respect_mutex: bool = True,
) -> dict[str, Any]:
    agents = [str(agent).strip() for agent in case.get("agents", []) if str(agent).strip()]
    agent_position = {agent: index for index, agent in enumerate(agents)}

    def append_agent(agent: str) -> None:
        if agent and agent not in agent_position:
            agent_position[agent] = len(agents)
            agents.append(agent)

    mandatory_edges: list[dict[str, Any]] = []
    for edge in case.get("mandatory_edges", []) or []:
        if not isinstance(edge, dict):
            continue
        before = edge_endpoint(edge, "before")
        after = edge_endpoint(edge, "after")
        if not before or not after or before == after:
            continue

        append_agent(before)
        append_agent(after)
        mandatory_edges.append(edge)

    pair_records: list[dict[str, Any]] = []
    if respect_mutex:
        pair_records.extend(normalized_pair_records(case, "mutex_pairs"))
        pair_records.extend(normalized_pair_records(case, "dedup_required_pairs"))
        for record in pair_records:
            append_agent(record["left"])
            append_agent(record["right"])

    # After all endpoints are known, make non-request_seq fallback canonical by
    # agent id instead of depending on dependency JSON input order.
    agent_position = {agent: index for index, agent in enumerate(sorted(agents))}

    preds: dict[str, set[str]] = {agent: set() for agent in agents}
    succs: dict[str, set[str]] = {agent: set() for agent in agents}
    for edge in mandatory_edges:
        before = edge_endpoint(edge, "before")
        after = edge_endpoint(edge, "after")
        preds[after].add(before)
        succs[before].add(after)

    conflict_pairs = (
        {
            frozenset((record["left"], record["right"]))
            for record in pair_records
            if record["left"] != record["right"]
        }
        if respect_mutex
        else set()
    )

    latency, completion_time = solve_min_latency_schedule(
        agents,
        preds,
        succs,
        conflict_pairs,
        request_seq_map,
        agent_position,
        node_weights,
    )

    oriented_edges = (
        oriented_pair_edges(pair_records, completion_time, request_seq_map, agent_position)
        if respect_mutex
        else []
    )

    return {
        "case_id": case.get("case_id"),
        "completion_time": completion_time,
        "latency": latency,
        "oriented_mutex_dedup_edges": oriented_edges,
        "final_dependency_edges": final_dependency_edges(
            mandatory_edges,
            oriented_edges,
            request_seq_map,
            agent_position,
        ),
    }


def validate_completion_schedule(
    dependency_case: dict[str, Any],
    benchmark_case: Optional[dict[str, Any]],
    prediction: dict[str, Any],
    *,
    respect_mutex: bool = True,
) -> dict[str, Any]:
    """Validate weighted precedence and non-overlap constraints."""
    errors: list[str] = []
    try:
        node_weights = node_weights_for_benchmark_case(benchmark_case)
    except ValueError as exc:
        return {"valid": False, "latency": 0, "errors": [str(exc)]}

    agents = {str(agent).strip() for agent in dependency_case.get("agents", []) if str(agent).strip()}
    mandatory_edges: list[tuple[str, str]] = []
    for edge in dependency_case.get("mandatory_edges", []) or []:
        if not isinstance(edge, dict):
            continue
        before = edge_endpoint(edge, "before")
        after = edge_endpoint(edge, "after")
        if before and after and before != after:
            agents.update((before, after))
            mandatory_edges.append((before, after))

    pair_records: list[dict[str, Any]] = []
    if respect_mutex:
        pair_records.extend(normalized_pair_records(dependency_case, "mutex_pairs"))
        pair_records.extend(normalized_pair_records(dependency_case, "dedup_required_pairs"))
        for record in pair_records:
            agents.update((record["left"], record["right"]))

    raw_completion = prediction.get("completion_time")
    if not isinstance(raw_completion, dict):
        return {"valid": False, "latency": 0, "errors": ["completion_time must be a JSON object"]}

    completion_time: dict[str, int] = {}
    for agent, value in raw_completion.items():
        agent_id = str(agent)
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"completion_time[{agent_id}] must be an integer")
            continue
        completion_time[agent_id] = value

    missing_agents = sorted(agents - set(completion_time))
    extra_agents = sorted(set(completion_time) - agents)
    if missing_agents:
        errors.append(f"missing completion times for agents: {missing_agents}")
    if extra_agents:
        errors.append(f"unknown agents in completion_time: {extra_agents}")

    missing_weights = sorted(agents - set(node_weights))
    if missing_weights:
        errors.append(f"missing time weights for agents: {missing_weights}")

    if errors:
        return {"valid": False, "latency": 0, "errors": errors}

    for agent in sorted(agents):
        if completion_time[agent] < node_weights[agent]:
            errors.append(
                f"{agent} completes at {completion_time[agent]} before its duration {node_weights[agent]}"
            )

    for before, after in mandatory_edges:
        required = completion_time[before] + node_weights[after]
        if completion_time[after] < required:
            errors.append(
                f"mandatory edge {before}->{after} requires completion_time[{after}] >= {required}"
            )

    seen_pairs: set[frozenset[str]] = set()
    for record in pair_records:
        left = record["left"]
        right = record["right"]
        pair = frozenset((left, right))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        left_start = completion_time[left] - node_weights[left]
        right_start = completion_time[right] - node_weights[right]
        if not (
            completion_time[left] <= right_start
            or completion_time[right] <= left_start
        ):
            errors.append(f"non-parallel agents overlap: {left}, {right}")

    latency = max(completion_time.values(), default=0)
    reported_latency = prediction.get("latency")
    if reported_latency is not None and reported_latency != latency:
        errors.append(f"reported latency {reported_latency} does not equal max completion time {latency}")
    return {
        "valid": not errors,
        "latency": latency,
        "num_agents": len(agents),
        "errors": errors,
    }


def completion_accuracy_for_case(
    benchmark_case: Optional[dict[str, Any]],
    prediction: dict[str, Any],
    *,
    schedule_valid: bool,
) -> dict[str, Any]:
    """Evaluate completion times using the benchmark's existing edge/case semantics."""
    if not benchmark_case:
        return {"edge_correct": 0, "edge_total": 0, "case_correct": False}

    try:
        node_weights = node_weights_for_benchmark_case(benchmark_case)
    except ValueError:
        return {"edge_correct": 0, "edge_total": 0, "case_correct": False}

    raw_completion = prediction.get("completion_time")
    if not isinstance(raw_completion, dict):
        return {"edge_correct": 0, "edge_total": 0, "case_correct": False}
    completion_time = {
        str(agent): value
        for agent, value in raw_completion.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    format_valid = set(completion_time) == set(node_weights)

    oracle = benchmark_case.get("oracle", {})
    if not isinstance(oracle, dict):
        oracle = {}
    raw_edges = oracle.get("dependency_edges", []) or []
    edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for edge in raw_edges:
        if not isinstance(edge, dict):
            continue
        before = edge_endpoint(edge, "before")
        after = edge_endpoint(edge, "after")
        key = (before, after)
        if (
            not before
            or not after
            or before == after
            or before not in node_weights
            or after not in node_weights
            or key in seen
        ):
            continue
        seen.add(key)
        edges.append(key)

    edge_correct = 0
    for before, after in edges:
        if before not in completion_time or after not in completion_time:
            continue
        after_start = completion_time[after] - node_weights[after]
        if completion_time[before] <= after_start:
            edge_correct += 1

    return {
        "edge_correct": edge_correct,
        "edge_total": len(edges),
        "case_correct": schedule_valid and format_valid and edge_correct == len(edges),
    }


def aggregate_metrics(
    accuracy_checks: list[dict[str, Any]],
    schedule_checks: list[dict[str, Any]],
    runtimes_ms: list[float],
) -> dict[str, float]:
    edge_correct = sum(int(check.get("edge_correct", 0)) for check in accuracy_checks)
    edge_total = sum(int(check.get("edge_total", 0)) for check in accuracy_checks)
    case_total = len(accuracy_checks)
    case_correct = sum(int(bool(check.get("case_correct"))) for check in accuracy_checks)
    latencies = [
        int(check.get("latency", 0))
        for check in schedule_checks
        if check.get("valid")
    ]
    return {
        "edge_accuracy": edge_correct / edge_total if edge_total else 1.0,
        "case_accuracy": case_correct / case_total if case_total else 0.0,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0.0,
        "avg_running_time_ms": sum(runtimes_ms) / len(runtimes_ms) if runtimes_ms else 0.0,
    }


def build_predictions(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    dependency_path = Path(args.dependency)
    dependency_data = read_json(dependency_path)
    dep_cases = dependency_cases(dependency_data)
    if args.limit and args.limit > 0:
        dep_cases = dep_cases[: args.limit]

    benchmark_path = Path(args.benchmark) if args.benchmark else None
    benchmark_cases = load_benchmark_cases(benchmark_path)
    benchmark_cases_by_id = benchmark_by_case_id(benchmark_cases)

    results: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    accuracy_checks: list[dict[str, Any]] = []
    schedule_checks: list[dict[str, Any]] = []
    runtimes_ms: list[float] = []
    progress_interval = max(0, int(getattr(args, "progress_interval", 0) or 0))
    for index, dep_case in enumerate(dep_cases, start=1):
        case_id = str(dep_case.get("case_id"))
        benchmark_case = benchmark_cases_by_id.get(case_id)
        request_seq_map = request_seq_map_for_benchmark_case(benchmark_case)
        case_start = time.perf_counter()
        errors: list[str] = []
        try:
            node_weights = node_weights_for_benchmark_case(benchmark_case)
            schedule = dependency_case_to_completion_time(
                dep_case,
                node_weights,
                request_seq_map=request_seq_map,
                respect_mutex=not args.ignore_mutex,
            )
            prediction = {
                "case_id": schedule.get("case_id"),
                "completion_time": schedule.get("completion_time", {}),
            }
            schedule_check = validate_completion_schedule(
                dep_case,
                benchmark_case,
                schedule,
                respect_mutex=not args.ignore_mutex,
            )
            if not schedule_check.get("valid"):
                errors.extend(str(error) for error in schedule_check.get("errors", []))
        except Exception as exc:
            errors.append(str(exc))
            schedule = {
                "case_id": dep_case.get("case_id"),
                "completion_time": {},
                "latency": 0,
                "oriented_mutex_dedup_edges": [],
                "final_dependency_edges": [],
            }
            prediction = {"case_id": dep_case.get("case_id"), "completion_time": {}}
            schedule_check = {
                "valid": False,
                "latency": 0,
                "errors": list(errors),
            }

        accuracy_check = completion_accuracy_for_case(
            benchmark_case,
            prediction,
            schedule_valid=bool(schedule_check.get("valid")),
        )
        runtime_ms = (time.perf_counter() - case_start) * 1000
        result = {
            "case_id": dep_case.get("case_id"),
            "prediction": prediction,
            "schedule_summary": {
                "latency": schedule.get("latency", 0),
                "oriented_mutex_dedup_edges": schedule.get("oriented_mutex_dedup_edges", []),
                "final_dependency_edges": schedule.get("final_dependency_edges", []),
            },
            "dependency_summary": {
                "num_agents": len(dep_case.get("agents", [])),
                "num_mandatory_edges": len(dep_case.get("mandatory_edges", []) or []),
                "num_mutex_pairs": len(dep_case.get("mutex_pairs", []) or []),
                "num_dedup_required_pairs": len(dep_case.get("dedup_required_pairs", []) or []),
            },
            "schedule_check": schedule_check,
            "errors": errors,
            "runtime_ms": runtime_ms,
        }
        results.append(result)
        predictions.append(result["prediction"])
        accuracy_checks.append(accuracy_check)
        schedule_checks.append(schedule_check)
        runtimes_ms.append(runtime_ms)
        if progress_interval and index % progress_interval == 0:
            elapsed_s = time.perf_counter() - start
            avg_case_s = (
                sum(float(r.get("runtime_ms", 0.0)) for r in results) / len(results) / 1000
                if results
                else 0.0
            )
            print(
                "[execution_evaluate timing] "
                f"dependency={dependency_path} "
                f"cases={index}/{len(dep_cases)} "
                f"cumulative_runtime_s={elapsed_s:.3f} "
                f"avg_case_runtime_s={avg_case_s:.6f}",
                file=sys.stderr,
            )

    metrics = aggregate_metrics(accuracy_checks, schedule_checks, runtimes_ms)

    output = {
        "dependency_path": str(dependency_path),
        "benchmark_path": str(benchmark_path) if benchmark_path else None,
        "prediction_path": str(args.pred_output),
        "respect_mutex": not args.ignore_mutex,
        "metrics": metrics,
        "results": results,
    }

    write_json(Path(args.output), output)
    write_jsonl(Path(args.pred_output), predictions)
    if args.eval_output:
        write_json(Path(args.eval_output), metrics)

    return output


def evaluate_prediction_file(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_path = Path(args.benchmark)
    benchmark_cases = read_jsonl(benchmark_path)
    benchmark_cases_by_id = benchmark_by_case_id(benchmark_cases)
    dependency_data = read_json(Path(args.dependency))
    dep_cases = dependency_cases(dependency_data)
    predictions = read_jsonl(Path(args.pred))
    pred_by_id = {str(pred.get("case_id")): pred for pred in predictions}

    checks: list[dict[str, Any]] = []
    accuracy_checks: list[dict[str, Any]] = []
    runtimes_ms: list[float] = []
    for dep_case in dep_cases:
        case_start = time.perf_counter()
        case_id = str(dep_case.get("case_id"))
        benchmark_case = benchmark_cases_by_id.get(case_id)
        prediction = pred_by_id.get(
            case_id,
            {"case_id": dep_case.get("case_id"), "completion_time": {}},
        )
        check = validate_completion_schedule(
            dep_case,
            benchmark_case,
            prediction,
            respect_mutex=not args.ignore_mutex,
        )
        checks.append(check)
        accuracy_checks.append(
            completion_accuracy_for_case(
                benchmark_case,
                prediction,
                schedule_valid=bool(check.get("valid")),
            )
        )
        runtimes_ms.append((time.perf_counter() - case_start) * 1000)

    output = aggregate_metrics(accuracy_checks, checks, runtimes_ms)
    if args.output:
        write_json(Path(args.output), output)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile dependency JSON to minimum-latency completion times")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Build weighted completion times from dependency JSON")
    run.add_argument("--dependency", type=Path, default=DEFAULT_DEPENDENCY)
    run.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Detailed result JSON")
    run.add_argument("--pred-output", type=Path, default=DEFAULT_PRED_OUTPUT, help="Prediction JSONL")
    run.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT, help="Evaluation metrics JSON")
    run.add_argument("--ignore-mutex", action="store_true", help="Ignore mutex_pairs / dedup_required_pairs")
    run.add_argument("--limit", type=int, default=0, help="Number of dependency cases to run; 0 means all")
    run.add_argument("--progress-interval", type=int, default=0, help="Print cumulative timing every N cases; 0 disables")
    run.set_defaults(func=run_command)

    ev = sub.add_parser("evaluate", help="Validate completion-time predictions")
    ev.add_argument("--dependency", type=Path, default=DEFAULT_DEPENDENCY)
    ev.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ev.add_argument("--pred", type=Path, default=DEFAULT_PRED_OUTPUT)
    ev.add_argument("--output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    ev.add_argument("--ignore-mutex", action="store_true", help="Ignore mutex_pairs / dedup_required_pairs")
    ev.set_defaults(func=evaluate_command)
    return parser


def run_command(args: argparse.Namespace) -> None:
    output = build_predictions(args)
    metrics = output["metrics"]
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved results to: {args.output}")
    print(f"Saved predictions to: {args.pred_output}")
    if args.eval_output:
        print(f"Saved evaluation to: {args.eval_output}")


def evaluate_command(args: argparse.Namespace) -> None:
    output = evaluate_prediction_file(args)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.output:
        print(f"Saved evaluation to: {args.output}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
