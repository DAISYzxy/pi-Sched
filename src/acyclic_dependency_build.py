#!/usr/bin/env python3
"""Build dependency constraints and globally remove directed cycles.

Rule-based candidate edges are produced first, then the weighted
minimum-arc-set heuristic below selects a DAG:

1. Count every remaining simple directed cycle inside the cyclic SCCs.
2. Remove the edge minimizing ``weight / number_of_cycles_containing_edge``.
3. Re-add dropped edges from highest to lowest weight whenever that does not
   create a path from the edge's destination back to its source.

An edge's weight is the number of distinct dependency reasons supporting it.
This is available without changing the input/output format, and makes an edge
with several independent pieces of evidence more expensive to remove.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parent / "results" / "parsed.json"
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "results" / "dependency.json"

# Conflicts and duplicate idempotency scopes are oriented by an explicit
# request_seq when present. New public-visible benchmarks omit request_seq, so
# the deterministic fallback is canonical agent-id order, never input order.
ORIENT_CONFLICTS_BY_REQUEST_SEQ = True

LATEST_FRESHNESS_MARKERS = ("latest", "final", "confirmed", "settled")
CURRENT_FRESHNESS_MARKERS = ("current", "present", "before", "pre")
STATIC_FRESHNESS_MARKERS = ("static", "immutable")

VALIDATION_START_WORDS = (
    "verify",
    "check",
    "review",
    "validate",
    "determine",
    "confirm",
    "assess",
    "inspect",
    "evaluate",
)
VALIDATION_OUTPUT_MARKERS = (
    "validation",
    "valid",
    "eligibility",
    "eligible",
    "allowed",
    "permission",
    "approval",
    "authorized",
)

EXPLICIT_REF_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_-]*)\.([A-Za-z_][A-Za-z0-9_-]*)")
TRAILING_NUM_RE = re.compile(r"(\d+)$")
NOTE_ID_RE = re.compile(r"\bnote[_ -]?id\s*[:=#-]?\s*([A-Za-z0-9_-]+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class StateRead:
    raw: str
    key: str
    freshness: str


@dataclass(frozen=True)
class StateWrite:
    raw: str
    key: str
    mode: str
    value: str


@dataclass(frozen=True)
class Operation:
    agent_id: str
    description: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    reads: tuple[StateRead, ...]
    writes: tuple[StateWrite, ...]
    effect: str
    idempotent: bool
    index: int
    request_seq: int | None


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_token(value: Any, *, keep_dot: bool = False) -> str:
    """Canonicalize symbols for deterministic, non-LLM matching."""
    text = clean_text(value).lower()
    text = text.strip().strip("\"'`.,;()[]{}")
    text = text.replace("-", "_")
    text = re.sub(r"\s+", "_", text)

    allowed = r"a-z0-9_:*"
    if keep_dot:
        allowed += r"\."
    text = re.sub(fr"[^{allowed}]+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = re.sub(r":_+", ":", text)
    text = re.sub(r"_+:", ":", text)
    return text.strip("_")


def normalize_key(value: Any) -> str:
    """Canonical key for reads/writes and structured state references."""
    return normalize_token(value, keep_dot=False)


def normalize_symbol(value: Any) -> str:
    """Canonical input/output symbol. Dots are preserved for explicit refs."""
    return normalize_token(value, keep_dot=True)


def normalize_alias(value: Any) -> str:
    return normalize_token(value, keep_dot=False)


def normalize_value(value: Any) -> str:
    return normalize_token(value, keep_dot=False)


def as_symbol_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = normalize_symbol(value)
        return (text,) if text else ()
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            text = normalize_symbol(item)
            if text:
                out.append(text)
        return tuple(out)
    text = normalize_symbol(value)
    return (text,) if text else ()


def as_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    text = clean_text(value).lower()
    if text in {"true", "yes", "y", "1", "idempotent"}:
        return True
    if text in {"false", "no", "n", "0", "non-idempotent", "non_idempotent"}:
        return False
    return default


def infer_request_seq(raw: dict[str, Any], agent_id: str, index: int) -> int | None:
    """Return explicit public request_seq if present.

    Do not synthesize request_seq from public input order. When this returns
    None, conflict orientation uses canonical agent-id order.
    """
    value = raw.get("request_seq")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_read(value: Any) -> StateRead:
    if isinstance(value, dict):
        raw = clean_text(value.get("raw") or value.get("description") or value.get("key") or "")
        key = normalize_key(value.get("key") or raw)
        freshness = clean_text(value.get("freshness") or "current").lower()
        return StateRead(raw=raw or key, key=key, freshness=freshness)
    raw = clean_text(value)
    return StateRead(raw=raw, key=normalize_key(raw), freshness="current")


def normalize_write(value: Any) -> StateWrite:
    if isinstance(value, dict):
        raw = clean_text(value.get("raw") or value.get("description") or value.get("key") or "")
        key = normalize_key(value.get("key") or raw)
        mode = normalize_value(value.get("mode") or "set") or "set"
        write_value = normalize_value(value.get("value", ""))
        return StateWrite(raw=raw or key, key=key, mode=mode, value=write_value)
    raw = clean_text(value)
    return StateWrite(raw=raw, key=normalize_key(raw), mode="set", value="")


def normalize_operation(raw: dict[str, Any], index: int) -> Operation:
    agent_id = clean_text(raw.get("agent_id") or raw.get("id") or raw.get("op_id") or f"op_{index + 1}")
    reads = tuple(normalize_read(item) for item in (raw.get("reads") or []))
    writes = tuple(normalize_write(item) for item in (raw.get("writes") or []))
    return Operation(
        agent_id=agent_id,
        description=clean_text(raw.get("description") or raw.get("instruction") or ""),
        inputs=as_symbol_tuple(raw.get("inputs")),
        outputs=as_symbol_tuple(raw.get("outputs")),
        reads=reads,
        writes=writes,
        effect=normalize_value(raw.get("effect") or "read_only") or "read_only",
        idempotent=as_bool(raw.get("idempotent", True), True),
        index=index,
        request_seq=infer_request_seq(raw, agent_id, index),
    )


def operation_order(ops: Iterable[Operation]) -> dict[str, int]:
    return {op.agent_id: op.index for op in ops}


def conflict_rank(op: Operation) -> tuple[int, int, str]:
    if op.request_seq is not None:
        return (0, op.request_seq, op.agent_id)
    return (1, 0, op.agent_id)


def ordered_by_request_seq(left: Operation, right: Operation) -> tuple[Operation, Operation]:
    return (left, right) if conflict_rank(left) <= conflict_rank(right) else (right, left)


def conflict_orientation_label(left: Operation, right: Operation) -> str:
    return "request_seq" if left.request_seq is not None and right.request_seq is not None else "canonical agent-id order"


def sorted_agents(agent_ids: Iterable[str], order: dict[str, int]) -> list[str]:
    return sorted(agent_ids, key=lambda agent_id: (order.get(agent_id, 10**9), agent_id))


def fmt_set(values: Iterable[str]) -> str:
    return ", ".join(sorted(set(values)))


def add_edge(
    edge_reasons: dict[tuple[str, str], set[str]],
    before: str,
    after: str,
    reason: str,
) -> None:
    if before == after:
        return
    edge_reasons.setdefault((before, after), set()).add(reason)


def add_mutex(
    mutex_reasons: dict[frozenset[str], set[str]],
    left: str,
    right: str,
    reason: str,
) -> None:
    if left == right:
        return
    mutex_reasons.setdefault(frozenset((left, right)), set()).add(reason)


def add_dedup(
    dedup_reasons: dict[frozenset[str], set[str]],
    left: str,
    right: str,
    reason: str,
) -> None:
    if left == right:
        return
    dedup_reasons.setdefault(frozenset((left, right)), set()).add(reason)


def read_keys(op: Operation) -> set[str]:
    return {read.key for read in op.reads if read.key}


def write_keys(op: Operation) -> set[str]:
    return {write.key for write in op.writes if write.key}


def exact_inputs(op: Operation) -> set[str]:
    return set(op.inputs)


def exact_outputs(op: Operation) -> set[str]:
    return set(op.outputs)


def split_key(key: str) -> list[str]:
    return [part for part in normalize_key(key).split(":") if part]


def key_overlap(left: str, right: str) -> bool:
    """Conservative deterministic overlap with ':' hierarchy and '*' wildcards.

    Examples:
      record:x:* overlaps record:x:state
      record:x overlaps record:x:state
      side_effect:notify:record:x:state does not overlap record:x:state
    """
    if not left or not right:
        return False
    if left == right:
        return True

    left_parts = split_key(left)
    right_parts = split_key(right)
    if not left_parts or not right_parts:
        return False

    n = max(len(left_parts), len(right_parts))
    left_parts = left_parts + ["*"] * (n - len(left_parts))
    right_parts = right_parts + ["*"] * (n - len(right_parts))

    for lpart, rpart in zip(left_parts, right_parts):
        if lpart == "*" or rpart == "*":
            continue
        if lpart != rpart:
            return False
    return True


def overlapping_key_pairs(left_keys: Iterable[str], right_keys: Iterable[str]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for left in left_keys:
        for right in right_keys:
            if key_overlap(left, right):
                pairs.add((left, right))
    return pairs


def overlap_labels(pairs: Iterable[tuple[str, str]]) -> set[str]:
    labels: set[str] = set()
    for left, right in pairs:
        labels.add(left if left == right else f"{left} ~ {right}")
    return labels


def writes_overlapping_key(op: Operation, key: str) -> list[StateWrite]:
    return [write for write in op.writes if key_overlap(write.key, key)]


def strip_token_punctuation(token: str) -> str:
    token = token.strip().strip("\"'`.,;()[]{}")
    if "=" in token:
        token = token.split("=", 1)[0]
    return token.strip().strip("\"'`.,;()[]{}")


def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


def extract_structured_keys(text: str) -> set[str]:
    """Extract resource-like keys embedded in inputs/outputs/free text."""
    out: set[str] = set()
    if not text:
        return out

    patterns = (
        r"side_effect:[^\s,;\]\)\}]+",
        r"notes:[^\s,;\]\)\}]+",
        r"record:[^\s,;\]\)\}=]+",
        r"document:[^\s,;\]\)\}=]+",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            key = normalize_key(strip_token_punctuation(match.group(0)))
            if not key:
                continue
            out.add(key)
            if key.startswith("side_effect:notify:"):
                out.add(remove_prefix(key, "side_effect:notify:"))
            if key.startswith("notes:"):
                out.add(remove_prefix(key, "notes:"))
    return out


def operation_keys(op: Operation) -> set[str]:
    keys = read_keys(op) | write_keys(op)
    for item in op.inputs + op.outputs:
        keys |= extract_structured_keys(item)
    return {key for key in keys if key}


def overlapping_exact_io(producer: Operation, consumer: Operation) -> set[str]:
    return exact_outputs(producer) & exact_inputs(consumer)


def freshness_class(freshness: str) -> str:
    text = clean_text(freshness).lower()
    if any(marker in text for marker in LATEST_FRESHNESS_MARKERS):
        return "latest"
    if any(marker in text for marker in CURRENT_FRESHNESS_MARKERS):
        return "current"
    if any(marker in text for marker in STATIC_FRESHNESS_MARKERS):
        return "static"
    return "unknown"


def build_alias_candidates(op: Operation, raw: dict[str, Any], index: int) -> list[tuple[str, int]]:
    """Return (alias, priority). Lower priority wins when resolving refs."""
    aliases: list[tuple[str, int]] = [(op.agent_id, 0)]

    for key in ("op_id", "operation_id", "task_id", "id"):
        value = raw.get(key)
        if isinstance(value, str) and clean_text(value):
            aliases.append((clean_text(value), 0))

    match = TRAILING_NUM_RE.search(op.agent_id)
    if match:
        suffix = match.group(1)
        aliases.append((f"op_{int(suffix)}", 1))
        aliases.append((f"op_{suffix}", 1))

    aliases.append((f"op_{index + 1}", 2))
    return aliases


def build_alias_map(raw_tasks: list[dict[str, Any]], ops: list[Operation]) -> dict[str, str | None]:
    by_alias: dict[str, list[tuple[int, str]]] = {}
    for raw, op in zip(raw_tasks, ops):
        for alias, priority in build_alias_candidates(op, raw, op.index):
            norm_alias = normalize_alias(alias)
            if norm_alias:
                by_alias.setdefault(norm_alias, []).append((priority, op.agent_id))

    resolved: dict[str, str | None] = {}
    for alias, candidates in by_alias.items():
        best_priority = min(priority for priority, _ in candidates)
        best_agents = {agent_id for priority, agent_id in candidates if priority == best_priority}
        resolved[alias] = next(iter(best_agents)) if len(best_agents) == 1 else None
    return resolved


def explicit_ref_producers(input_value: str, alias_map: dict[str, str | None], warnings: list[str]) -> set[str]:
    producers: set[str] = set()
    for match in EXPLICIT_REF_RE.finditer(input_value):
        alias = normalize_alias(match.group(1))
        producer = alias_map.get(alias)
        if producer:
            producers.add(producer)
        elif alias in alias_map:
            warnings.append(f"ambiguous explicit reference ignored: {input_value!r}")
        else:
            warnings.append(f"unknown explicit reference ignored: {input_value!r}")
    return producers


def build_data_dependency_edges(
    ops: list[Operation],
    raw_tasks: list[dict[str, Any]],
    edge_reasons: dict[tuple[str, str], set[str]],
    warnings: list[str],
) -> None:
    producers_by_output: dict[str, set[str]] = {}
    for op in ops:
        for output in op.outputs:
            producers_by_output.setdefault(output, set()).add(op.agent_id)

    alias_map = build_alias_map(raw_tasks, ops)
    for consumer in ops:
        for input_value in consumer.inputs:
            for producer in producers_by_output.get(input_value, set()):
                add_edge(
                    edge_reasons,
                    producer,
                    consumer.agent_id,
                    f"data_dependency: output/input match {input_value!r}",
                )
            for producer in explicit_ref_producers(input_value, alias_map, warnings):
                add_edge(
                    edge_reasons,
                    producer,
                    consumer.agent_id,
                    f"data_dependency: explicit reference in input {input_value!r}",
                )


def handle_read_write(
    writer: Operation,
    reader: Operation,
    edge_reasons: dict[tuple[str, str], set[str]],
    mutex_reasons: dict[frozenset[str], set[str]],
) -> None:
    if not writer.writes or not reader.reads:
        return

    by_class: dict[str, set[str]] = {"latest": set(), "current": set(), "unknown": set(), "static": set()}
    for write in writer.writes:
        for read in reader.reads:
            if key_overlap(write.key, read.key):
                by_class[freshness_class(read.freshness)].add(
                    write.key if write.key == read.key else f"{write.key} ~ {read.key}"
                )

    if by_class["latest"]:
        add_edge(
            edge_reasons,
            writer.agent_id,
            reader.agent_id,
            f"state_read_write: reader needs latest/final state for {fmt_set(by_class['latest'])}",
        )
    if by_class["current"]:
        add_edge(
            edge_reasons,
            reader.agent_id,
            writer.agent_id,
            f"state_read_write: reader needs current pre-write state for {fmt_set(by_class['current'])}",
        )
    if by_class["unknown"]:
        add_mutex(
            mutex_reasons,
            writer.agent_id,
            reader.agent_id,
            f"state_read_write: unknown freshness for {fmt_set(by_class['unknown'])}",
        )
    # static reads intentionally produce no edge.


def is_unordered_log_key(key: str) -> bool:
    lowered = normalize_key(key)
    return lowered.startswith("notes:") or "unordered_log" in lowered or lowered.endswith(":log")


def note_id(write: StateWrite) -> str:
    text = f"{write.raw} {write.value}"
    match = NOTE_ID_RE.search(text)
    return normalize_value(match.group(1)) if match else ""


def write_pair_commutative(a: Operation, wa: StateWrite, b: Operation, wb: StateWrite) -> bool:
    if wa.mode == "add_to_set" and wb.mode == "add_to_set":
        return True

    if wa.mode == "set" and wb.mode == "set" and a.idempotent and b.idempotent:
        return bool(wa.value) and wa.value == wb.value

    if wa.mode == "append" and wb.mode == "append" and (
        is_unordered_log_key(wa.key) or is_unordered_log_key(wb.key)
    ):
        left_note_id = note_id(wa)
        right_note_id = note_id(wb)
        return bool(left_note_id and right_note_id and left_note_id != right_note_id)

    return False


def overlapping_write_pairs(a: Operation, b: Operation) -> list[tuple[StateWrite, StateWrite]]:
    pairs: list[tuple[StateWrite, StateWrite]] = []
    for wa in a.writes:
        for wb in b.writes:
            if key_overlap(wa.key, wb.key):
                pairs.append((wa, wb))
    return pairs


def commutative_writes(a: Operation, b: Operation, pairs: Sequence[tuple[StateWrite, StateWrite]]) -> bool:
    if not pairs:
        return True
    return all(write_pair_commutative(a, wa, b, wb) for wa, wb in pairs)


def build_state_dependency_edges_and_mutexes(
    ops: list[Operation],
    edge_reasons: dict[tuple[str, str], set[str]],
    mutex_reasons: dict[frozenset[str], set[str]],
) -> None:
    for idx, left in enumerate(ops):
        for right in ops[idx + 1 :]:
            handle_read_write(left, right, edge_reasons, mutex_reasons)
            handle_read_write(right, left, edge_reasons, mutex_reasons)

            write_pairs = overlapping_write_pairs(left, right)
            if not write_pairs or commutative_writes(left, right, write_pairs):
                continue

            overlap_text = fmt_set(overlap_labels((wa.key, wb.key) for wa, wb in write_pairs))

            # Duplicate external side effects are handled below as
            # duplicate_side_effect_key edges, not as unordered mutexes.
            if (
                left.effect == "external_side_effect"
                and right.effect == "external_side_effect"
                and any(wa.key.startswith("side_effect:") and wb.key.startswith("side_effect:") for wa, wb in write_pairs)
            ):
                continue

            if ORIENT_CONFLICTS_BY_REQUEST_SEQ:
                before, after = ordered_by_request_seq(left, right)
                add_edge(
                    edge_reasons,
                    before.agent_id,
                    after.agent_id,
                    "write_write_conflict: non-commutative overlap serialized by "
                    f"{conflict_orientation_label(left, right)} on {overlap_text}",
                )
            else:
                add_mutex(
                    mutex_reasons,
                    left.agent_id,
                    right.agent_id,
                    f"write_write_conflict: non-commutative overlap on {overlap_text}",
                )


def is_validation(op: Operation) -> bool:
    lowered = op.description.lower().strip()
    if any(lowered.startswith(word + " ") for word in VALIDATION_START_WORDS):
        return True
    for output in op.outputs:
        if any(marker in output for marker in VALIDATION_OUTPUT_MARKERS):
            return True
    return False


def is_precommit_operation(op: Operation) -> bool:
    """Operations allowed to guard a commit without being commits themselves."""
    if is_validation(op):
        return True
    if op.effect == "internal_write" and op.idempotent:
        return True
    return False


def is_non_idempotent_commit(op: Operation) -> bool:
    """Non-idempotent operations that should not be run speculatively."""
    if op.effect == "external_side_effect" and not op.idempotent:
        return True
    if op.effect == "internal_write" and bool(op.writes) and not op.idempotent:
        return True
    return False


def relevant(left: Operation, right: Operation) -> bool:
    if overlapping_key_pairs(operation_keys(left), operation_keys(right)):
        return True
    return bool(overlapping_exact_io(left, right) or overlapping_exact_io(right, left))


def side_effect_writes(op: Operation) -> list[StateWrite]:
    return [write for write in op.writes if write.key.startswith("side_effect:")]


def side_effect_overlap_pairs(left: Operation, right: Operation) -> list[tuple[StateWrite, StateWrite]]:
    if left.effect != "external_side_effect" or right.effect != "external_side_effect":
        return []
    pairs: list[tuple[StateWrite, StateWrite]] = []
    for lw in side_effect_writes(left):
        for rw in side_effect_writes(right):
            if key_overlap(lw.key, rw.key):
                pairs.append((lw, rw))
    return pairs


def duplicate_side_effect_pairs(left: Operation, right: Operation) -> list[tuple[StateWrite, StateWrite]]:
    pairs: list[tuple[StateWrite, StateWrite]] = []
    for lw, rw in side_effect_overlap_pairs(left, right):
        same_mode = lw.mode == rw.mode
        same_value = bool(lw.value) and lw.value == rw.value
        same_raw = bool(lw.raw and rw.raw) and normalize_value(lw.raw) == normalize_value(rw.raw)
        if same_mode and (same_value or same_raw):
            pairs.append((lw, rw))
    return pairs


def build_side_effect_constraints(
    ops: list[Operation],
    edge_reasons: dict[tuple[str, str], set[str]],
    mutex_reasons: dict[frozenset[str], set[str]],
    dedup_reasons: dict[frozenset[str], set[str]],
) -> None:
    # Commit barrier: relevant validations / safe internal writes should precede
    # non-idempotent commits. Other commits are not used as precommit guards,
    # which avoids accidental A->B and B->A cycles.
    for commit in ops:
        if not is_non_idempotent_commit(commit):
            continue
        for other in ops:
            if other.agent_id == commit.agent_id:
                continue
            if is_non_idempotent_commit(other):
                continue
            if (commit.agent_id, other.agent_id) in edge_reasons:
                continue
            if relevant(other, commit) and is_precommit_operation(other):
                add_edge(
                    edge_reasons,
                    other.agent_id,
                    commit.agent_id,
                    "commit_constraint: relevant validation/idempotent internal write precedes non-idempotent commit",
                )

    # Non-idempotent external side effects on the same external target are
    # idempotency-key dependencies. They are deterministic directed edges,
    # serialized by request_seq when present and otherwise by canonical
    # agent-id order, not merely mutex pairs.
    # Do not require same value/raw wording: parser outputs like
    # "customer_facing_notice" and "external_notification" can describe the
    # same notify side effect.
    for idx, left in enumerate(ops):
        for right in ops[idx + 1 :]:
            if not (is_non_idempotent_commit(left) and is_non_idempotent_commit(right)):
                continue
            overlap = side_effect_overlap_pairs(left, right)
            if not overlap:
                continue

            before, after = ordered_by_request_seq(left, right)
            overlap_text = fmt_set(overlap_labels((lw.key, rw.key) for lw, rw in overlap))
            add_edge(
                edge_reasons,
                before.agent_id,
                after.agent_id,
                "duplicate_side_effect_key: same non-idempotent side-effect target serialized by "
                f"{conflict_orientation_label(left, right)} on {overlap_text}",
            )
            add_dedup(
                dedup_reasons,
                left.agent_id,
                right.agent_id,
                f"dedup_required: duplicate non-idempotent side effect on {overlap_text}",
            )


def edge_items(edge_reasons: dict[tuple[str, str], set[str]], order: dict[str, int]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for (before, after), reasons in sorted(
        edge_reasons.items(),
        key=lambda item: (order.get(item[0][0], 10**9), order.get(item[0][1], 10**9), item[0]),
    ):
        items.append({"before": before, "after": after, "reasons": sorted(reasons)})
    return items


def pair_items(pair_reasons: dict[frozenset[str], set[str]], order: dict[str, int]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for pair, reasons in sorted(
        pair_reasons.items(),
        key=lambda item: tuple(order.get(agent_id, 10**9) for agent_id in sorted_agents(item[0], order)),
    ):
        items.append({"agents": sorted_agents(pair, order), "reasons": sorted(reasons)})
    return items


def mandatory_cycle_components(nodes: list[str], edges: Iterable[tuple[str, str]], order: dict[str, int]) -> list[list[str]]:
    """Return SCCs with size > 1 in mandatory_edges."""
    adjacency: dict[str, list[str]] = {node: [] for node in nodes}
    for before, after in edges:
        adjacency.setdefault(before, []).append(after)
        adjacency.setdefault(after, [])

    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    components: list[list[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for succ in adjacency.get(node, []):
            if succ not in indices:
                strongconnect(succ)
                lowlink[node] = min(lowlink[node], lowlink[succ])
            elif succ in on_stack:
                lowlink[node] = min(lowlink[node], indices[succ])

        if lowlink[node] == indices[node]:
            component: list[str] = []
            while True:
                item = stack.pop()
                on_stack.remove(item)
                component.append(item)
                if item == node:
                    break
            if len(component) > 1:
                components.append(sorted_agents(component, order))

    for node in nodes:
        if node not in indices:
            strongconnect(node)

    return sorted(components, key=lambda comp: tuple(order.get(agent_id, 10**9) for agent_id in comp))


def raw_agent_tasks(case: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = case.get("agent_tasks") or case.get("tasks") or case.get("parsed_tasks") or []
    if isinstance(tasks, dict):
        out: list[dict[str, Any]] = []
        for agent_id, item in tasks.items():
            raw = dict(item) if isinstance(item, dict) else {}
            raw.setdefault("agent_id", agent_id)
            out.append(raw)
        return out
    if isinstance(tasks, list):
        return [item for item in tasks if isinstance(item, dict)]
    return []


def extract_cases(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return [case for case in data["cases"] if isinstance(case, dict)]
    if isinstance(data, dict) and isinstance(data.get("agent_tasks"), list):
        return [data]
    if isinstance(data, list):
        return [case for case in data if isinstance(case, dict)]
    raise ValueError("input must be a parsed case, a list of cases, or an object with a cases array")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")



Edge = Tuple[str, str]


def edge_weight(reasons: Iterable[str]) -> int:
    """Return the finite evidence weight used by cycle resolution."""
    return max(1, len(set(reasons)))


def node_sort_key(agent_id: str, order: dict[str, int]) -> tuple[int, str]:
    return (order.get(agent_id, 10**9), agent_id)


def enumerate_cycles_in_component(
    component: Iterable[str],
    edges: set[Edge],
    order: dict[str, int],
) -> list[tuple[Edge, ...]]:
    """Enumerate each simple directed cycle once in one cyclic SCC.

    The smallest ordered node in a cycle is its unique DFS start node.  This
    removes rotational duplicates while retaining distinct cycles that share
    nodes or edges.
    """
    nodes = sorted(set(component), key=lambda node: node_sort_key(node, order))
    node_set = set(nodes)
    adjacency: dict[str, list[str]] = {node: [] for node in nodes}
    for before, after in edges:
        if before in node_set and after in node_set:
            adjacency[before].append(after)
    for successors in adjacency.values():
        successors.sort(key=lambda node: node_sort_key(node, order))

    rank = {node: index for index, node in enumerate(nodes)}
    cycles: list[tuple[Edge, ...]] = []
    for start in nodes:
        path = [start]
        visited = {start}

        def dfs(node: str) -> None:
            for successor in adjacency[node]:
                if successor == start:
                    cycles.append(tuple(zip(path, path[1:] + [start])))
                elif successor not in visited and rank[successor] >= rank[start]:
                    visited.add(successor)
                    path.append(successor)
                    dfs(successor)
                    path.pop()
                    visited.remove(successor)

        dfs(start)
    return cycles


def remaining_cycles(nodes: list[str], edges: set[Edge], order: dict[str, int]) -> list[tuple[Edge, ...]]:
    """Enumerate simple cycles only in SCCs that are currently cyclic."""
    components = mandatory_cycle_components(nodes, edges, order)
    cycles: list[tuple[Edge, ...]] = []
    for component in components:
        cycles.extend(enumerate_cycles_in_component(component, edges, order))
    return cycles


def has_path(edges: set[Edge], start: str, target: str) -> bool:
    """Return whether a directed path from start to target exists."""
    if start == target:
        return True
    adjacency: dict[str, list[str]] = {}
    for before, after in edges:
        adjacency.setdefault(before, []).append(after)

    stack = [start]
    visited = {start}
    while stack:
        node = stack.pop()
        for successor in adjacency.get(node, []):
            if successor == target:
                return True
            if successor not in visited:
                visited.add(successor)
                stack.append(successor)
    return False


def select_acyclic_edges(
    nodes: list[str],
    edge_reasons: dict[Edge, set[str]],
    order: dict[str, int],
) -> tuple[set[Edge], list[Edge]]:
    """Apply weighted cycle removal followed by reverse-add refinement."""
    selected = set(edge_reasons)
    weights = {edge: edge_weight(reasons) for edge, reasons in edge_reasons.items()}
    dropped: list[Edge] = []

    while True:
        cycles = remaining_cycles(nodes, selected, order)
        if not cycles:
            break

        cycle_counts: Counter[Edge] = Counter(edge for cycle in cycles for edge in cycle)
        if not cycle_counts:
            raise RuntimeError("cyclic SCC found but no directed cycle was enumerated")

        def removal_key(edge: Edge) -> tuple[Fraction, int, tuple[int, str], tuple[int, str]]:
            return (
                Fraction(weights[edge], cycle_counts[edge]),
                weights[edge],
                node_sort_key(edge[0], order),
                node_sort_key(edge[1], order),
            )

        edge_to_drop = min(cycle_counts, key=removal_key)
        selected.remove(edge_to_drop)
        dropped.append(edge_to_drop)

    # Python's stable sort makes equal-weight ties deterministic: the edge
    # removed earlier is considered first during reverse addition.
    for edge in sorted(dropped, key=lambda item: weights[item], reverse=True):
        before, after = edge
        if not has_path(selected, after, before):
            selected.add(edge)

    if mandatory_cycle_components(nodes, selected, order):
        raise RuntimeError("acyclic dependency refinement produced a cycle")
    return selected, dropped


def build_dependency(case: dict[str, Any]) -> dict[str, Any]:
    """Build rule-based dependency candidates, then return an acyclic DAG."""
    raw_tasks = raw_agent_tasks(case)
    ops = [normalize_operation(raw, index) for index, raw in enumerate(raw_tasks)]
    order = operation_order(ops)
    edge_reasons: dict[Edge, set[str]] = {}
    mutex_reasons: dict[frozenset[str], set[str]] = {}
    dedup_reasons: dict[frozenset[str], set[str]] = {}
    warnings: list[str] = []

    build_data_dependency_edges(ops, raw_tasks, edge_reasons, warnings)
    build_state_dependency_edges_and_mutexes(ops, edge_reasons, mutex_reasons)
    build_side_effect_constraints(ops, edge_reasons, mutex_reasons, dedup_reasons)

    nodes = [op.agent_id for op in ops]
    selected, dropped = select_acyclic_edges(nodes, edge_reasons, order)
    if dropped:
        final_dropped = len(set(edge_reasons) - selected)
        warnings.append(
            f"temporarily removed {len(dropped)} candidate dependency edge(s) during weighted cycle resolution; "
            f"reverse-add refinement left {final_dropped} edge(s) excluded"
        )

    selected_reasons = {edge: edge_reasons[edge] for edge in selected}
    return {
        "case_id": case.get("case_id"),
        "num_agent_tasks": len(ops),
        "agents": [op.agent_id for op in ops],
        "mandatory_edges": edge_items(selected_reasons, order),
        "mutex_pairs": pair_items(mutex_reasons, order),
        "dedup_required_pairs": pair_items(dedup_reasons, order),
        "has_mandatory_cycle": False,
        "mandatory_cycle_components": [],
        "warnings": sorted(set(warnings)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    input_path = Path(args.input)
    cases = extract_cases(read_json(input_path))
    if args.case_id:
        cases = [case for case in cases if str(case.get("case_id")) == args.case_id]
        if not cases:
            raise SystemExit(f"case_id not found: {args.case_id}")

    results: list[dict[str, Any]] = []
    case_runtimes_ms: list[float] = []
    progress_interval = max(0, int(args.progress_interval or 0))
    progress_timing: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        case_start = time.perf_counter()
        result = build_dependency(case)
        runtime_ms = (time.perf_counter() - case_start) * 1000
        result["runtime_ms"] = runtime_ms
        results.append(result)
        case_runtimes_ms.append(runtime_ms)
        if progress_interval and index % progress_interval == 0:
            elapsed_s = time.perf_counter() - start
            avg_case_s = sum(case_runtimes_ms) / len(case_runtimes_ms) / 1000
            progress_timing.append(
                {
                    "cases_done": index,
                    "cases_total": len(cases),
                    "cumulative_runtime_s": elapsed_s,
                    "avg_case_runtime_s": avg_case_s,
                }
            )
            print(
                "[acyclic_dependency_build timing] "
                f"input={input_path} cases={index}/{len(cases)} "
                f"cumulative_runtime_s={elapsed_s:.3f} avg_case_runtime_s={avg_case_s:.6f}",
                file=sys.stderr,
            )

    avg_case_runtime_ms = sum(case_runtimes_ms) / len(case_runtimes_ms) if case_runtimes_ms else 0.0
    total_runtime_ms = (time.perf_counter() - start) * 1000
    output = {
        "source_path": str(input_path.resolve()),
        "schema": {
            "mandatory_edges": {"before": "agent_id", "after": "agent_id", "reasons": "list[str]"},
            "mutex_pairs": {"agents": "list[agent_id, agent_id]", "reasons": "list[str]"},
            "dedup_required_pairs": {
                "agents": "list[agent_id, agent_id]",
                "reasons": "list[str]",
                "meaning": "serializing alone may still be unsafe because operations look like duplicate non-idempotent side effects",
            },
            "has_mandatory_cycle": "boolean",
            "mandatory_cycle_components": "list[list[agent_id]]",
            "scheduling": "not_implemented",
        },
        "num_cases": len(results),
        "num_agent_tasks": sum(case["num_agent_tasks"] for case in results),
        "num_mandatory_edges": sum(len(case["mandatory_edges"]) for case in results),
        "num_mutex_pairs": sum(len(case["mutex_pairs"]) for case in results),
        "num_dedup_required_pairs": sum(len(case["dedup_required_pairs"]) for case in results),
        "num_cases_with_mandatory_cycles": sum(1 for case in results if case["has_mandatory_cycle"]),
        "metrics": {
            "total_runtime_ms": total_runtime_ms,
            "avg_case_runtime_ms": avg_case_runtime_ms,
            "progress_timing": progress_timing,
        },
        "cases": results,
    }
    if args.output:
        write_json(Path(args.output), output)
    if args.stdout:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build acyclic dependency constraints from parsed agent task JSON")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to parse_task output JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output dependency JSON path")
    parser.add_argument("--case-id", default="", help="Optional single case_id to process")
    parser.add_argument("--stdout", action="store_true", help="Also print dependency JSON to stdout")
    parser.add_argument("--progress-interval", type=int, default=0, help="Print cumulative timing every N cases; 0 disables")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output = run(args)
    if args.output:
        print(
            f"Wrote {args.output} cases={output['num_cases']} edges={output['num_mandatory_edges']} "
            f"mutex_pairs={output['num_mutex_pairs']} dedup_required_pairs={output['num_dedup_required_pairs']} "
            f"cycle_cases={output['num_cases_with_mandatory_cycles']} "
            f"total_runtime_ms={output['metrics']['total_runtime_ms']:.3f} "
            f"avg_case_runtime_ms={output['metrics']['avg_case_runtime_ms']:.3f}"
        )


if __name__ == "__main__":
    main()
