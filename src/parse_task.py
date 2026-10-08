#!/usr/bin/env python3
"""
Parse public benchmark task text into resource-level task signatures.

The script reads only public agent_id + natural-language instruction text,
calls an LLM, then normalizes the result into structured reads/writes suitable
for dependency and DAG construction.

Default:
  python src/parse_task.py run

Inspect one prompt:
  python src/parse_task.py inspect-prompt --case-id case_000000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    token_usage_source: str


class ChatCompletionClient:
    """Minimal client for an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout: int = 120,
        max_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.session = requests.Session()

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        key = self.api_key.strip()
        headers = {
            "Authorization": key if key.lower().startswith("bearer ") else f"Bearer {key}",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    self.base_url, headers=headers, json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                usage = data.get("usage") or {}
                return LLMResponse(
                    content=data["choices"][0]["message"]["content"],
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    total_tokens=int(usage.get("total_tokens", 0) or 0),
                    token_usage_source="api_usage" if usage else "not_returned",
                )
            except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
                if attempt == self.max_retries:
                    raise
                time.sleep(min(30, 2**attempt))
        raise RuntimeError("LLM request failed")


def extract_json_object(text: str) -> dict[str, Any]:
    """Accept plain JSON or a JSON object embedded in model text."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("model response contains no JSON object")


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR.parent / "data" / "sgd_micro_10_public.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "results" / "parsed.json"
DEFAULT_MODEL_NAME = os.environ.get("LLM_MODEL", "gpt-4o-mini")
DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1/chat/completions")

AGENT_TASK_KEYS = (
    "agent_id",
    "description",
    "inputs",
    "outputs",
    "reads",
    "writes",
    "effect",
    "idempotent",
)
READ_KEYS = ("raw", "key", "freshness")
WRITE_KEYS = ("raw", "key", "mode", "value")
EFFECT_VALUES = ("read_only", "internal_write", "external_side_effect")


# ---------------------------------------------------------------------------
# I/O and public helpers
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def public_requests(case: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        return list(case["public_input"]["agent_requests"])
    except KeyError:
        try:
            return list(case["input"]["agent_requests"])
        except KeyError as exc:
            raise ValueError("case does not contain public_input.agent_requests or input.agent_requests") from exc


def agent_lines(case: dict[str, Any]) -> list[str]:
    return [f"{req['agent_id']}: {req['instruction']}" for req in public_requests(case)]


def public_context(case: dict[str, Any]) -> str:
    public = case.get("public_input") or {}
    blocks: list[str] = []
    task = str(public.get("task") or "").strip()
    if task:
        blocks.append(f"Task context: {task}")

    rules = public.get("domain_rules") or []
    if isinstance(rules, list):
        cleaned = [str(rule).strip() for rule in rules if str(rule).strip()]
        if cleaned:
            blocks.append("Public domain rules:\n" + "\n".join(f"- {rule}" for rule in cleaned))

    return "\n\n".join(blocks)


def attach_public_request_metadata(task: dict[str, Any], req: dict[str, Any]) -> dict[str, Any]:
    out = dict(task)
    if "request_seq" in req:
        try:
            out["request_seq"] = int(req["request_seq"])
        except (TypeError, ValueError):
            pass
    return out


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def build_parse_prompt(case: dict[str, Any]) -> tuple[str, str]:
    case_id = str(case["case_id"])
    agents = "\n".join(agent_lines(case))
    context = public_context(case)
    context_block = f"\nPublic context:\n{context}\n" if context else ""

    system_prompt = (
        "Extract conservative task signatures from natural-language agent requests. "
        "Use only public text and ordinary operational meaning. Capture field-level "
        "resources, produced prerequisite values, resource reads/writes, and external "
        "side effects for dependency analysis. Return JSON only."
    )

    user_prompt = f"""
Return this JSON shape:
{{
  "case_id": "{case_id}",
  "agent_tasks": [
    {{
      "agent_id": "<agent_id>",
      "description": "<original task text>",
      "inputs": [],
      "outputs": [],
      "reads": [{{"raw": "<resource read>", "key": "<canonical key>", "freshness": "current|latest|static"}}],
      "writes": [{{"raw": "<resource write>", "key": "<canonical key>", "mode": "set|append|send", "value": "<value>"}}],
      "effect": "read_only|internal_write|external_side_effect",
      "idempotent": true
    }}
  ]
}}

General rules:
- Preserve every public agent exactly once, and keep "description" equal to the original task text.
- Split compound instructions into all visible produced values, required values, reads, writes, and external actions.
- Be conservative: do not infer dependency structure that the wording does not support, but do extract prerequisites that the wording explicitly creates.
- Downstream matching is exact. Missing resource-specific tokens lose dependencies; over-general tokens create false dependencies.

Recommended extraction pass for each request:
1. Identify the smallest visible business resource. If a field/attribute is named, make the resource field-level, not whole-record-level.
2. Identify semantic values the request produces for later use, such as internal keys, validation results, approvals, receipts, readiness markers, and derived decisions.
3. Identify semantic values the request explicitly requires, such as "using its internal key", "through the resolved key", "after validation passes", or "once eligibility is confirmed".
4. Identify resource reads, resource writes, and irreversible external side effects.
5. Check that every input/output token and read/write key includes enough resource identity to avoid matching unrelated resources.

Resource keys:
- Use stable lowercase snake-style keys so the same visible resource receives the same key across agents.
- Prefer prefixes such as "field:", "record:", "document:", "table:", "column:", "file:", "api:", "notes:", and "side_effect:".
- Field-level keys should include the visible field/attribute plus the record/entity identity. Keep visible record IDs, organizations/sites, dates, channels, event IDs, and action names when present.
- Different fields of the same record are different resources. Example shape: "field:<record_id>:<field_name>".
- Do not collapse resources to generic keys such as "record:current_record", "field:status", or "token:internal_record_key".

Semantic inputs and outputs:
- Use inputs/outputs only for produced prerequisites or intermediate values, not for every read/write.
- If a request finds, locates, identifies, or resolves an internal record/system/operational key for a resource, output "token:internal_record_key:<resource_key>" even when the concrete key value is not visible.
- If a request says it uses an internal record/system/operational key, the resolved internal key, or the internal key for that same field, add input "token:internal_record_key:<resource_key>".
- If a request validates/checks/reviews eligibility or operational conditions for a resource, output "token:validation:<resource_key>".
- If a guarded write or side effect says it runs after validation, once eligibility is confirmed, only after a check passes, or after an approval/permission result, add the matching resource-specific validation/approval input.
- For visible produced IDs such as KEY-17, APPROVAL-3, RECEIPT-9, use "token:<visible_id>" only if the ID is specific enough. Otherwise include the resource key too.
- Never use one generic token for many resources. Tokens for internal keys, validation, approval, readiness, and receipts must be resource-specific unless a visible globally unique ID is present.
- The phrase naming a produced identifier is not automatically a read. Add a read only when the task says to inspect, check, validate, summarize, use current state, or use latest/final/confirmed state.

Reads and writes:
- Reads are resources consulted by the task. Writes are resources changed by the task.
- A pure update/revise/apply/set to a field is a write, not also a read, unless the text explicitly says current/latest/final/confirmed/settled state is used.
- "current", "present", "inspect", "read", "summarize", "validate", and "check" usually indicate a read.
- Use freshness "latest" for final/latest/confirmed/settled state; use "static" for reference documents or schemas; otherwise use "current".
- Writes use mode "set" for updates or published internal artifacts, "append" for notes/logs, and "send" for irreversible external actions.
- Appending a note writes a notes/log resource, preferably "notes:<resource_key>", with mode "append"; keep visible note IDs in raw/value.
- effect is "read_only" for no writes, "internal_write" for internal writes only, and "external_side_effect" if any irreversible external action is present.
- idempotent is false for external sends and appends unless the task explicitly says the operation is retry-safe. A deduplication scope identifies the side effect, but does not by itself make the external action idempotent.

External actions:
- Represent each visible send/issue/trigger/print/order/refund/email/API action as its own write.
- If an idempotency or deduplication scope is visible, make that scope the side-effect identity under a "side_effect:" key. The same visible dedup scope must produce the exact same key.
- If no deduplication scope is visible, build the side-effect key from the visible event ID, channel/API, action, and target resource.
- External side-effect keys identify the irreversible action. Do not add event IDs, channels, or dedup scopes as ordinary reads.

Mini examples:
- "Locate the internal system record key for the room assignment field of booking record BK-7." =>
  outputs ["token:internal_record_key:field:bk_7:room_assignment"], read_only, no writes.
- "Update the room assignment field of booking record BK-7 to 'A101' using its internal operational record key." =>
  inputs ["token:internal_record_key:field:bk_7:room_assignment"], writes [{{"key":"field:bk_7:room_assignment","mode":"set","value":"A101"}}], no reads unless current/latest state is also requested.
- "Validate whether the badge print status field of visitor access record ACCESS-4 passes the required operational validation." =>
  outputs ["token:validation:field:access_4:badge_print_status"], current read of that field.
- "After the required validation passes, issue the refund request on the payments-api channel for event EVT-9, using the latest confirmed booking status field of booking record BK-7. Use deduplication scope EVT-9/payments-api/refund-request." =>
  inputs ["token:validation:field:bk_7:booking_status"], latest read of "field:bk_7:booking_status", and one send write with key "side_effect:evt_9:payments_api:refund_request".
{context_block}

Agent tasks:
{agents}
""".strip()
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Resource parsing
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().strip("."))


def read_obj(raw: str, key: str, freshness: str) -> dict[str, str]:
    return {"raw": clean_text(raw), "key": key, "freshness": freshness}


def write_obj(raw: str, key: str, mode: str, value: str) -> dict[str, str]:
    return {"raw": clean_text(raw), "key": key, "mode": mode, "value": str(value)}


def fallback_task(agent_id: str, description: str) -> dict[str, Any]:
    """Schema-only fallback used when an LLM response is missing or invalid."""
    return {
        "agent_id": agent_id,
        "description": description,
        "inputs": [],
        "outputs": [f"result:{agent_id}"],
        "reads": [],
        "writes": [],
        "effect": "read_only",
        "idempotent": True,
    }


# ---------------------------------------------------------------------------
# LLM output normalization
# ---------------------------------------------------------------------------


def as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def normalize_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "idempotent", "1"}:
        return True
    if text in {"false", "no", "non-idempotent", "non_idempotent", "0"}:
        return False
    return default


def normalize_read(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        text = clean_text(value)
        return read_obj(text, text, "current")
    raw = clean_text(value.get("raw") or value.get("description") or value.get("key") or "")
    key = clean_text(value.get("key") or raw)
    freshness = clean_text(value.get("freshness") or "current").lower()
    return read_obj(raw or key, key, freshness)


def normalize_write(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        text = clean_text(value)
        return write_obj(text, text, "set", "")
    raw = clean_text(value.get("raw") or value.get("description") or value.get("key") or "")
    key = clean_text(value.get("key") or raw)
    mode = clean_text(value.get("mode") or "set").lower()
    write_value = value.get("value", "")
    return write_obj(raw or key, key, mode, str(write_value))


def raw_agent_task_items(raw: dict[str, Any]) -> list[dict[str, Any]]:
    items = raw.get("agent_tasks") or raw.get("tasks") or raw.get("parsed_tasks") or []
    if isinstance(items, dict):
        out = []
        for agent_id, value in items.items():
            item = dict(value) if isinstance(value, dict) else {}
            item.setdefault("agent_id", agent_id)
            out.append(item)
        return out
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def normalize_model_task(raw_item: Any, agent_id: str, description: str) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    fallback = fallback_task(agent_id, description)
    if not isinstance(raw_item, dict):
        return fallback, ["agent task must be an object; used schema-only fallback"]

    extra = sorted(set(raw_item) - set(AGENT_TASK_KEYS))
    if extra:
        errors.append(f"dropped extra keys: {extra}")

    raw_agent_id = str(raw_item.get("agent_id") or "")
    if raw_agent_id and raw_agent_id != agent_id:
        errors.append(f"agent_id mismatch {raw_agent_id!r}; used public agent_id {agent_id!r}")

    raw_description = str(raw_item.get("description") or "")
    if raw_description and raw_description != description:
        errors.append("description mismatch; used public task text")

    effect = clean_text(raw_item.get("effect") or fallback["effect"]).lower()
    if effect not in EFFECT_VALUES:
        errors.append(f"invalid effect {effect!r}; used fallback effect {fallback['effect']!r}")
        effect = fallback["effect"]

    model_task = {
        "agent_id": agent_id,
        "description": description,
        "inputs": as_string_list(raw_item.get("inputs", fallback["inputs"])),
        "outputs": as_string_list(raw_item.get("outputs", fallback["outputs"])),
        "reads": [normalize_read(item) for item in (raw_item.get("reads") or fallback["reads"])],
        "writes": [normalize_write(item) for item in (raw_item.get("writes") or fallback["writes"])],
        "effect": effect,
        "idempotent": normalize_bool(raw_item.get("idempotent", fallback["idempotent"]), fallback["idempotent"]),
    }

    return model_task, errors


def normalize_case_parse(raw: dict[str, Any], case: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], list[dict[str, str]]]:
    case_errors: list[str] = []
    task_errors: list[dict[str, str]] = []
    reqs = public_requests(case)
    valid_agents = {str(req["agent_id"]) for req in reqs}
    by_agent = {str(item.get("agent_id", "")): item for item in raw_agent_task_items(raw)}

    if raw.get("case_id") not in {case.get("case_id"), str(case.get("case_id"))}:
        case_errors.append(f"model case_id mismatch: {raw.get('case_id')!r}")

    unknown = sorted(set(by_agent) - valid_agents - {""})
    if unknown:
        case_errors.append(f"ignored unknown agent_id values: {unknown}")

    normalized: list[dict[str, Any]] = []
    for req in reqs:
        agent_id = str(req["agent_id"])
        description = str(req["instruction"])
        raw_item = by_agent.get(agent_id)
        if raw_item is None:
            task = fallback_task(agent_id, description)
            task_errors.append({"agent_id": agent_id, "error": "missing model task; used schema-only fallback"})
        else:
            task, errors = normalize_model_task(raw_item, agent_id, description)
            for error in errors:
                task_errors.append({"agent_id": agent_id, "error": error})
        task = attach_public_request_metadata(task, req)
        normalized.append(task)

    return normalized, case_errors, task_errors


# ---------------------------------------------------------------------------
# LLM runner
# ---------------------------------------------------------------------------


def parse_case_with_llm(
    case: dict[str, Any],
    client: ChatCompletionClient,
    *,
    include_raw: bool = False,
) -> dict[str, Any]:
    system_prompt, user_prompt = build_parse_prompt(case)
    start = time.perf_counter()
    response = client.complete(system_prompt, user_prompt)
    runtime_ms = (time.perf_counter() - start) * 1000
    raw = extract_json_object(response.content)
    agent_tasks, case_errors, task_errors = normalize_case_parse(raw, case)
    result: dict[str, Any] = {
        "case_id": case["case_id"],
        "agent_tasks": agent_tasks,
        "errors": case_errors,
        "task_errors": task_errors,
        "runtime_ms": runtime_ms,
        "token_usage": {
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
            "source": response.token_usage_source,
        },
    }
    if include_raw:
        result["raw_model_output"] = response.content
    return result


def aggregate_token_usage(cases: Iterable[dict[str, Any]]) -> dict[str, int]:
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for case in cases:
        token_usage = case.get("token_usage", {}) or {}
        for key in usage:
            usage[key] += int(token_usage.get(key, 0) or 0)
    return usage


def count_agent_task_errors(cases: Iterable[dict[str, Any]]) -> int:
    return sum(len(case.get("task_errors", [])) for case in cases)


def run_parse(args: argparse.Namespace) -> None:
    dataset = read_jsonl(Path(args.dataset))
    if args.limit and args.limit > 0:
        dataset = dataset[: args.limit]

    api_key = (
        args.api_key
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise SystemExit("Missing API key. Pass --api-key or set LLM_API_KEY / OPENAI_API_KEY.")

    client = ChatCompletionClient(
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        temperature=args.temperature,
        max_retries=args.max_retries,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
    )

    results: list[dict[str, Any]] = []
    start_all = time.perf_counter()
    for idx, case in enumerate(dataset, start=1):
        print(f"[{idx}/{len(dataset)}] Parsing {case['case_id']}")
        try:
            parsed_case = parse_case_with_llm(case, client, include_raw=args.include_raw)
        except Exception as exc:
            tasks = [
                attach_public_request_metadata(
                    fallback_task(str(req["agent_id"]), str(req["instruction"])),
                    req,
                )
                for req in public_requests(case)
            ]
            parsed_case = {
                "case_id": case.get("case_id", f"case_{idx}"),
                "agent_tasks": tasks,
                "errors": [str(exc)],
                "task_errors": [
                    {"agent_id": str(req["agent_id"]), "error": "case-level LLM parse failed; used schema-only fallback"}
                    for req in public_requests(case)
                ],
                "runtime_ms": 0.0,
                "token_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "source": "error",
                },
            }
        results.append(parsed_case)
        print(
            "  "
            f"agents={len(parsed_case['agent_tasks'])} "
            f"case_errors={len(parsed_case.get('errors', []))} "
            f"task_errors={len(parsed_case.get('task_errors', []))} "
            f"latency_ms={float(parsed_case.get('runtime_ms', 0.0)):.2f}"
        )

    total_runtime_ms = (time.perf_counter() - start_all) * 1000
    avg_case_runtime_s = (total_runtime_ms / max(1, len(results))) / 1000
    avg_llm_case_runtime_s = (
        sum(float(case.get("runtime_ms", 0.0) or 0.0) for case in results)
        / max(1, len(results))
        / 1000
    )
    output = {
        "dataset_path": str(Path(args.dataset).resolve()),
        "model": args.model,
        "base_url": args.base_url,
        "parser_mode": "llm",
        "schema": {
            "agent_task_keys": list(AGENT_TASK_KEYS),
            "read_keys": list(READ_KEYS),
            "write_keys": list(WRITE_KEYS),
            "effect_values": list(EFFECT_VALUES),
            "idempotent_type": "boolean",
        },
        "num_cases": len(results),
        "num_agent_tasks": sum(len(case.get("agent_tasks", [])) for case in results),
        "total_runtime_ms": total_runtime_ms,
        "total_runtime_s": total_runtime_ms / 1000,
        "avg_case_runtime_s": avg_case_runtime_s,
        "avg_llm_case_runtime_s": avg_llm_case_runtime_s,
        "cases_with_errors": sum(1 for case in results if case.get("errors")),
        "agent_tasks_with_errors": count_agent_task_errors(results),
        "token_usage": aggregate_token_usage(results),
        "cases": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved parsed tasks to: {output_path}")
    print(f"Average case runtime: {avg_case_runtime_s:.3f}s")
    print(f"Average LLM call runtime: {avg_llm_case_runtime_s:.3f}s")


def inspect_prompt(args: argparse.Namespace) -> None:
    dataset = read_jsonl(Path(args.dataset))
    for case in dataset:
        if str(case["case_id"]) == args.case_id:
            system_prompt, user_prompt = build_parse_prompt(case)
            print("===== SYSTEM =====")
            print(system_prompt)
            print("\n===== USER =====")
            print(user_prompt)
            return
    raise SystemExit(f"case_id not found: {args.case_id}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse public benchmark task text into resource event JSON")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Call the LLM parser and save parsed task JSON")
    run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Public semantic dependency-order JSONL dataset")
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Detailed parsed task JSON")
    run.add_argument("--model", default=DEFAULT_MODEL_NAME)
    run.add_argument("--base-url", default=DEFAULT_BASE_URL)
    run.add_argument("--api-key", default="", help="API key. Defaults to LLM_API_KEY or OPENAI_API_KEY.")
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--max-retries", type=int, default=3)
    run.add_argument("--timeout", type=int, default=120)
    run.add_argument("--max-tokens", type=int, default=None)
    run.add_argument("--limit", type=int, default=0, help="Number of cases to parse; 0 means all")
    run.add_argument("--include-raw", action="store_true", help="Include raw model text in the result JSON")
    run.set_defaults(func=run_parse)

    ip = sub.add_parser("inspect-prompt", help="Print the exact prompt for one public case")
    ip.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ip.add_argument("--case-id", required=True)
    ip.set_defaults(func=inspect_prompt)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
