from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
from datetime import UTC, datetime
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


JUDGE_SCHEMA_KEYS = [
    "label_is_acceptable",
    "relevance_correct",
    "direction_correct",
    "strength_correct",
    "sentiment_correct",
    "judge_confidence",
    "issue_type",
    "suggested_is_relevant",
    "suggested_price_pressure_direction",
    "suggested_price_pressure_strength",
    "suggested_primary_topic",
    "judge_reason",
]

TOPICS = ["shortage", "tariff", "demand", "supply_chain", "export_controls", "oversupply", "other"]
DIRECTIONS = ["upward", "downward", "neutral"]
ISSUE_TYPES = ["none", "relevance", "direction", "strength", "sentiment", "topic", "unsupported_reason", "format"]
JUDGE_VERSION = "v1"

GUIDED_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "label_is_acceptable": {"type": "boolean"},
        "relevance_correct": {"type": "boolean"},
        "direction_correct": {"type": "boolean"},
        "strength_correct": {"type": "boolean"},
        "sentiment_correct": {"type": "boolean"},
        "judge_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "issue_type": {"type": "string", "enum": ISSUE_TYPES},
        "suggested_is_relevant": {"type": "boolean"},
        "suggested_price_pressure_direction": {"type": "string", "enum": DIRECTIONS},
        "suggested_price_pressure_strength": {"type": "integer", "minimum": 0, "maximum": 5},
        "suggested_primary_topic": {"type": "string", "enum": TOPICS},
        "judge_reason": {"type": "string"},
    },
    "required": JUDGE_SCHEMA_KEYS,
    "additionalProperties": False,
}


def _json_from_text(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        payload = match.group(0)
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(payload)
            if not isinstance(parsed, dict):
                raise
            return parsed


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _clip_float(value: object, low: float, high: float) -> float:
    parsed = float(value or 0)
    return max(low, min(high, parsed))


def _clip_int(value: object, low: int, high: int) -> int:
    parsed = int(float(value or 0))
    return max(low, min(high, parsed))


def _normalise_judgement(parsed: dict) -> dict:
    issue_type = str(parsed.get("issue_type", "none")).lower()
    if issue_type not in ISSUE_TYPES:
        issue_type = "format"

    direction = str(parsed.get("suggested_price_pressure_direction", "neutral")).lower()
    if direction not in DIRECTIONS:
        direction = "neutral"

    topic = str(parsed.get("suggested_primary_topic", "other")).lower()
    if topic not in TOPICS:
        topic = "other"

    return {
        "label_is_acceptable": _as_bool(parsed.get("label_is_acceptable", False)),
        "relevance_correct": _as_bool(parsed.get("relevance_correct", False)),
        "direction_correct": _as_bool(parsed.get("direction_correct", False)),
        "strength_correct": _as_bool(parsed.get("strength_correct", False)),
        "sentiment_correct": _as_bool(parsed.get("sentiment_correct", False)),
        "judge_confidence": _clip_float(parsed.get("judge_confidence", 0), 0.0, 1.0),
        "issue_type": issue_type,
        "suggested_is_relevant": _as_bool(parsed.get("suggested_is_relevant", False)),
        "suggested_price_pressure_direction": direction,
        "suggested_price_pressure_strength": _clip_int(parsed.get("suggested_price_pressure_strength", 0), 0, 5),
        "suggested_primary_topic": topic,
        "judge_reason": str(parsed.get("judge_reason", ""))[:700],
    }


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            if line.strip():
                records.append(json.loads(line))
    return records


def _input_files(input_dir: str, input_path: str | None) -> list[Path]:
    if input_path:
        return [Path(input_path)]
    root = Path(input_dir)
    files = sorted(root.glob("news_to_enrich_*.jsonl"))
    if files:
        return files
    fallback = root.parent / "news_to_enrich.jsonl"
    return [fallback] if fallback.exists() else []


def _enriched_files(enriched_dir: str, enriched_path: str | None) -> list[Path]:
    if enriched_path:
        return [Path(enriched_path)]
    root = Path(enriched_dir)
    files = sorted(root.glob("news_enriched_*.jsonl"))
    if files:
        return files
    fallback = root / "news_enriched.jsonl"
    return [fallback] if fallback.exists() else []


def _batch_suffix(path: Path) -> str:
    for prefix in ("news_enriched_", "news_to_enrich_"):
        if path.stem.startswith(prefix):
            return path.stem[len(prefix) :]
    return path.stem


def _build_input_lookup(input_files: list[Path]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for path in input_files:
        for record in _load_jsonl(path):
            url = record.get("url")
            if url:
                lookup[str(url)] = record
    return lookup


def _sample_records(records: list[dict], sample_size: int, seed: int) -> list[dict]:
    if sample_size <= 0 or len(records) <= sample_size:
        return records

    relevant = [record for record in records if record.get("is_relevant")]
    non_relevant = [record for record in records if not record.get("is_relevant")]
    relevant_target = min(len(relevant), max(1, sample_size // 2))
    non_relevant_target = min(len(non_relevant), sample_size - relevant_target)

    rng = random.Random(seed)
    sample = rng.sample(relevant, relevant_target) + rng.sample(non_relevant, non_relevant_target)
    if len(sample) < sample_size:
        remaining = [record for record in records if record not in sample]
        sample.extend(rng.sample(remaining, min(len(remaining), sample_size - len(sample))))
    sample.sort(key=lambda record: str(record.get("published_at", "")))
    return sample


def _prompt(source_record: dict, enriched_record: dict) -> str:
    label_payload = {
        "is_relevant": enriched_record.get("is_relevant"),
        "relevance_score": enriched_record.get("relevance_score"),
        "sentiment_score": enriched_record.get("sentiment_score", enriched_record.get("sentiment")),
        "price_pressure_direction": enriched_record.get("price_pressure_direction"),
        "price_pressure_strength": enriched_record.get("price_pressure_strength"),
        "primary_topic": enriched_record.get("primary_topic"),
        "reason_short": enriched_record.get("reason_short"),
    }
    return (
        "Audit this AI label for an electronics and semiconductor price-pressure dataset.\n"
        "Return JSON only with these keys: "
        + ", ".join(JUDGE_SCHEMA_KEYS)
        + ".\n"
        "Judge whether the label is acceptable for downstream monthly ML features.\n"
        "Important rules:\n"
        "- Relevant means the article directly concerns semiconductor/electronics component prices, supply, demand, tariffs, export controls, inventories, production capacity, foundries, chips, wafers, memory, or electronics manufacturing.\n"
        "- Do not mark food chips, sports microchips, entertainment, generic politics, or generic macro news as relevant unless there is a clear semiconductor/electronics market link.\n"
        "- upward means likely higher semiconductor/electronics prices or tighter supply.\n"
        "- downward means likely lower semiconductor/electronics prices, weaker demand, oversupply, or new capacity easing scarcity.\n"
        "- neutral means no clear price-pressure direction.\n"
        "- Focus on whether the label is plausible and supported by the article text, not whether it is perfect.\n\n"
        f"Article title: {source_record.get('title', enriched_record.get('title', ''))}\n"
        f"Source: {source_record.get('source', enriched_record.get('source', ''))}\n"
        f"Published at: {source_record.get('published_at', enriched_record.get('published_at', ''))}\n"
        f"Existing label JSON:\n{json.dumps(label_payload, ensure_ascii=True)}\n\n"
        f"Article text:\n{source_record.get('text', '')}"
    )


def _build_llm(
    model: str,
    max_tokens: int,
    max_model_len: int | None,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    trust_remote_code: bool,
    tokenizer_mode: str | None,
    config_format: str | None,
    load_format: str | None,
    guided_json: bool,
) -> tuple[LLM, SamplingParams]:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HF_TOKEN", "")
    llm_kwargs = {
        "model": model,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": trust_remote_code,
    }
    if max_model_len:
        llm_kwargs["max_model_len"] = max_model_len
    if tokenizer_mode:
        llm_kwargs["tokenizer_mode"] = tokenizer_mode
    if config_format:
        llm_kwargs["config_format"] = config_format
    if load_format:
        llm_kwargs["load_format"] = load_format

    print(f"Loading judge model: {model}", flush=True)
    llm = LLM(**llm_kwargs)
    if guided_json:
        print("Judge guided JSON decoding: enabled", flush=True)
        guided_decoding_params = GuidedDecodingParams(json=GUIDED_JSON_SCHEMA)
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_tokens,
            guided_decoding=guided_decoding_params,
        )
    else:
        print("Judge guided JSON decoding: disabled", flush=True)
        sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
    return llm, sampling_params


def judge_records(
    input_lookup: dict[str, dict],
    enriched_records: list[dict],
    output_path: Path,
    failed_output_path: Path,
    model: str,
    llm: LLM,
    sampling_params: SamplingParams,
    sample_size: int,
    seed: int,
) -> tuple[int, int, int]:
    records_with_source = []
    missing_source = []
    for enriched in enriched_records:
        source = input_lookup.get(str(enriched.get("url")))
        if source:
            records_with_source.append({"source": source, "enriched": enriched})
        else:
            missing_source.append(enriched)

    selected_records = _sample_records(records_with_source, sample_size=sample_size, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not selected_records:
        output_path.write_text("", encoding="utf-8")
        failed_output_path.write_text("", encoding="utf-8")
        print("No records available for judge step.", flush=True)
        return 0, 0, len(missing_source)

    conversations = [
        [
            {"role": "system", "content": "You are a strict but fair data quality auditor. Return valid JSON only."},
            {"role": "user", "content": _prompt(record["source"], record["enriched"])},
        ]
        for record in selected_records
    ]

    print(f"Running judge on {len(selected_records)} labels", flush=True)
    outputs = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=True)

    success_count = 0
    failure_count = len(missing_source)
    with output_path.open("w", encoding="utf-8") as dst, failed_output_path.open("w", encoding="utf-8") as failed_dst:
        for missing in missing_source:
            failed_dst.write(
                json.dumps(
                    {
                        "url": missing.get("url"),
                        "title": missing.get("title"),
                        "error": "Original input article text was not found for judge step.",
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

        for record, output in zip(selected_records, outputs, strict=True):
            enriched = record["enriched"]
            raw_text = output.outputs[0].text
            try:
                parsed = _normalise_judgement(_json_from_text(raw_text))
                judged = {
                    "url": enriched.get("url"),
                    "title": enriched.get("title"),
                    "provider": enriched.get("provider", record["source"].get("provider")),
                    "source": enriched.get("source"),
                    "published_at": enriched.get("published_at"),
                    "month": enriched.get("month"),
                    "enrichment_version": enriched.get("enrichment_version"),
                    "judge_backend": "aau_ai_lab_vllm",
                    "judge_model": model,
                    "judge_version": f"aau_ai_lab_vllm_{model}_{JUDGE_VERSION}",
                    "judged_at": datetime.now(UTC).isoformat(),
                    "original_is_relevant": enriched.get("is_relevant"),
                    "original_relevance_score": enriched.get("relevance_score"),
                    "original_sentiment_score": enriched.get("sentiment_score", enriched.get("sentiment")),
                    "original_price_pressure_direction": enriched.get("price_pressure_direction"),
                    "original_price_pressure_strength": enriched.get("price_pressure_strength"),
                    "original_primary_topic": enriched.get("primary_topic"),
                    **parsed,
                }
                dst.write(json.dumps(judged, ensure_ascii=True) + "\n")
                success_count += 1
            except Exception as exc:
                failure_count += 1
                failed_dst.write(
                    json.dumps(
                        {
                            "url": enriched.get("url"),
                            "title": enriched.get("title"),
                            "error": str(exc),
                            "raw_model_output": raw_text,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

    return len(selected_records), success_count, failure_count


def judge_enrichment(
    input_dir: str,
    enriched_dir: str,
    output_dir: str,
    model: str,
    max_tokens: int,
    max_model_len: int | None,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    trust_remote_code: bool,
    sample_size: int,
    seed: int,
    tokenizer_mode: str | None,
    config_format: str | None,
    load_format: str | None,
    guided_json: bool,
    input_path: str | None = None,
    enriched_path: str | None = None,
) -> None:
    input_files = _input_files(input_dir=input_dir, input_path=input_path)
    enriched_files = _enriched_files(enriched_dir=enriched_dir, enriched_path=enriched_path)
    if not input_files:
        raise FileNotFoundError(f"No judge input articles found in {input_dir}")
    if not enriched_files:
        raise FileNotFoundError(f"No enrichment outputs found in {enriched_dir}")

    input_lookup = _build_input_lookup(input_files)
    llm, sampling_params = _build_llm(
        model=model,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=trust_remote_code,
        tokenizer_mode=tokenizer_mode,
        config_format=config_format,
        load_format=load_format,
        guided_json=guided_json,
    )

    total_sampled = 0
    total_success = 0
    total_failures = 0
    output_root = Path(output_dir)
    for enriched_file in enriched_files:
        suffix = _batch_suffix(enriched_file)
        enriched_records = _load_jsonl(enriched_file)
        sampled, success, failures = judge_records(
            input_lookup=input_lookup,
            enriched_records=enriched_records,
            output_path=output_root / f"news_judged_{suffix}.jsonl",
            failed_output_path=output_root / f"failed_judgements_{suffix}.jsonl",
            model=model,
            llm=llm,
            sampling_params=sampling_params,
            sample_size=sample_size,
            seed=seed,
        )
        total_sampled += sampled
        total_success += success
        total_failures += failures

    print(
        f"Finished judge step. sampled={total_sampled}, success={total_success}, failures={total_failures}",
        flush=True,
    )
    if total_success == 0 and total_sampled:
        raise RuntimeError("All judge calls failed. Check model output and logs.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an AI-LAB LLM-as-judge quality audit for enrichment labels.")
    parser.add_argument("--input-dir", default="inputs")
    parser.add_argument("--input")
    parser.add_argument("--enriched-dir", default="results")
    parser.add_argument("--enriched")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--model",
        default=os.getenv("AI_LAB_JUDGE_MODEL", "mistralai/Mistral-Small-24B-Instruct-2501"),
    )
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("AI_LAB_JUDGE_MAX_TOKENS", "500")))
    parser.add_argument("--max-model-len", type=int, default=int(os.getenv("AI_LAB_JUDGE_MAX_MODEL_LEN", "4096")))
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.getenv("AI_LAB_JUDGE_GPU_MEMORY_UTILIZATION", os.getenv("AI_LAB_GPU_MEMORY_UTILIZATION", "0.85"))),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=int(os.getenv("AI_LAB_JUDGE_TENSOR_PARALLEL_SIZE", os.getenv("AI_LAB_TENSOR_PARALLEL_SIZE", "1"))),
    )
    parser.add_argument("--sample-size", type=int, default=int(os.getenv("AI_LAB_JUDGE_SAMPLE_SIZE", "500")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("AI_LAB_JUDGE_SEED", "42")))
    parser.add_argument("--tokenizer-mode", default=os.getenv("AI_LAB_JUDGE_TOKENIZER_MODE", "mistral"))
    parser.add_argument("--config-format", default=os.getenv("AI_LAB_JUDGE_CONFIG_FORMAT", "mistral"))
    parser.add_argument("--load-format", default=os.getenv("AI_LAB_JUDGE_LOAD_FORMAT", "mistral"))
    parser.add_argument(
        "--guided-json",
        action="store_true",
        help="Use vLLM guided JSON decoding. Disabled by default because it can fail with Mistral tokenizers.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    judge_enrichment(
        input_dir=args.input_dir,
        input_path=args.input,
        enriched_dir=args.enriched_dir,
        enriched_path=args.enriched,
        output_dir=args.output_dir,
        model=args.model,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        sample_size=args.sample_size,
        seed=args.seed,
        tokenizer_mode=args.tokenizer_mode,
        config_format=args.config_format,
        load_format=args.load_format,
        guided_json=args.guided_json,
    )


if __name__ == "__main__":
    main()
