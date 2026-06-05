from __future__ import annotations

import argparse
import ast
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


SCHEMA_KEYS = [
    "is_relevant",
    "relevance_score",
    "sentiment_score",
    "price_pressure_direction",
    "price_pressure_strength",
    "primary_topic",
    "reason_short",
]

PROMPT_VERSION = "v2"
EMPTY_MODEL_OUTPUT = "<empty model output>"

GUIDED_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "is_relevant": {"type": "boolean"},
        "relevance_score": {"type": "number", "minimum": 0, "maximum": 1},
        "sentiment_score": {"type": "number", "minimum": -1, "maximum": 1},
        "price_pressure_direction": {"type": "string", "enum": ["upward", "downward", "neutral"]},
        "price_pressure_strength": {"type": "integer", "minimum": 0, "maximum": 5},
        "primary_topic": {
            "type": "string",
            "enum": ["shortage", "tariff", "demand", "supply_chain", "export_controls", "oversupply", "other"],
        },
        "reason_short": {"type": "string"},
    },
    "required": SCHEMA_KEYS,
    "additionalProperties": False,
}


def _prompt(record: dict) -> str:
    return (
        "Classify this article for an electronics and semiconductor price-pressure model.\n"
        "Return JSON only with these keys: "
        + ", ".join(SCHEMA_KEYS)
        + ".\n"
        "Strict relevance rules:\n"
        "- is_relevant must be true only if the article directly concerns semiconductor/electronics component prices, supply, demand, tariffs, export controls, inventories, production capacity, foundries, chips, wafers, memory, or electronics manufacturing.\n"
        "- Mark is_relevant=false for food chips, poker chips, sports/health microchips, microchipped mouthguards, restaurants, entertainment, generic politics, or generic business news unless the article clearly links to electronics/semiconductor markets.\n"
        "- If unsure, choose is_relevant=false.\n"
        "- relevance_score must be a number from 0 to 1.\n"
        "- sentiment_score must be a number from -1 to 1, where negative means worse supply/price pressure news.\n"
        "- price_pressure_direction must be upward, downward, or neutral.\n"
        "- upward means likely higher semiconductor/electronics prices: shortages, tariffs, export controls, production delays, factory shutdowns, logistics bottlenecks, geopolitical supply risk, or demand exceeding supply.\n"
        "- downward means likely lower semiconductor/electronics prices: oversupply, weak demand, falling prices, excess inventories, or new capacity easing scarcity.\n"
        "- neutral means relevant background with no clear price-pressure direction.\n"
        "- price_pressure_strength must be an integer from 0 to 5. Use 0 when direction is neutral or is_relevant=false.\n"
        "- primary_topic must be shortage, tariff, demand, supply_chain, export_controls, oversupply, or other.\n"
        "- For is_relevant=false, return relevance_score=0, sentiment_score=0, price_pressure_direction=neutral, price_pressure_strength=0, primary_topic=other.\n"
        "- Do not treat generic negative news as upward pressure unless it affects electronics/semiconductor markets.\n\n"
        "Examples:\n"
        "- 'Semiconductor shortage disrupts car production' => relevant, upward, shortage.\n"
        "- 'New chip fabs increase supply and inventories rise' => relevant, downward, oversupply.\n"
        "- 'Fish and chips restaurant opens' => not relevant.\n"
        "- 'Microchipped mouthguards used in rugby' => not relevant.\n\n"
        f"Title: {record.get('title', '')}\n"
        f"Source: {record.get('source', '')}\n"
        f"Published at: {record.get('published_at', '')}\n"
        f"Article text:\n{record.get('text', '')}"
    )


def _retry_prompt(record: dict, previous_output: str, previous_error: str) -> str:
    raw_output = previous_output.strip() or EMPTY_MODEL_OUTPUT
    return (
        "The previous classification attempt did not return parseable JSON.\n"
        "Classify the article again and return one compact JSON object only.\n"
        "Do not use markdown. Do not explain outside JSON. The first character must be { and the last character must be }.\n"
        "Required JSON keys: "
        + ", ".join(SCHEMA_KEYS)
        + ".\n"
        "Allowed values:\n"
        "- is_relevant: true or false.\n"
        "- relevance_score: number from 0 to 1.\n"
        "- sentiment_score: number from -1 to 1.\n"
        "- price_pressure_direction: upward, downward, or neutral.\n"
        "- price_pressure_strength: integer from 0 to 5.\n"
        "- primary_topic: shortage, tariff, demand, supply_chain, export_controls, oversupply, or other.\n"
        "- reason_short: short string.\n\n"
        "If the article is not clearly about semiconductor/electronics price pressure, return:\n"
        '{"is_relevant":false,"relevance_score":0,"sentiment_score":0,'
        '"price_pressure_direction":"neutral","price_pressure_strength":0,'
        '"primary_topic":"other","reason_short":"Not directly related to semiconductor or electronics price pressure."}\n\n'
        f"Previous parser error: {previous_error}\n"
        f"Previous model output: {raw_output[:1000]}\n\n"
        f"Title: {record.get('title', '')}\n"
        f"Source: {record.get('source', '')}\n"
        f"Published at: {record.get('published_at', '')}\n"
        f"Article text:\n{record.get('text', '')}"
    )


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


def _normalise_label(parsed: dict) -> dict:
    is_relevant = _as_bool(parsed.get("is_relevant", False))

    direction = str(parsed.get("price_pressure_direction", "neutral")).lower()
    if direction not in {"upward", "downward", "neutral"}:
        direction = "neutral"

    topic = str(parsed.get("primary_topic", "other")).lower()
    if topic not in {"shortage", "tariff", "demand", "supply_chain", "export_controls", "oversupply", "other"}:
        topic = "other"

    strength = int(float(parsed.get("price_pressure_strength", 0) or 0))
    strength = max(0, min(5, strength))

    relevance_score = float(parsed.get("relevance_score", 0) or 0)
    relevance_score = max(0.0, min(1.0, relevance_score))

    sentiment_score = float(parsed.get("sentiment_score", 0) or 0)
    sentiment_score = max(-1.0, min(1.0, sentiment_score))

    if not is_relevant:
        relevance_score = 0.0
        sentiment_score = 0.0
        direction = "neutral"
        strength = 0
        topic = "other"
    elif direction == "neutral":
        strength = 0
    elif strength == 0:
        strength = 1

    if is_relevant and topic in {"shortage", "tariff", "supply_chain", "export_controls"} and direction == "downward":
        direction = "upward"
    if is_relevant and topic == "oversupply" and direction == "upward":
        direction = "downward"

    return {
        "is_relevant": is_relevant,
        "relevance_score": relevance_score,
        "sentiment_score": sentiment_score,
        "price_pressure_direction": direction,
        "price_pressure_strength": strength,
        "primary_topic": topic,
        "reason_short": str(parsed.get("reason_short", ""))[:500],
    }


def _load_records(input_path: str) -> list[dict]:
    records = []
    with Path(input_path).open("r", encoding="utf-8") as src:
        for line in src:
            if line.strip():
                records.append(json.loads(line))
    return records


def _build_llm(
    model: str,
    max_tokens: int,
    max_model_len: int | None,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    trust_remote_code: bool,
) -> tuple[LLM, SamplingParams, SamplingParams]:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HF_TOKEN", "")

    llm_kwargs = {
        "model": model,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": trust_remote_code,
    }
    if max_model_len:
        llm_kwargs["max_model_len"] = max_model_len

    print(f"Loading model: {model}", flush=True)
    llm = LLM(**llm_kwargs)
    guided_decoding_params = GuidedDecodingParams(json=GUIDED_JSON_SCHEMA)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        guided_decoding=guided_decoding_params,
    )
    fallback_sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
    )
    return llm, sampling_params, fallback_sampling_params


def _write_record(dst, source_record: dict, parsed: dict, model: str) -> None:
    output = {
        "url": source_record.get("url"),
        "title": source_record.get("title"),
        "provider": source_record.get("provider"),
        "source": source_record.get("source"),
        "published_at": source_record.get("published_at"),
        "month": source_record.get("month"),
        "content_char_count": source_record.get("content_char_count", 0),
        "scored_char_count": len(source_record.get("text", "")),
        "ai_backend": "aau_ai_lab_vllm",
        "enrichment_version": f"aau_ai_lab_vllm_{model}_{PROMPT_VERSION}",
        "enriched_at": datetime.now(UTC).isoformat(),
        **_normalise_label(parsed),
    }
    dst.write(json.dumps(output, ensure_ascii=True) + "\n")
    dst.flush()


def _enrich_records(
    records: list[dict],
    batch_label: str,
    output_path: str,
    failed_output_path: str,
    model: str,
    llm: LLM,
    sampling_params: SamplingParams,
    fallback_sampling_params: SamplingParams,
) -> tuple[int, int, int]:
    conversations = [
        [
            {"role": "system", "content": "You are a precise data labeling assistant. Return valid JSON only."},
            {"role": "user", "content": _prompt(record)},
        ]
        for record in records
    ]

    success_count = 0
    failure_count = 0
    output_file = Path(output_path)
    failed_output_file = Path(failed_output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    failed_output_file.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        output_file.write_text("", encoding="utf-8")
        failed_output_file.write_text("", encoding="utf-8")
        print(f"Skipping empty batch: {batch_label}", flush=True)
        return 0, 0, 0

    print(f"Running enrichment for {batch_label}: {len(records)} articles", flush=True)
    outputs = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=True)

    with output_file.open("w", encoding="utf-8") as dst, failed_output_file.open("w", encoding="utf-8") as failed_dst:
        for record, output in zip(records, outputs, strict=True):
            title = record.get("title", "")[:100]
            text = output.outputs[0].text
            try:
                parsed = _json_from_text(text)
                _write_record(dst, record, parsed, model)
                success_count += 1
                print(f"OK: {title}", flush=True)
            except Exception as exc:
                retry_text = ""
                retry_conversation = [
                    {
                        "role": "system",
                        "content": "You are a precise data labeling assistant. Return valid JSON only.",
                    },
                    {"role": "user", "content": _retry_prompt(record, text, str(exc))},
                ]
                try:
                    retry_outputs = llm.chat(
                        [retry_conversation],
                        sampling_params=fallback_sampling_params,
                        use_tqdm=False,
                    )
                    retry_text = retry_outputs[0].outputs[0].text
                    parsed = _json_from_text(retry_text)
                    _write_record(dst, record, parsed, model)
                    success_count += 1
                    print(f"OK after retry: {title}", flush=True)
                except Exception as retry_exc:
                    failure_count += 1
                    failed_dst.write(
                        json.dumps(
                            {
                                "url": record.get("url"),
                                "title": record.get("title"),
                                "error": str(exc),
                                "raw_model_output": text,
                                "retry_error": str(retry_exc),
                                "retry_raw_model_output": retry_text,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
                    failed_dst.flush()
                    print(f"Failed: {title} ({exc}; retry: {retry_exc})", flush=True)

    print(
        f"Finished {batch_label}. success={success_count}, failures={failure_count}",
        flush=True,
    )
    return len(records), success_count, failure_count


def enrich_file(
    input_path: str,
    output_path: str,
    failed_output_path: str,
    model: str,
    max_tokens: int,
    max_model_len: int | None,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    trust_remote_code: bool,
) -> None:
    records = _load_records(input_path)
    llm, sampling_params, fallback_sampling_params = _build_llm(
        model=model,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=trust_remote_code,
    )
    _, success_count, failure_count = _enrich_records(
        records=records,
        batch_label=Path(input_path).name,
        output_path=output_path,
        failed_output_path=failed_output_path,
        model=model,
        llm=llm,
        sampling_params=sampling_params,
        fallback_sampling_params=fallback_sampling_params,
    )
    print(f"Finished enrichment. success={success_count}, failures={failure_count}", flush=True)
    if success_count == 0 and records:
        raise RuntimeError("All vLLM enrichment calls failed. Check model output and logs.")


def _batch_suffix(input_path: Path) -> str:
    prefix = "news_to_enrich_"
    if input_path.stem.startswith(prefix):
        return input_path.stem[len(prefix) :]
    return input_path.stem


def enrich_directory(
    input_dir: str,
    output_dir: str,
    model: str,
    max_tokens: int,
    max_model_len: int | None,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    trust_remote_code: bool,
) -> None:
    input_files = sorted(Path(input_dir).glob("news_to_enrich_*.jsonl"))
    if not input_files:
        raise FileNotFoundError(f"No batch input files found in {input_dir}")

    llm, sampling_params, fallback_sampling_params = _build_llm(
        model=model,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=trust_remote_code,
    )

    total_records = 0
    total_success = 0
    total_failures = 0
    output_root = Path(output_dir)
    for input_path in input_files:
        suffix = _batch_suffix(input_path)
        records = _load_records(str(input_path))
        record_count, success_count, failure_count = _enrich_records(
            records=records,
            batch_label=input_path.name,
            output_path=str(output_root / f"news_enriched_{suffix}.jsonl"),
            failed_output_path=str(output_root / f"failed_records_{suffix}.jsonl"),
            model=model,
            llm=llm,
            sampling_params=sampling_params,
            fallback_sampling_params=fallback_sampling_params,
        )
        total_records += record_count
        total_success += success_count
        total_failures += failure_count

    print(
        f"Finished all batches. records={total_records}, success={total_success}, failures={total_failures}",
        flush=True,
    )
    if total_success == 0 and total_records:
        raise RuntimeError("All vLLM enrichment calls failed. Check model output and logs.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI-LAB vLLM article enrichment.")
    parser.add_argument("--input", default="news_to_enrich.jsonl")
    parser.add_argument("--input-dir")
    parser.add_argument("--output", default="news_enriched.jsonl")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--failed-output", default="failed_records.jsonl")
    parser.add_argument("--model", default=os.getenv("AI_LAB_MODEL", "Qwen/Qwen2.5-32B-Instruct"))
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--max-model-len", type=int, default=int(os.getenv("AI_LAB_MAX_MODEL_LEN", "8192")))
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.getenv("AI_LAB_GPU_MEMORY_UTILIZATION", "0.85")))
    parser.add_argument("--tensor-parallel-size", type=int, default=int(os.getenv("AI_LAB_TENSOR_PARALLEL_SIZE", "1")))
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    if args.input_dir:
        enrich_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model=args.model,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        enrich_file(
            input_path=args.input,
            output_path=args.output,
            failed_output_path=args.failed_output,
            model=args.model,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=args.trust_remote_code,
        )


if __name__ == "__main__":
    main()
