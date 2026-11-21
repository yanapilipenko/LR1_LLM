
import json
import time
import uuid
import asyncio
import statistics
import math
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

import yaml
from openai import AsyncOpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "endpoint.yaml"
PROMPTS_DIR = PROJECT_ROOT / "data" / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"

# Уровни нагрузки для CPU
CLIENT_LEVELS = [1, 2, 4]

PROMPT_LENGTH = 1024
OUTPUT_LENGTH = 128

TEMPERATURE = 0
MIN_REQUESTS = 16

# Экспериментальные ограничения для CPU
TTFT_LIMIT_MS = 120000
TPOT_LIMIT_MS = 500


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_prompts():
    """Загрузка 16 разных запросов."""

    prompts = []

    paths = sorted(
        PROMPTS_DIR.glob("prompt_1024_load_*.json")
    )

    for path in paths:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if data["prompt_tokens"] != PROMPT_LENGTH:
            raise ValueError(
                f"Неверная длина запроса: {path.name}"
            )

        prompts.append(data["prompt"])

    if len(prompts) != MIN_REQUESTS:
        raise ValueError(
            f"Ожидалось 16 запросов, найдено {len(prompts)}"
        )

    return prompts


def save_jsonl(filename, record):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    path = RESULTS_DIR / filename

    with open(path, "a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def get_metrics_url(config):
    """Получение адреса /metrics из адреса API."""

    server_url = config["server"]["base_url"].rstrip("/")

    if server_url.endswith("/v1"):
        server_url = server_url[:-3]

    return server_url + "/metrics"


def collect_metrics(metrics_url, clients_count, stage):
    """Сохранение метрик сервера до или после нагрузки."""

    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        request = urllib.request.Request(
            metrics_url,
            headers={"Accept": "text/plain"}
        )

        with urllib.request.urlopen(
            request, timeout=5
        ) as response:
            metrics_text = response.read().decode("utf-8")

        available = True
        error = None

    except Exception as exc:
        metrics_text = ""
        available = False
        error = str(exc)

    # Извлекаем показатели vLLM, если сервер их предоставляет.
    vllm_metrics = {}

    for line in metrics_text.splitlines():
        if line.startswith("#") or not line.strip():
            continue

        name = line.split()[0].split("{")[0]

        if name in (
            "vllm:num_requests_waiting",
            "vllm:kv_cache_usage_perc"
        ):
            try:
                vllm_metrics[name] = float(line.split()[-1])
            except (ValueError, IndexError):
                pass

    record = {
        "condition": "experiment_3_metrics",
        "timestamp": timestamp,
        "clients": clients_count,
        "stage": stage,
        "metrics_url": metrics_url,
        "available": available,
        "error": error,
        "vllm_metrics": vllm_metrics,
        "raw_metrics": metrics_text
    }

    save_jsonl("experiment_3_metrics.jsonl", record)

    print(f"\nМЕТРИКИ {stage.upper()}:")

    if available:
        print(metrics_text if metrics_text else "Метрики пустые.")
    else:
        print("Метрики недоступны:", error)

    return record


async def measure_request(client, model, prompt, request_id):

    start = time.perf_counter()

    first_token_time = None
    last_token_time = None
    response_tokens = 0

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=OUTPUT_LENGTH,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"ignore_eos": True}
        )

        async for chunk in stream:
            now = time.perf_counter()

            if chunk.choices:
                content = chunk.choices[0].delta.content

                if content:
                    if first_token_time is None:
                        first_token_time = now

                    last_token_time = now

            if chunk.usage is not None:
                response_tokens = chunk.usage.completion_tokens

        finish = time.perf_counter()

        ttft = (
            (first_token_time - start) * 1000
            if first_token_time else None
        )

        tpot = (
            (last_token_time - first_token_time) * 1000
            / (response_tokens - 1)
            if first_token_time and response_tokens > 1
            else None
        )

        return {
            "id": request_id,
            "input_tokens": PROMPT_LENGTH,
            "output_tokens": response_tokens,
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "total_ms": (finish - start) * 1000,
            "status": "success"
        }

    except Exception as error:
        return {
            "id": request_id,
            "status": "error",
            "error": str(error)
        }


async def client_worker(
    client_id,
    client,
    model,
    prompts,
    total_requests,
    clients_count,
    results
):
    """
    Каждый клиент последовательно отправляет свои запросы.
    Общее количество запросов распределяется между клиентами.
    """

    for request_index in range(
        client_id, total_requests, clients_count
    ):

        prompt = prompts[request_index]

        result = await measure_request(
            client,
            model,
            prompt,
            str(uuid.uuid4())
        )

        result["client_id"] = client_id + 1
        result["request_number"] = request_index + 1

        results.append(result)

        print(
            f"Клиент {client_id + 1}: "
            f"запрос {request_index + 1}/{total_requests}, "
            f"TTFT={result.get('ttft_ms')}, "
            f"TPOT={result.get('tpot_ms')}, "
            f"статус={result['status']}"
        )


async def run_level(config, prompts, clients_count):

    total_requests = max(
        MIN_REQUESTS, 4 * clients_count
    )

    print("\n" + "=" * 60)
    print(f"КЛИЕНТОВ: {clients_count}")
    print(f"ВСЕГО ЗАПРОСОВ: {total_requests}")
    print("=" * 60)

    client_config = config["server"]

    client = AsyncOpenAI(
        base_url=client_config["base_url"],
        api_key=client_config["api_key"],
        timeout=1800
    )

    metrics_url = get_metrics_url(config)

    # Метрики перед нагрузкой
    metrics_before = collect_metrics(
        metrics_url,
        clients_count,
        "before"
    )

    results = []

    start = time.perf_counter()

    await asyncio.gather(*[
        client_worker(
            i,
            client,
            config["model"]["name"],
            prompts,
            total_requests,
            clients_count,
            results
        )
        for i in range(clients_count)
    ])

    elapsed = time.perf_counter() - start

    # Метрики после нагрузки
    metrics_after = collect_metrics(
        metrics_url,
        clients_count,
        "after"
    )

    successful = [
        r for r in results
        if r["status"] == "success"
    ]

    ttft_values = [
        r["ttft_ms"] for r in successful
        if r.get("ttft_ms") is not None
    ]

    tpot_values = [
        r["tpot_ms"] for r in successful
        if r.get("tpot_ms") is not None
    ]

    output_tokens = sum(
        r.get("output_tokens", 0)
        for r in successful
    )

    good_requests = [
        r for r in successful
        if r.get("ttft_ms") is not None
        and r.get("tpot_ms") is not None
        and r["ttft_ms"] <= TTFT_LIMIT_MS
        and r["tpot_ms"] <= TPOT_LIMIT_MS
    ]

    summary = {
        "condition": "experiment_3_summary",
        "timestamp": datetime.now(timezone.utc).isoformat(),

        "clients": clients_count,
        "total_requests": total_requests,

        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),

        "elapsed_seconds": elapsed,

        "output_tokens_per_second": (
            output_tokens / elapsed if elapsed else 0
        ),

        "requests_per_second": (
            len(successful) / elapsed if elapsed else 0
        ),

        "median_ttft_ms": (
            statistics.median(ttft_values)
            if ttft_values else None
        ),

        "p90_ttft_ms": (
            sorted(ttft_values)[
                math.ceil(0.9 * len(ttft_values)) - 1
            ] if ttft_values else None
        ),

        "median_tpot_ms": (
            statistics.median(tpot_values)
            if tpot_values else None
        ),

        "good_requests": len(good_requests),

        "goodput_requests_per_second": (
            len(good_requests) / elapsed if elapsed else 0
        ),

        "goodput_percent": (
            len(good_requests) / len(results) * 100
            if results else 0
        ),

        "ttft_limit_ms": TTFT_LIMIT_MS,
        "tpot_limit_ms": TPOT_LIMIT_MS,

        "metrics_before_available": metrics_before["available"],
        "metrics_after_available": metrics_after["available"],

        "vllm_metrics_before": metrics_before["vllm_metrics"],
        "vllm_metrics_after": metrics_after["vllm_metrics"]
    }

    print("\nИТОГ УРОВНЯ:")

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=4
        )
    )

    for result in results:
        result["clients"] = clients_count
        save_jsonl("experiment_3.jsonl", result)

    save_jsonl("experiment_3.jsonl", summary)

    await client.close()

    return summary


async def main():
    config = load_config()
    prompts = load_prompts()
    for clients in CLIENT_LEVELS:
        summary = await run_level(config, prompts, clients)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
