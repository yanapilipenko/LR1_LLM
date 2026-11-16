
import json
import time
import uuid

from pathlib import Path
from datetime import datetime, timezone

import yaml
from openai import OpenAI
try:
    from src.tracking import log_record
except ModuleNotFoundError:
    from tracking import log_record

# Корневая папка проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Файл конфигурации сервера
CONFIG_PATH = PROJECT_ROOT / "endpoint.yaml"

# Файл для сохранения результатов
RESULTS_PATH = PROJECT_ROOT / "results" / "results.jsonl"


def load_config():
    """Загрузка конфигурации сервера из endpoint.yaml."""

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_client(config):
    """Создание клиента для подключения к llama.cpp."""

    return OpenAI(
        base_url=config["server"]["base_url"],
        api_key=config["server"]["api_key"]
    )


def save_result(record):
    """Сохранение одной записи в JSONL."""

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, "a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def measure_request(
    messages,
    condition="manual_test",
    max_tokens=None,
    temperature=None,
    request_id=None
):
    """
    Отправка потокового запроса и измерение производительности.

    Измеряются:
    - TTFT;
    - TPOT;
    - полное время;
    - количество входных и выходных токенов.
    """

    config = load_config()
    client = create_client(config)

    generation = config["generation"]

    if max_tokens is None:
        max_tokens = generation["max_tokens"]

    if temperature is None:
        temperature = generation["temperature"]

    if request_id is None:
        request_id = str(uuid.uuid4())

    # Временные отметки
    t0 = time.perf_counter()

    first_token_time = None
    last_fragment_time = None

    fragment_times = []
    response_text = ""

    prompt_tokens = None
    completion_tokens = None
    finish_reason = None

    status = "success"

    try:
        stream = client.chat.completions.create(
            model=config["model"]["name"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={
                "include_usage": True
            },
            extra_body={
                "ignore_eos": generation.get("ignore_eos", True)
            }
        )

        for chunk in stream:

            current_time = time.perf_counter()

            # Получение статистики использования токенов
            if chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens

            # Получение содержимого фрагмента
            if chunk.choices:

                choice = chunk.choices[0]

                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason

                content = choice.delta.content

                if content:

                    if first_token_time is None:
                        first_token_time = current_time

                    last_fragment_time = current_time

                    fragment_times.append(
                        current_time - t0
                    )

                    response_text += content

    except Exception as error:
        status = "error"
        error_message = str(error)

    else:
        error_message = None

    tN = time.perf_counter()

    # Расчёт метрик в миллисекундах
    total_ms = (tN - t0) * 1000

    if first_token_time is not None:
        ttft_ms = (first_token_time - t0) * 1000
    else:
        ttft_ms = None

    if (
        first_token_time is not None
        and last_fragment_time is not None
        and completion_tokens is not None
        and completion_tokens > 1
    ):
        tpot_ms = (
            (last_fragment_time - first_token_time)
            * 1000
            / (completion_tokens - 1)
        )
    else:
        tpot_ms = None

    record = {
        "id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "condition": condition,

        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,

        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "total_ms": total_ms,

        "finish_reason": finish_reason,
        "status": status,

        "response": response_text,
        "fragment_times_ms": [
            round(value * 1000, 3)
            for value in fragment_times
        ],

        "error": error_message
    }

    save_result(record)
    # Затем регистрируем измерение в MLflow
    try:
       log_record(record, config)
    except Exception as error:
       print(f"Ошибка сохранения в MLflow: {error}")

    return record


if __name__ == "__main__":

    messages = [
        {
            "role": "user",
            "content": "Привет! Кратко расскажи о себе."
        }
    ]

    result = measure_request(
        messages=messages,
        condition="first_streaming_test",
        max_tokens=16,
        temperature=0
    )

    print("\nРезультат измерения:")

    print(json.dumps(
        result,
        ensure_ascii=False,
        indent=4
    ))
