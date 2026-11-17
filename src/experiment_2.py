
from pathlib import Path
import json

from src.client import measure_request


# Корневая папка проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Папка с подготовленными запросами
PROMPTS_DIR = PROJECT_ROOT / "data" / "prompts"

# Экспериментальные параметры
PROMPT_LENGTH = 1024
OUTPUT_LENGTHS = [256, 1024]
REPEATS = 5

TEMPERATURE = 0


def load_prompt():
    """Загрузка запроса длиной 1024 токена."""

    prompt_path = PROMPTS_DIR / f"prompt_{PROMPT_LENGTH}.json"

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Не найден файл: {prompt_path}"
        )

    with open(prompt_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def run_warmup(data):
    """Прогрев модели перед измерениями."""

    print("\nВыполняется прогрев модели...")

    result = measure_request(
        messages=[
            {
                "role": "user",
                "content": data["prompt"]
            }
        ],
        condition="experiment_2_warmup",
        max_tokens=16,
        temperature=TEMPERATURE
    )

    if result["status"] != "success":
        raise RuntimeError(
            f"Ошибка прогрева: {result.get('error')}"
        )

    print("Прогрев завершён.")


def run_experiments():
    """Эксперимент 2: влияние длины ответа на время генерации."""

    data = load_prompt()

    print("=" * 55)
    print("ЭКСПЕРИМЕНТ 2. ДЛИНА ОТВЕТА И СКОРОСТЬ ГЕНЕРАЦИИ")
    print("=" * 55)

    print(f"Длина входного запроса: {PROMPT_LENGTH}")
    print(f"Длины ответов: {OUTPUT_LENGTHS}")
    print(f"Количество повторов: {REPEATS}")

    # Прогрев
    run_warmup(data)

    print("\nНачало измерений.")

    for output_length in OUTPUT_LENGTHS:

        print("\n" + "=" * 55)
        print(f"Целевая длина ответа: {output_length} токенов")
        print("=" * 55)

        for repeat in range(1, REPEATS + 1):

            condition = (
                f"response_{output_length}_repeat_{repeat}"
            )

            print(
                f"\nИзмерение {repeat}/{REPEATS}"
            )

            result = measure_request(
                messages=[
                    {
                        "role": "user",
                        "content": data["prompt"]
                    }
                ],
                condition=condition,
                max_tokens=output_length,
                temperature=TEMPERATURE,
                request_id=(
                    data["id"]
                    + f"_response_{output_length}_{repeat}"
                )
            )

            if result["status"] != "success":
                print(
                    f"Ошибка запроса: {result.get('error')}"
                )
                continue

            print(
                f"Входные токены: {result['input_tokens']}"
            )

            print(
                f"Выходные токены: {result['output_tokens']}"
            )

            print(
                f"TTFT: {result['ttft_ms']:.2f} мс"
            )

            print(
                f"TPOT: {result['tpot_ms']:.2f} мс"
                if result["tpot_ms"] is not None
                else "TPOT: не рассчитан"
            )

            print(
                f"Полное время: {result['total_ms']:.2f} мс"
            )

            print(
                f"Причина завершения: {result['finish_reason']}"
            )

            if result["output_tokens"] != output_length:
                print(
                    "ВНИМАНИЕ: фактическое число токенов "
                    "не совпало с заданным!"
                )

    print("\nЭксперимент 2 завершён.")
    print("Результаты записаны в results/results.jsonl")


if __name__ == "__main__":
    run_experiments()
