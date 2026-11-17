
import json
from pathlib import Path

from src.client import measure_request


# Корневая папка проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Папка с подготовленными запросами
PROMPTS_DIR = PROJECT_ROOT / "data" / "prompts"

# Размеры входных запросов
TOKEN_LENGTHS = [2048, 4096]

# Количество повторений
REPEATS = 5

# Параметры генерации
MAX_TOKENS = 16
TEMPERATURE = 0


def load_prompt(length):
    """Загрузка подготовленного запроса."""

    prompt_path = PROMPTS_DIR / f"prompt_{length}.json"

    with open(prompt_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def run_warmup():
    """Прогрев модели перед измерениями."""

    print("\nВыполняется прогрев модели...")

    data = load_prompt(128)

    result = measure_request(
        messages=[
            {
                "role": "user",
                "content": data["prompt"]
            }
        ],
        condition="warmup",
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE
    )

    if result["status"] != "success":
        raise RuntimeError(
            f"Ошибка прогрева: {result.get('error')}"
        )

    print("Прогрев успешно завершён.")


def run_experiments():
    """Проведение основной серии измерений."""

    # Проверяем наличие всех подготовленных файлов
    for length in TOKEN_LENGTHS:
        path = PROMPTS_DIR / f"prompt_{length}.json"

        if not path.exists():
            raise FileNotFoundError(
                f"Не найден файл: {path}"
            )

    # Прогрев
    run_warmup()

    print("\nНачало экспериментальной серии.")

    for length in TOKEN_LENGTHS:

        data = load_prompt(length)

        print("\n" + "=" * 45)
        print(f"Целевая длина: {length} токенов")
        print("=" * 45)

        for repeat in range(1, REPEATS + 1):

            print(f"\nИзмерение {repeat}/{REPEATS}")

            result = measure_request(
                messages=[
                    {
                        "role": "user",
                        "content": data["prompt"]
                    }
                ],
                condition=f"book_{length}_repeat_{repeat}",
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                request_id=data["id"] + f"_{repeat}"
            )

            if result["status"] != "success":
                print(f"Ошибка запроса: {result['error']}")
                continue

            print(
                f"Целевая длина: {length}"
            )

            print(
                f"Фактические входные токены: "
                f"{result['input_tokens']}"
            )

            print(
                f"Выходные токены: "
                f"{result['output_tokens']}"
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
                f"Общее время: {result['total_ms']:.2f} мс"
            )

    print("\nЭкспериментальная серия завершена!")
    print("Результаты сохранены в results/results.jsonl")
    print("Метрики зарегистрированы в MLflow.")


if __name__ == "__main__":
    run_experiments()
