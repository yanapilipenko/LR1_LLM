import json
import re
import uuid
import urllib.request
from pathlib import Path


# Пути к файлам
PROJECT_DIR = Path(__file__).resolve().parent.parent
BOOK_PATH = PROJECT_DIR / "data" / "pride_and_prejudice.txt"
OUTPUT_DIR = PROJECT_DIR / "data" / "prompts"

# Адрес локального сервера llama.cpp
SERVER_URL = "http://127.0.0.1:8080"

# Требуемые размеры запросов
TOKEN_LENGTHS = [128, 512, 1024, 2048, 4096]


def read_book(path):
    """Чтение книги в кодировке UTF-8."""
    text = path.read_text(encoding="utf-8-sig")

    # Удаляем служебный заголовок Project Gutenberg
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK"
    ]

    for marker in start_markers:
        position = text.find(marker)
        if position != -1:
            end_line = text.find("\n", position)
            text = text[end_line + 1:]
            break

    # Удаляем служебное окончание
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK"
    ]

    for marker in end_markers:
        position = text.find(marker)
        if position != -1:
            text = text[:position]
            break

    return text.strip()


def tokenize(text):
    """Подсчёт токенов через токенизатор llama.cpp."""

    url = SERVER_URL + "/tokenize"

    data = json.dumps({
        "content": text,
        "add_special": True,
        "parse_special": True
    }).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))

    return len(result["tokens"])



def create_prompt(book_text, target_tokens, start_position=0):
    """Формирование запроса заданной длины из отдельного фрагмента книги."""

    label = f"REQUEST_{uuid.uuid4().hex[:12]}"

    instruction = (
        f"[{label}]\n"
        "You are an English language text analysis assistant. "
        "Read the following excerpt from Pride and Prejudice "
        "by Jane Austen. Identify its main subject and summarize "
        "the provided text.\n\n"
        "TEXT:\n"
    )

    # Берём фрагмент книги, начиная с указанной позиции
    source_text = book_text[start_position:]

    left = 0
    right = len(source_text)
    best_prompt = None
    best_count = 0

    while left <= right:
        middle = (left + right) // 2

        prompt = instruction + source_text[:middle]
        count = tokenize(prompt)

        if count <= target_tokens:
            best_prompt = prompt
            best_count = count
            left = middle + 1
        else:
            right = middle - 1

    if best_prompt is None or best_count != target_tokens:
        raise ValueError(
            f"Не удалось получить ровно {target_tokens} токенов. "
            f"Получено: {best_count}"
        )

    return {
        "id": label,
        "target_tokens": target_tokens,
        "prompt_tokens": best_count,
        "prompt": best_prompt
    }



def prepare_load_prompts(book_text):
    """Создание 16 разных запросов по 1024 токена."""

    count = 16
    length = 1024

    segment_size = len(book_text) // count

    for i in range(count):

        start_position = i * segment_size

        print(
            f"\nПодготовка нагрузочного запроса "
            f"{i + 1}/{count}..."
        )

        result = create_prompt(
            book_text,
            length,
            start_position
        )

        output_path = (
            OUTPUT_DIR / f"prompt_1024_load_{i + 1:02d}.json"
        )

        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=4),
            encoding="utf-8"
        )

        print(f"Количество токенов: {result['prompt_tokens']}")
        print(f"Файл: {output_path.name}")


def prepare_prompts():
    """Подготовка запросов всех заданных размеров."""

    print("Чтение книги...")
    book_text = read_book(BOOK_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for length in TOKEN_LENGTHS:
        print("\nПодготовка запросов для эксперимента 3...")
        prepare_load_prompts(book_text)
        print(f"\nПодготовка запроса на {length} токенов...")

        result = create_prompt(book_text, length)

        output_path = OUTPUT_DIR / f"prompt_{length}.json"

        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=4),
            encoding="utf-8"
        )

        print(f"Метка: {result['id']}")
        print(f"Количество токенов: {result['prompt_tokens']}")
        print(f"Файл сохранён: {output_path}")

    print("\nПодготовка запросов завершена!")


if __name__ == "__main__":
    prepare_prompts()