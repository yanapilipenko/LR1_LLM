import csv
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path

BASE_URL = "http://127.0.0.1:8080"
INTERVAL_SEC = 0.5
OUT = Path("results/experiment_3_live_metrics.csv")

METRICS = [
    "llamacpp:requests_processing",
    "llamacpp:requests_deferred",
    "llamacpp:kv_cache_usage_ratio",
    "llamacpp:kv_cache_tokens",
    "llamacpp:n_tokens_max",
    "llamacpp:prompt_tokens_seconds",
    "llamacpp:predicted_tokens_seconds",
]

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "experiment-3-monitor"})
    with urllib.request.urlopen(req, timeout=5) as response:
        return response.read().decode("utf-8", errors="replace")

def parse_metrics(raw):
    values = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$", line)
        if match:
            values[match.group(1)] = match.group(2)
    return values

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("Мониторинг запущен. Остановить: Ctrl+C")
    print(f"Результат будет сохранён в: {OUT}")

    fields = ["timestamp"] + METRICS + ["slots_raw"]
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        try:
            while True:
                stamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
                row = {"timestamp": stamp, "slots_raw": ""}
                try:
                    raw = fetch(BASE_URL + "/metrics")
                    values = parse_metrics(raw)
                    for metric in METRICS:
                        row[metric] = values.get(metric, "")
                except Exception as e:
                    row["slots_raw"] = "metrics_error: " + str(e)

                # Сохраняем ответ /slots целиком: в разных версиях llama.cpp
                # состав полей может отличаться.
                try:
                    row["slots_raw"] = fetch(BASE_URL + "/slots").replace("\r", " ").replace("\n", " ")
                except Exception as e:
                    if not row["slots_raw"]:
                        row["slots_raw"] = "slots_error: " + str(e)

                writer.writerow(row)
                f.flush()

                deferred = row.get("llamacpp:requests_deferred", "")
                processing = row.get("llamacpp:requests_processing", "")
                kv = row.get("llamacpp:kv_cache_usage_ratio", "")
                print(f"{stamp} | processing={processing or '?'} | deferred={deferred or '?'} | KV={kv or 'метрика отсутствует'}")
                time.sleep(INTERVAL_SEC)

        except KeyboardInterrupt:
            print("\nМониторинг остановлен.")

if __name__ == "__main__":
    main()
