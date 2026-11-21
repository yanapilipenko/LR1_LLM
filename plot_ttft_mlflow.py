import json
import os
from statistics import median
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # без окон, только сохранение в файл
import matplotlib.pyplot as plt
import numpy as np
import mlflow

# mlruns создастся рядом со скриптом, тогда обычный `mlflow ui --port 5000`
# запущенный из этой папки, всё увидит
os.chdir(Path(__file__).resolve().parent)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("lab1")

def records(name):
    # Исторический журнал содержит две JSON-записи на одной строке.
    text = Path("results", name).read_text(encoding="utf-8-sig").strip()
    decoder = json.JSONDecoder()
    while text:
        row, end = decoder.raw_decode(text)
        yield row
        text = text[end:].lstrip()


def measured_series(name, prefix, first_condition, metric):
    rows = list(records(name))
    # Начало последней серии определяется порядком запуска, до проверки метрик.
    start = max(i for i, row in enumerate(rows) if row["condition"] == first_condition)
    groups = {}
    for row in rows[start:]:
        if row["condition"].startswith(prefix):
            length = int(row["condition"].split("_")[1])
            if row["condition"].endswith("_repeat_1"):
                groups[length] = []
            groups[length].append(row)
    for length, group in groups.items():
        if len(group) != 5 or any(row["status"] != "success" for row in group):
            raise ValueError(f"Проверьте состав серии {name}, длина {length}")
    lengths = sorted(groups)
    samples = [[row[metric] / 1000 for row in groups[length]] for length in lengths]
    return lengths, samples


prompt_len, timings = measured_series("results.jsonl", "book_", "book_128_repeat_1", "ttft_ms")
ttft = [median(values) for values in timings]
out_len, timings_out = measured_series("results_2.jsonl", "response_", "response_16_repeat_1", "total_ms")
total_time = [median(values) for values in timings_out]
SLOPE, INTERCEPT = np.polyfit(out_len, total_time, 1)
loads = [row for row in records("experiment_3.jsonl") if row.get("condition") == "experiment_3_summary"]
clients = [row["clients"] for row in loads]
p90_ttft = [row["p90_ttft_ms"] / 1000 for row in loads]
tok_s = [row["output_tokens_per_second"] for row in loads]


# ---------------- Эксперимент 1 ----------------
with mlflow.start_run(run_name="experiment_1_ttft_vs_prompt"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(prompt_len, ttft, yerr=[[m - min(v) for m, v in zip(ttft, timings)],
                 [max(v) - m for m, v in zip(ttft, timings)]], fmt="o-", capsize=4)
    for x, y in zip(prompt_len, ttft):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(-6, 7), ha="right", fontsize=9)
    ax.set_xlabel("Длина запроса, токенов")
    ax.set_ylabel("Медиана TTFT, с")
    ax.set_title("Зависимость TTFT от длины запроса")
    ax.set_xticks(prompt_len)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    mlflow.log_figure(fig, "plot1_ttft_vs_prompt_length.png")
    plt.close(fig)

    # сами значения: шаг = длина запроса
    for x, y in zip(prompt_len, ttft):
        mlflow.log_metric("median_ttft_s", y, step=x)


# ---------------- Эксперимент 2 ----------------
with mlflow.start_run(run_name="experiment_2_total_time_vs_output"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(out_len, total_time, color="tab:red", zorder=3, label="Измерения")
    xs = np.linspace(0, max(out_len) * 1.05, 100)
    ax.plot(xs, SLOPE * xs + INTERCEPT, "--", color="tab:gray",
            label=f"y = {SLOPE}x + {INTERCEPT}")
    ax.set_xlabel("Длина ответа, токенов")
    ax.set_ylabel("Медиана полного времени, с")
    ax.set_title("Зависимость полного времени ответа от длины ответа")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    mlflow.log_figure(fig, "plot2_total_time_vs_output_length.png")
    plt.close(fig)

    mlflow.log_param("fit_slope", SLOPE)
    mlflow.log_param("fit_intercept", INTERCEPT)
    for x, y in zip(out_len, total_time):
        mlflow.log_metric("median_total_time_s", y, step=x)


# ---------------- Эксперимент 3 ----------------
with mlflow.start_run(run_name="experiment_3_throughput_vs_latency"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(p90_ttft, tok_s, "o-", color="tab:green")
    for c, x, y in zip(clients, p90_ttft, tok_s):
        label = f"{c} клиент" if c == 1 else f"{c} клиента"
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9)
    ax.set_xlabel("P90 TTFT, с")
    ax.set_ylabel("Output tok/s")
    ax.set_title("Пропускная способность и задержка при разном числе клиентов")
    ax.set_ylim(min(tok_s) - 0.05, max(tok_s) + 0.08)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    mlflow.log_figure(fig, "plot3_throughput_vs_latency.png")
    plt.close(fig)

    # шаг = число клиентов
    for c, p, t in zip(clients, p90_ttft, tok_s):
        mlflow.log_metric("p90_ttft_s", p, step=c)
        mlflow.log_metric("output_tok_per_s", t, step=c)

print("Готово: 3 рана записаны в эксперимент lab1")
