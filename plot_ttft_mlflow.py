import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # без окон, только сохранение в файл
import matplotlib.pyplot as plt
import numpy as np
import mlflow

# mlruns создастся рядом со скриптом, тогда обычный `mlflow ui --port 5000`
# запущенный из этой папки, всё увидит
os.chdir(Path(__file__).resolve().parent)
mlflow.set_experiment("lab1")

# ---------------- Данные ----------------
# 1. TTFT от длины запроса
prompt_len = [128, 512, 1024, 2048, 4096]
ttft = [5.671, 27.219, 71.712, 238.972, 882.635]

# 2. Полное время от длины ответа
out_len = [16, 64, 256, 1024]
total_time = [94.24, 110.44, 117.36, 303.13]
SLOPE, INTERCEPT = 0.2082, 85.57

# 3. Пропускная способность и задержка при разном числе клиентов
clients = [1, 2, 4]
p90_ttft = [89.77, 204.03, 437.75]
tok_s = [1.329, 1.227, 1.209]


# ---------------- Эксперимент 1 ----------------
with mlflow.start_run(run_name="experiment_1_ttft_vs_prompt"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(prompt_len, ttft, "o-", color="tab:blue")
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