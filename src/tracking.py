
import mlflow
import json

from pathlib import Path


# Корневая папка проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Локальное хранилище MLflow
MLFLOW_PATH = PROJECT_ROOT / "mlflow.db"

# Название эксперимента
EXPERIMENT_NAME = "LR1_LLM_Performance"


def setup_mlflow():
    """Настройка локального хранилища MLflow."""

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_PATH.as_posix()}")

    experiment = mlflow.set_experiment(EXPERIMENT_NAME)

    return experiment


def log_record(record, config):
    """
    Сохранение одного измерения в MLflow.

    Сохраняются:
    - параметры модели и генерации;
    - метрики производительности;
    - статус запроса;
    - полный JSON с результатом;
    - текст ответа.
    """

    setup_mlflow()

    generation = config["generation"]
    model = config["model"]

    run_name = (
        f'{record["condition"]}_'
        f'{record["id"][:8]}'
    )

    with mlflow.start_run(run_name=run_name):

        # Параметры эксперимента
        params = {
            "condition": record["condition"],
            "model": model["name"],
            "format": model.get("format", "unknown"),
            "quantization": model.get("quantization", "unknown"),
            "temperature": generation.get("temperature", 0),
            "max_tokens": generation.get("max_tokens", 16),
            "ignore_eos": generation.get("ignore_eos", True),
            "stream": generation.get("stream", True),
            "base_url": config["server"]["base_url"]
        }

        mlflow.log_params({
            key: str(value)
            for key, value in params.items()
        })

        # Числовые метрики
        metrics = {
            "input_tokens": record.get("input_tokens"),
            "output_tokens": record.get("output_tokens"),
            "ttft_ms": record.get("ttft_ms"),
            "tpot_ms": record.get("tpot_ms"),
            "total_ms": record.get("total_ms"),
            "fragment_count": len(
                record.get("fragment_times_ms", [])
            )
        }

        for name, value in metrics.items():

            if value is not None:
                mlflow.log_metric(name, float(value))

        # Дополнительная информация
        mlflow.set_tags({
            "request_id": record["id"],
            "status": record["status"],
            "finish_reason": str(record.get("finish_reason")),
            "error": str(record.get("error"))
        })

        # Полная запись измерения
        mlflow.log_dict(
            record,
            "artifacts/record.json"
        )

        # Текст ответа
        mlflow.log_text(
            record.get("response", ""),
            "artifacts/response.txt"
        )

        print(
            f'MLflow: результат {record["id"]} сохранён.'
        )
