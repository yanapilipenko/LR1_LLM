# LR1. Инференс и развёртывание языковой модели

[Отчёт](reports/LR1.pdf) · [Паспорт сервера](server_passport.md)

Сохранённые измерения находятся в results/, база экспериментов — в mlflow.db. Для просмотра результатов модель не требуется. Для нового прогона положите GGUF из паспорта в models/, а сборку llama.cpp b12359 для Windows CPU — в tools/llama-cpu/.

Для запуска интерфейса MLflow используется команда:
python plot_ttft_mlflow.py #графики
mlflow ui --backend-store-uri sqlite:///mlflow.db

http://127.0.0.1:5000


-----------------------------------------------------------------------
& ".\tools\llama-cpu\llama-server.exe" `
  -m ".\models\Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf" `
  -c 5120 `
  -t 2 `
  -ngl 0 `
  -np 1 `
  -b 64 `
  -ub 32 `
  --no-cache-prompt `
  --host 127.0.0.1 `
  --port 8080
   
   //без кеширования

   python -m src.experiment_runner     #запуск эксперимента 1 
   python -m src.experiment_2          #запуск эксперимента 2


# 3 эксперимент (4 четыре параллельных слота)
# -c 5120 при четырёх равных слотах даёт по 1280 токенов
& ".\tools\llama-cpu\llama-server.exe" `
  -m ".\models\Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf" `
  -c 5120 `
  -t 2 `
  -ngl 0 `
  -np 4 `
  -b 64 `
  -ub 32 `
  --no-cache-prompt `
  --host 127.0.0.1

  python -m src.prompt_builder       # подготовка запросов
  python -m src.experiment_3         # нагрузка 1, 2 и 4 клиента
