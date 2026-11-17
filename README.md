Для запуска интерфейса MLflow используется команда:

mlflow ui --backend-store-uri sqlite:///mlflow.db

После запуска интерфейс доступен в браузере:

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


# 4 эксперимент (4 четыре параллельных слота)
# -c 20480 — общий размер контекста, рассчитанный как 4 × 5120
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

  python .\src\prompt_builder.py     #запуск эксперимента 3