import requests
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()
api_key = "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"

# Проверка API-ключа
if not api_key:
    raise ValueError("❌ API-ключ не найден. Убедитесь, что переменная OPENROUTER_API_KEY задана в .env.")

# История диалога
chat_history = [
    {
        "role": "system",
        "content": "Ты отвечаешь только на русском языке. Ответы всегда краткие, понятные и строго по сути. Не используй английский язык."
    }
]

def get_chatgpt_response(prompt, model="openai/gpt-3.5-turbo"):
    if not prompt or not isinstance(prompt, str):
        return "❌ Пустой или некорректный запрос."

    chat_history.append({"role": "user", "content": prompt})

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": chat_history,
            },
            timeout=20
        )

        data = response.json()

        if "choices" in data and data["choices"]:
            reply = data["choices"][0]["message"]["content"].strip()
            chat_history.append({"role": "assistant", "content": reply})
            return reply
        else:
            return f"❌ Пустой или некорректный ответ от API: {data}"

    except Exception as e:
        return f"❌ Ошибка при запросе к OpenRouter API: {e}"

# === PROMPTS ===

def summarize_metrics(metrics_dict):
    if not metrics_dict or not isinstance(metrics_dict, dict):
        return "❌ Невозможно прокомментировать метрики — передан некорректный формат."

    summary = "Краткие метрики модели:\n"
    for key, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            summary += f"- {key}: {value:.3f}\n"

    prompt = (
        summary +
        "\nДай короткий, понятный и точный комментарий к этим метрикам. Объясняй, просто оцени их как эксперт."
    )
    return get_chatgpt_response(prompt)

def suggest_model(df_info):
    if not df_info:
        return "❌ Нет данных для анализа."

    prompt = (
        f"Вот структура данных:\n{df_info}\n\n"
        "Выбери одну из моделей: Decision Tree, Logistic Regression, Neural Network. "
        "Ответь только названием подходящей модели и переменной (с обяснениеям), которую нужно предсказывать. Объясни в 1-2 предложении."
    )
    return get_chatgpt_response(prompt)

def post_prediction_advice(y_pred, model_type, target_name, user_goal):
    try:
        # Преобразование предсказаний к списку
        if hasattr(y_pred, 'tolist'):
            y_pred_list = y_pred.tolist()
        else:
            y_pred_list = list(y_pred)

        if not y_pred_list:
            return "❌ Нет предсказаний для анализа."

        short_pred = y_pred_list[:10]
        prompt = (
            f"Использована модель {model_type} для предсказания '{target_name}'. "
            f"Пример предсказаний: {short_pred}. Цель пользователя: {user_goal}. "
            "Дай короткий, практичный совет (1–2 предложения), как интерпретировать результат или что делать дальше."
        )
        return get_chatgpt_response(prompt)

    except Exception as e:
        return f"❌ Ошибка при обработке предсказаний: {e}"

def continue_chat(user_message):
    if not user_message or not isinstance(user_message, str):
        return "❌ Пустой или некорректный запрос."

    prompt = user_message.strip() + "\nОтветь чётко, по делу, не выходи за рамки вопроса."
    return get_chatgpt_response(prompt)