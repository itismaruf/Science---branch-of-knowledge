import requests
from dotenv import load_dotenv
import os
import streamlit as st

# Загрузка переменных окружения
load_dotenv()
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# Проверка API-ключа
if not api_key:
    raise ValueError("❌ API-ключ не найден. Убедитесь, что переменная OPENROUTER_API_KEY задана в .env.")

# История диалога и контекста
chat_history = [
    {
        "role": "system",
        "content": (
            "Ты помощник на русском языке. Всегда отвечай четко, по делу, коротко и с точностью, "
            "но добавляй практические рекомендации (если тебе скажуть что не надо, не давай рекомендации) "
            "и полезные комментарии, чтобы помочь пользователю понять, как использовать данные. "
            "Если набор данных содержит информацию, например, о студентах, обрати внимание на аспекты эмоционального и образовательного состояния. "
            "Будь дружелюбным, эмпатичным, кратким, ясным и конкретным в ответах."
            "Некогда не говори по англисский! Всегда по русский!"
            "Будь внимательно, конскнтрируйся на то что просять, не скажи лишное слова!"
        )
    }
]

# Глобальный контекст проекта
context = {}

def update_context(key, value):
    """Обновление контекста проекта (фиксирует информацию о данных и цели)."""
    context[key] = value

def get_chatgpt_response(prompt, model="deepseek/deepseek-r1:free"):
    if not prompt or not isinstance(prompt, str):
        return "❌ Пустой или некорректный запрос."

    # Собираем текущий контекст в виде строк
    context_info = "\n".join([f"{key}: {value}" for key, value in context.items()])
    # Формируем полный запрос: сначала фиксированный контекст проекта, потом запрос
    full_prompt = f"Контекст проекта:\n{context_info}\n\n{prompt}\n\nОтвечай кратко, конкретно и добавляй практические советы, если это уместно."
    chat_history.append({"role": "user", "content": full_prompt})

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
    """Объясняет метрики модели, полученные в результате предсказания."""
    if not metrics_dict or not isinstance(metrics_dict, dict):
        return "❌ Невозможно объяснить метрики — передан некорректный формат."

    summary = "Краткие метрики модели:\n"
    for key, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            summary += f"- {key}: {value:.3f}\n"

    prompt = (
        summary +
        "\nПоясни кратко, что означает каждая из этих метрик, без аналитических рекомендаций или предложений по улучшению. Объяснение должно быть простым и доступным, чтобы пользователь мог легко понять результаты предсказания."
    )
    
    try:
        response = get_chatgpt_response(prompt)
        if response and response.strip():
            return response
        else:
            return "❌ Ответ от ИИ отсутствует. Проверьте подключение или повторите запрос."
    except Exception as e:
        return f"❌ Произошла ошибка при обращении к ИИ: {e}"


def suggest_model(df_info, user_goal):
    """Рекомендация модели на основе структуры данных и цели анализа."""
    if not df_info or not user_goal:
        return "❌ Недостаточно данных для рекомендации модели."

    update_context("df_info", df_info)
    update_context("user_goal", user_goal)

    prompt = (
        f"Вот структура данных:\n{df_info}\n\n"
        f"Цель пользователя: {user_goal}\n\n"
        "Какая модель (например, Decision Tree, Logistic Regression, Neural Network) лучше всего подойдет для достижения цели? "
        "Ответь названием модели и кратким объяснением выбора, добавив комментарии о том, как эта модель может помочь в реальной жизни."
    )
    return get_chatgpt_response(prompt)

def post_prediction_advice(y_pred, model_type, target_name, user_goal):
    """Совет по интерпретации результатов прогнозирования."""
    try:
        if hasattr(y_pred, 'tolist'):
            y_pred_list = y_pred.tolist()
        else:
            y_pred_list = list(y_pred)

        if not y_pred_list:
            return "❌ Нет предсказаний для анализа."

        short_pred = y_pred_list[:10]
        update_context("last_predictions", short_pred)

        prompt = (
            f"Использована модель {model_type} для предсказания '{target_name}'. "
            f"Пример предсказаний: {short_pred}. Цель пользователя: {user_goal}. "
            "Объясни простыми словами, что означают результаты, и дай рекомендации, что делать дальше, "
            "обращая внимание на практические шаги для улучшения ситуации."
        )
        return get_chatgpt_response(prompt)

    except Exception as e:
        return f"❌ Ошибка при обработке предсказаний: {e}"

def continue_chat(user_message):
    """Обрабатывает любое сообщение от пользователя с учетом контекста проекта."""
    if not user_message or not isinstance(user_message, str):
        return "❌ Пустой или некорректный запрос."

    prompt = user_message.strip() + "\nОтветь четко, с учетом контекста проекта и целей пользователя."
    return get_chatgpt_response(prompt)
