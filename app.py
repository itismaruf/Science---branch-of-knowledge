# app.py

import streamlit as st
import pandas as pd
import os
import time
from utils import (
    load_data, analyze_data_quality, 
    summarize_columns_for_gpt, ask_gpt_smart_cleaning, apply_gpt_cleaning,
    plot_data_visualizations, train_model, plot_predictions
)
from AI_helper import (
    summarize_metrics, suggest_model, post_prediction_advice, continue_chat
)

# Конфигурация страницы
st.set_page_config(layout="wide")

# Заставка при первом запуске
if "app_loaded" not in st.session_state:
    st.markdown("""
        <style>
            body {
                margin: 0;
                padding: 0;
            }
            .splash-container {
                position: fixed;
                top: 0; left: 0;
                width: 100vw;
                height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                background-color: var(--background-color);
                color: var(--text-color);
                z-index: 9999;
                font-family: 'Segoe UI', sans-serif;
            }
                .splash-title {
                    font-size: 3.2em;
                    font-weight: bold;
                    text-align: center;
                    margin-bottom: 20px;
                    margin-top: -5%; 
                }
            }
            .splash-author {
                position: absolute;
                bottom: 20px;
                right: 30px;
                font-size: 0.9em;
                color: gray;
            }
        </style>
        <div class="splash-container">
            <div class="splash-title">🤖 Интеллектуальная система<br>автоматизации аналитических отчётов</div>
        </div>
    """, unsafe_allow_html=True)
    placeholder = st.empty()
    time.sleep(3.5)
    st.session_state.app_loaded = True
    st.rerun()

# --- Установка API-ключа из секретов, если есть ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# --- Инициализация первой страницы при запуске ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'Загрузка данных'

# --- Функция переключения страниц ---
def set_page(page_name):
    st.session_state['page'] = page_name

# --- Сайдбар с навигацией и стилем кнопок ---
st.sidebar.header("🔧 Навигация")
pages = {
    "Загрузка данных": "📥",
    "Анализ данных": "📊",
    "Визуализация": "📈",
    "Обучение модели": "🔬",
    "Предсказание и советы": "📖",
    "Продолжить диалог": "💬"
}

# Настройка CSS для кнопок (цвета при наведении)
st.markdown("""
    <style>
        div.stButton > button {
            background-color: #f0f2f6;
            color: black;
            border: 1px solid #ccc;
            border-radius: 6px;
        }
        div.stButton > button:hover {
            background-color: #e0f0ff;
            color: #007BFF;
            border: 1px solid #007BFF;
        }
    </style>
""", unsafe_allow_html=True)

# Навигационные кнопки
for name, icon in pages.items():
    st.sidebar.button(f"{icon} {name}", on_click=set_page, args=(name,))

# Кнопка для очистки всех данных
if st.sidebar.button("🔄 Очистить всё"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# ===================== СТРАНИЦЫ =======================

# === Загрузка данных ===
if st.session_state["page"] == "Загрузка данных":
    st.title("📥 Загрузка данных")
    uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])

    if uploaded_file:
        try:
            # Загружаем и отображаем данные
            df = load_data(uploaded_file)
            st.session_state["df"] = df
            st.success("✅ Данные загружены")
            st.dataframe(df.head())

            # Кнопка умной очистки данных с помощью GPT
            with st.expander("📖 Принцип очистки данных и заполнения пропусков", expanded=False):
                st.write("""
                    Для очистки данных будут использованы следующие принципы:
                    - Пропущенные значения могут быть удалены или заполнены в зависимости от метода очистки.
                    - Если столбец содержит числовые данные, пропуски могут быть заменены средним значением или медианой.
                    - В случае категориальных данных пропуски могут быть заменены на наиболее частое значение.
                    - Будут отображены изменения после очистки, чтобы вы могли увидеть, сколько пропусков было удалено или заполнено.
                """)

            
        except Exception as e:
            st.error(f"❌ Ошибка при загрузке или очистке: {e}")
    else:
        st.info("⬆️ Загрузите CSV-файл для начала")

# === Анализ данных ===
elif st.session_state["page"] == "Анализ данных":
    st.title("📊 Анализ данных")

    if "df" in st.session_state:
        df = st.session_state["df"]

        # Краткий анализ данных
        st.markdown(analyze_data_quality(df))

        # Умная очистка
        if st.button("🧠 Умная очистка (через LLM)"):
            with st.spinner("Запрос к ИИ..."):
                nulls_before = df.isnull().sum()
                total_before = nulls_before.sum()

                if total_before == 0:
                    st.info("❌ Пропущенных значений не найдено.")
                else:
                    st.write("**До очистки (пропущенные значения):**")
                    st.dataframe(nulls_before[nulls_before > 0])

                    # Подготовка данных и запрос к GPT
                    summary = summarize_columns_for_gpt(df)
                    gpt_response = ask_gpt_smart_cleaning(summary)
                    cleaned_columns = apply_gpt_cleaning(df, gpt_response)
                    st.session_state["df"] = df

                    nulls_after = df.isnull().sum()
                    total_after = nulls_after.sum()
                    total_cleaned = total_before - total_after

                    st.success("✅ Обработка завершена")

                    if total_cleaned > 0:
                        st.success(f"🧹 Обработано {int(total_cleaned)} пропусков.")

                    # Показываем лог очистки от GPT
                    if "cleaning_log" in st.session_state and st.session_state["cleaning_log"]:
                        st.markdown("**📝 Лог очистки от ИИ:**")
                        for line in st.session_state["cleaning_log"]:
                            st.write(f"- {line}")

                    # Показываем инструкцию GPT
                    st.markdown("**📋 Инструкция от ИИ:**")
                    st.code(gpt_response)

        # Рекомендация модели
        if st.button("📌 Рекомендовать модель"):
            summary = df.describe(include='all').to_string()
            st.info(suggest_model(summary))

    else:
        st.warning("Сначала загрузите данные")

# === Визуализация ===
elif st.session_state["page"] == "Визуализация":
    st.title("📈 Визуализация")
    if "df" in st.session_state:
        df = st.session_state["df"]
        x = st.selectbox("X переменная", df.columns)
        y = st.selectbox("Y переменная (опционально)", [""] + list(df.columns))
        y = y if y else None

        st.markdown("🎨 Настройка графика")
        chart_type = st.selectbox("Тип графика", ["Автоматически", "Гистограмма", "Круговая диаграмма", "Точечный график", "Boxplot", "Bar-график"])

        # Фильтры по числовым значениям
        filters = {}
        for col in [x, y] if y else [x]:
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val, max_val = float(df[col].min()), float(df[col].max())
                selected = st.slider(f"Фильтр для {col}", min_val, max_val, (min_val, max_val))
                filters[col] = selected

        # Топ-N категорий
        top_n = None
        if st.checkbox("Показать только top-N категорий"):
            top_n = st.slider("Выберите N", 3, 30, 10)

        # Построение графика
        fig = plot_data_visualizations(df, x=x, y=y, top_n=top_n, numeric_filters=filters, chart_type=chart_type)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Сначала загрузите данные")

# === Обучение модели ===
elif st.session_state["page"] == "Обучение модели":
    st.title("🔬 Обучение модели")
    if "df" in st.session_state:
        df = st.session_state["df"]
        target = st.selectbox("Целевая переменная", df.columns)
        model_type = st.selectbox("Модель", ["", "Decision Tree", "Logistic Regression", "Neural Network"])

        if st.button("Автовыбор модели"):
            model_type = "Neural Network"

        # Обучение и вывод метрик
        if model_type and target:
            metrics, model, X_test, y_test, y_pred = train_model(df, target, model_type)
            st.subheader("📋 Метрики модели")
            st.info(summarize_metrics(metrics["weighted avg"]))
            with st.expander("Полный отчёт"):
                st.json(metrics)
            st.subheader("📊 Предсказания модели")
            plot_predictions(y_test, y_pred)

            # Сохраняем результаты
            st.session_state.update({
                "X_test": X_test,
                "y_pred": y_pred,
                "model_type": model_type,
                "target": target
            })

            if st.button("Комментарий от ИИ"):
                st.info(summarize_metrics(metrics["weighted avg"]))
        else:
            st.info("Выберите параметры")
    else:
        st.warning("Сначала загрузите данные")

# === Советы по предсказанию ===
elif st.session_state["page"] == "Предсказание и советы":
    st.title("📖 Советы по предсказанию")
    if all(k in st.session_state for k in ["X_test", "y_pred", "model_type"]):
        if st.button("Получить совет от ИИ"):
            st.subheader("📬 Советы")
            st.success(post_prediction_advice(
                st.session_state["X_test"],
                st.session_state["y_pred"],
                st.session_state["model_type"],
                st.session_state.get("target", "target")
            ))
    else:
        st.warning("Сначала обучите модель")

# === Диалог с ИИ ===
elif st.session_state["page"] == "Продолжить диалог":
    st.title("💬 Диалог с ИИ")
    user_input = st.text_area("Ваш вопрос:")
    if st.button("Отправить"):
        response = continue_chat(user_input)
        st.success(response)

# Футер внизу страницы (автор)
# Постоянная надпись внизу справа, вне зависимости от содержимого
st.markdown("""
    <style>
        .bottom-right {
            position: fixed;
            right: 15px;
            bottom: 10px;
            font-size: 0.75em;
            color: gray;
            z-index: 9999;
        }
    </style>
    <div class="bottom-right">© Created by Rahimov M.A.</div>
""", unsafe_allow_html=True)