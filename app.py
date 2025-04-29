# app.py

import streamlit as st
import pandas as pd
import os
import time
from utils import (
    load_data, 
    summarize_columns_for_gpt, ask_gpt_smart_cleaning, apply_gpt_cleaning,
    plot_data_visualizations, train_model, plot_predictions
)
from AI_helper import (
    summarize_metrics, suggest_model, post_prediction_advice, continue_chat, get_chatgpt_response, update_context
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
    time.sleep(1.5)
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
    "Автообработка данных": "⚙️",
    "Визуализация": "📊",
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

    # Поле загрузки файла
    uploaded_file = st.file_uploader("Загрузите файл", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        try:
            # Загружаем данные
            df = load_data(uploaded_file)
            st.session_state["df"] = df
            st.success("✅ Данные успешно загружены")

            # Отображаем пример данных (первые 5 строк)
            st.subheader("Пример данных:")
            st.dataframe(df.head())

            # Формируем краткую сводку по структуре данных
            num_rows = df.shape[0]
            num_cols = df.shape[1]
            col_names = ", ".join(list(df.columns))
            data_summary = f"Данные содержат {num_rows} строк и {num_cols} столбцов. Столбцы: {col_names}"

            # Сохраняем сводку данных в контексте для ИИ
            update_context("data_summary", data_summary)

            # Экспандер для интерпретации данных
            with st.expander("📋 Интерпретация данных от ИИ"):                
                # Формируем запрос и получаем интерпретацию от ИИ
                prompt_auto = (
                    f"Набор данных имеет следующие характеристики: {data_summary}. "
                    "Проанализируй данные и дай максимально краткую интерпретацию: "
                    "если данные о студентах, охарактеризуй атрибуты. Если о банковских операциях, дай анализ. "
                    "Представь только важные моменты."
                )
                with st.spinner("Получаем интерпретацию..."):
                    ai_auto_interpretation = get_chatgpt_response(prompt_auto)
                st.write(ai_auto_interpretation)

            # Экспандер для пользовательского описания данных и цели анализа
            with st.expander("📝 Описание данных и цель анализа"):
                user_description = st.text_area(
                    "Опишите, что представляют собой данные, вашу цель и ожидания от анализа.",
                    placeholder="Например: У меня данные о продажах, цель — предсказать доходы и выявить сезонные закономерности."
                )
                if user_description.strip():
                    update_context("user_description", user_description)
                    st.success("✅ Ваше описание сохранено.")
        except Exception as e:
            st.error(f"❌ Ошибка при обработке данных: {e}")
    else:
        st.info("⬆️ Загрузите файл для анализа.")



# === Автообработка данных ===
elif st.session_state["page"] == "Автообработка данных":
    st.title("⚙️ Обработка данных")
    
    if "df" in st.session_state:
        df = st.session_state["df"]

        # Экспандер с описанием принципов очистки данных
        with st.expander("📖 Принципы очистки данных и заполнения пропусков", expanded=False):
            st.write("""
                **Почему важна очистка данных?**
                
                Корректные данные — основной залог успешного анализа и построения модели. Пропущенные или ошибочные значения
                могут снизить точность предсказаний и исказить визуализацию. Автоматическая очистка помогает:
                - Улучшить точность модели,
                - Обеспечить достоверность визуализации,
                - Снизить уровень шума в наборе данных.
                
                **Принципы очистки:**
                - Пропуски могут быть удалены или заполнены (средним, медианой или модой),
                - Для категориальных данных — заполнение наиболее частым значением,
                - Выводится итог по изменённым столбцам после проведения обработки.
            """)

        # Кнопка для запуска умной очистки через LLM
        if st.button("✨🧹 Умная очистка (через LLM)"):
            with st.spinner("Запрос к ИИ..."):
                nulls_before = df.isnull().sum()
                total_before = nulls_before.sum()

                if total_before == 0:
                    st.info("❌ Пропущенных значений не найдено.")
                else:
                    st.markdown("**Проверка пропущенных значений до очистки:**")
                    st.dataframe(nulls_before[nulls_before > 0])

                    # Подготовка данных для запроса к ИИ
                    summary = summarize_columns_for_gpt(df)
                    gpt_response = ask_gpt_smart_cleaning(summary)

                    # Применение рекомендаций ИИ для очистки данных
                    apply_gpt_cleaning(df, gpt_response)
                    st.session_state["df"] = df

                    nulls_after = df.isnull().sum()
                    total_after = nulls_after.sum()
                    total_cleaned = total_before - total_after

                    st.success("✅ Очистка завершена.")
                    if total_cleaned > 0:
                        st.success(f"🧹 Обработано {int(total_cleaned)} пропусков.")

                    # Фильтруем лог очистки: показываем только важные изменения
                    all_logs = st.session_state.get("cleaning_log", [])
                    # Отбираем записи, содержащие ключевые слова (важные действия)
                    filtered_logs = [
                        log for log in all_logs 
                        if any(sub in log for sub in ["заполнено", "не удалось", "ошибка", "содержит пропуски"])
                    ]

                    if filtered_logs:
                        st.markdown("### Итог очистки:")
                        # Формируем краткий отчет по столбцам.
                        report = {}
                        for log in filtered_logs:
                            parts = log.split(":")
                            if len(parts) >= 2:
                                col = parts[0].strip()
                                action = ":".join(parts[1:]).strip()
                                # Если для столбца уже есть запись, объединяем сообщения.
                                if col in report:
                                    report[col] += "; " + action
                                else:
                                    report[col] = action
                        for col, action in report.items():
                            st.write(f"**{col}**: {action}")
                    else:
                        st.info("Изменения не требовали отображения дополнительного лога.")

                    st.markdown("**📋 Инструкция от ИИ:**")
                    st.code(gpt_response)
    else:
        st.warning("Сначала загрузите данные")


# === Визуализация ===
elif st.session_state["page"] == "Визуализация":
    st.title("📊 Визуализация")
    if "df" in st.session_state:
        df = st.session_state["df"]
        x = st.selectbox("X переменная", df.columns)
        y = st.selectbox("Y переменная (опционально)", [""] + list(df.columns))
        y = y if y else None

        st.markdown("🎨 Настройка графика")
        chart_type = st.selectbox(
            "Тип графика",
            ["Автоматически", "Гистограмма", "Круговая диаграмма", "Точечный график", "Boxplot", "Bar-график", "Лайнплот"]
        )

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
    
    # Описание раздела обучения модели с использованием экспандера
    with st.expander("ℹ️ Описание раздела:"):
        st.markdown("""
        Здесь вы можете обучить модель на основе загруженных данных для выявления ключевых закономерностей и последующего прогнозирования. 
        Выберите целевую переменную (например, прогноз продаж, оценка кредитоспособности, результаты тестов и т.п.) и подходящую модель.
        
        **Рекомендуемые универсальные модели:**
        - **Decision Tree:** Наглядная и интерпретируемая, подходит для небольших бизнес-задач и образовательных данных.
        - **Logistic Regression:** Эффективна для бинарной классификации, часто используется в финансовой аналитике.
        - **Neural Network:** Идеальна для сложных наборов данных (например, крупные бизнес-данные, маркетинговые исследования), где требуется модель глубокого обучения.
        
        Если вы не уверены в выборе, воспользуйтесь режимом автоматического выбора модели.
        """)

    if "df" in st.session_state:
        df = st.session_state["df"]
        target = st.selectbox("Целевая переменная", df.columns)
        model_type = st.selectbox("Модель", ["", "Decision Tree", "Logistic Regression", "Neural Network", "Random Forest", "Gradient Boosting", "SVM", "KNN"])
        
        # Автоматика выбора модели – если пользователь не уверен, то можно задать нейронную сеть по умолчанию
        if st.button("Автовыбор модели"):
            model_type = "Neural Network"
    
        # Проверяем наличие выбранных параметров
        if model_type and target:
            metrics, model, X_test, y_test, y_pred = train_model(df, target, model_type)
            if metrics is not None:
                st.subheader("📋 Метрики модели")
                st.info(summarize_metrics(metrics["weighted avg"]))
                with st.expander("Полный отчет"):
                    st.json(metrics)
    
                st.subheader("📊 Предсказания модели")
                plot_predictions(y_test, y_pred)
    
                # Сохраняем результаты для дальнейшего использования
                st.session_state.update({
                    "X_test": X_test,
                    "y_pred": y_pred,
                    "model_type": model_type,
                    "target": target
                })
    
                if st.button("Комментарий от ИИ"):
                    st.info(summarize_metrics(metrics["weighted avg"]))
            else:
                st.error(
                    "Ошибка при обучении модели. Проверьте, что данные корректны: "
                    "все пропущенные значения обработаны, выбранная целевая переменная имеет как минимум два класса, "
                    "и данные соответствуют требованиям модели."
                )
        else:
            st.info("Выберите параметры для обучения модели.")
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