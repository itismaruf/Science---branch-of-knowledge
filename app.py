# app.py

import streamlit as st
import pandas as pd
import os
import time
from utils import (
    load_data, 
    summarize_columns_for_gpt, ask_gpt_smart_cleaning,
    plot_data_visualizations, train_model, plot_predictions, remove_outliers_iqr, suggest_visualization_combinations
)
from AI_helper import (
    summarize_metrics, continue_chat, get_chatgpt_response, update_context, apply_gpt_cleaning, default_cleaning, context
)

# Конфигурация страницы
st.set_page_config(layout="wide")


# === Заставка ===
if "app_loaded" not in st.session_state:
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

            html, body {
                margin: 0;
                padding: 0;
                height: 100%;
                width: 100%;
                font-family: 'Inter', sans-serif;
                overflow: hidden;
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
                background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
                color: #0f172a;
                z-index: 9999;
                animation: fadeIn 1s ease-in-out;
                transition: opacity 1s ease-out;
            }

            .splash-container.fade-out {
                opacity: 0;
                pointer-events: none;
            }

            .ai-emoji {
                font-size: 3.2em;
                margin-bottom: 20px;
                animation: pulse 2s infinite;
            }

            .splash-title {
                font-size: 2.4em;
                font-weight: 700;
                text-align: center;
                opacity: 0;
                animation: fadeUp 1.2s ease-out forwards;
                animation-delay: 0.4s;
            }

            .splash-subtext {
                font-size: 1em;
                margin-top: 12px;
                color: #475569;
                opacity: 0;
                animation: fadeUp 1.4s ease-out forwards;
                animation-delay: 0.8s;
                text-align: center;
                max-width: 600px;
                padding: 0 16px;
            }

            .splash-footer {
                position: absolute;
                bottom: 18px;
                font-size: 0.8em;
                color: #64748b;
                text-align: center;
            }

            @keyframes fadeUp {
                0% { opacity: 0; transform: translateY(20px); }
                100% { opacity: 1; transform: translateY(0); }
            }

            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }

            @keyframes pulse {
                0%, 100% {
                    transform: scale(1);
                    opacity: 1;
                }
                50% {
                    transform: scale(1.15);
                    opacity: 0.75;
                }
            }
        </style>

        <div class="splash-container" id="splash">
            <div class="ai-emoji">✨</div>
            <div class="splash-title">ClariData</div>
            <div class="splash-subtext">Интеллектуальная система анализа данных<br>с автоочисткой, визуализацией, предсказаниями и пояснениями</div>
            <div class="splash-footer">© Created by Rahimov M.A.</div>
        </div>

        <script>
            setTimeout(() => {
                const splash = document.getElementById("splash");
                if (splash) splash.classList.add("fade-out");
            }, 3000);
        </script>
    """, unsafe_allow_html=True)

    time.sleep(5)
    st.session_state.app_loaded = True
    st.rerun()


# --- Установка API-ключа из секретов, если есть ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# --- Инициализация первой страницы при запуске ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'Загрузка данных'

st.markdown("""
    <style>
        /* Когда сайдбар открыт (aria-expanded="true"), основной контент смещается вправо */
        [data-testid="stSidebar"][aria-expanded="true"] ~ .main .block-container {
            margin-left: 300px;
            transition: margin-left 0.3s ease;
        }
        /* Когда сайдбар свернут (aria-expanded="false"), основной контент возвращается в исходное положение */
        [data-testid="stSidebar"][aria-expanded="false"] ~ .main .block-container {
            margin-left: 1rem;
            transition: margin-left 0.3s ease;
        }
    </style>
""", unsafe_allow_html=True)


# --- Функция переключения страниц ---
def set_page(page_name):
    st.session_state['page'] = page_name

# --- Сайдбар с навигацией и стилем кнопок ---
st.sidebar.header("🔧 Навигация")
pages = {
    "Загрузка данных": "📥",
    "Автообработка данных": "⚙️",
    "Визуализация": "📊",
    "Предсказание модели": "🔬",
    "Разъяснение результатов (с ИИ)": "💬",
    "Документация": "📄"
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

    # Зона загрузки
    with st.container():
        st.markdown("### 🔄 Загрузите файл данных (.csv, .xlsx, .xls)")
        uploaded_file = st.file_uploader("", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        try:
            # Загружаем данные
            df = load_data(uploaded_file)
            st.session_state["df"] = df
            st.success("Данные успешно загружены", icon="✅")

            # Отображаем пример данных
            st.markdown("### 🧾 Пример данных (первые 5 строк):")
            st.dataframe(df.head(), use_container_width=True)

            # Краткая сводка
            num_rows = df.shape[0]
            num_cols = df.shape[1]
            col_names = ", ".join(list(df.columns))
            data_summary = f"Данные содержат {num_rows} строк и {num_cols} столбцов. Столбцы: {col_names}"
            update_context("data_summary", data_summary)

            # Expander с описанием и кнопкой
            with st.expander("🎯 Описание данных и цель анализа", expanded=True):
                st.markdown("##### 🗒️ Расскажите о данных и вашей цели:")
                user_description = st.text_area(
                    "",
                    placeholder="Например: У меня данные о продажах, цель — предсказать доходы и выявить сезонные закономерности.",
                    height=100
                )

                # Кнопка без изменений
                if st.button("✨ Получить интерпретацию"):
                    if user_description.strip():
                        update_context("user_description", user_description)
                        prompt = (
                            f"Набор данных содержит следующую информацию: {data_summary}\n\n"
                            f"Описание от пользователя: {user_description}\n\n"
                            "Проанализируй структуру и содержание данных. Кратко опиши, какие ключевые признаки присутствуют, какие могут быть важны для анализа или принятия решений.(максимально коротко) "
                            "Выдели интересные закономерности, возможные проблемы или пробелы в данных.(короткость важно!) "
                            "Смотри данных — в зависимости от их тематики (например, если это об образовании, финансах, здравоохранении, городской среде и т.д.).(короткость важно!) "
                            "Формулируй ответ ясно и конкретно, без излишней общности. (Обрашай внимание на короткость и ясность чтобы читать не лень было!!!)(короткость важно!)"
                        )
                        with st.spinner("🔍 Получаем интерпретацию от ИИ..."):
                            ai_interpretation = get_chatgpt_response(prompt)
                        st.markdown("### 📊 Интерпретация данных от ИИ:")
                        st.info(ai_interpretation, icon="💡")
                    else:
                        st.warning("⚠️ Пожалуйста, введите описание данных и цели анализа.", icon="⚠️")
        except Exception as e:
            st.error(f"❌ Ошибка при обработке данных: {e}", icon="🚫")
    else:
        st.info("⬆ Загрузите файл для анализа.", icon="📁")


# === Автообработка данных ===
elif st.session_state["page"] == "Автообработка данных":
    st.title("⚙️ Автоматическая обработка данных")
    st.markdown("---")

    if "df" in st.session_state:
        df = st.session_state["df"]

        # Экспандер с описанием принципов очистки данных
        with st.expander("📖 Принципы очистки данных и заполнения пропусков", expanded=False):
            st.markdown("""
                #### 🧼 Почему важна очистка данных?
                Корректные данные — залог успешного анализа и построения моделей.

                - Повышается точность модели
                - Улучшается качество визуализации
                - Снижается уровень шума

                #### ⚙️ Принципы автоматической очистки:
                - Пропуски: удаление или заполнение (среднее, медиана, мода)
                - Категориальные: заполнение наиболее частым значением
            """)

        st.markdown("#### 🤖 Умная очистка данных")
        if st.button("♻️ Умная очистка (через LLM)"):
            with st.spinner("🔍 Запрос к ИИ..."):
                nulls_before = df.isnull().sum()
                total_before = nulls_before.sum()

                if total_before == 0:
                    st.info("✅ Пропущенных значений не найдено.", icon="✅")
                else:
                    st.markdown("#### 📉 Пропущенные значения до очистки:")
                    st.dataframe(nulls_before[nulls_before > 0])

                    # Подготовка данных для запроса к ИИ
                    summary = summarize_columns_for_gpt(df)
                    gpt_response = ask_gpt_smart_cleaning(summary)

                    # Применение рекомендаций ИИ
                    apply_gpt_cleaning(df, gpt_response)
                    st.session_state["df"] = df

                    nulls_after = df.isnull().sum()
                    total_after = nulls_after.sum()
                    total_cleaned = total_before - total_after

                    st.success("✅ Очистка завершена.", icon="🧹")
                    if total_cleaned > 0:
                        st.info(f"✨ Обработано {int(total_cleaned)} пропусков.", icon="📊")

                    # Итоговый лог
                    all_logs = st.session_state.get("cleaning_log", [])
                    filtered_logs = [
                        log for log in all_logs 
                        if any(sub in log for sub in ["заполнено", "не удалось", "ошибка", "содержит пропуски"])
                    ]
                    if filtered_logs:
                        st.markdown("#### 📘 Итог очистки:")
                        report = {}
                        for log in filtered_logs:
                            parts = log.split(":")
                            if len(parts) >= 2:
                                col = parts[0].strip()
                                action = ":".join(parts[1:]).strip()
                                if col in report:
                                    report[col] += "; " + action
                                else:
                                    report[col] = action
                        for col, action in report.items():
                            st.write(f"**{col}**: {action}")
                    else:
                        st.info("Очистка завершена — дополнительных действий не потребовалось.", icon="✅")

                    st.markdown("### 📋 Инструкция от ИИ:")
                    st.code(gpt_response)

            # Ручная очистка по неуказанным колонкам
            unspecified_columns = st.session_state.get("unspecified_columns", [])
            if unspecified_columns:
                st.markdown("#### 🔧 Дополнительная очистка по колонкам:")
                for col in unspecified_columns.copy():
                    if st.button(f"Очистить колонку «{col}» вручную", key=f"clean_{col}"):
                        result = default_cleaning(df, col)
                        st.success(result)
                        st.session_state["cleaning_log"].append(result)
                        unspecified_columns.remove(col)
                        st.session_state["unspecified_columns"] = unspecified_columns

        st.markdown("---")

        # Добавляем описание перед кнопкой удаления выбросов
        st.markdown("#### 📦 Удаление выбросов")
        with st.expander("ℹ️ Описание удаления выбросов", expanded=False):
            st.markdown("""
            **Метод IQR для удаления выбросов:**
            
            - Для числовых переменных рассчитываются первый (Q1) и третий (Q3) квартили и интерквартильный размах (IQR = Q3 - Q1).
            - Удаляются строки, значения которых выходят за пределы интервала: [Q1 - 1.5·IQR, Q3 + 1.5·IQR].
            
            **Важно:**
            - Этот метод хорошо работает на непрерывных переменных.
            - Если в данных присутствуют бинарные или дискретные переменные (например, 0 и 1), где наблюдений мало,
              применение этого метода может удалить важные данные. Убедитесь, что удаление выбросов применимо именно
              к вашим данным.
            """)

        if st.button("🗑️ Удалить выбросы (IQR)"):
            with st.spinner("📉 Удаление выбросов по IQR..."):
                df, removed_rows = remove_outliers_iqr(df)
                st.session_state["df"] = df
            st.success("✅ Выбросы удалены.", icon="🧽")
            if removed_rows > 0:
                st.info(f"Удалено {removed_rows} строк с выбросами.", icon="📈")
            else:
                st.info("Выбросов не обнаружено или они не требуют удаления.", icon="✔️")
    else:
        st.warning("📥 Сначала загрузите данные.", icon="⚠️")


# === Визуализация ===
elif st.session_state["page"] == "Визуализация":
    st.title("📊 Визуализация данных")
    st.markdown("---")

    if "df" in st.session_state:
        df = st.session_state["df"]

        # 🧭 Выбор переменных
        st.markdown("#### 🧭 Выбор переменных")
        x = st.selectbox("Переменная по оси X", df.columns)
        y = st.selectbox("Переменная по оси Y (необязательно)", [""] + list(df.columns))
        y = y if y else None

        # 💡 Блок с рекомендациями сразу под выбором переменных
        with st.container():
            st.markdown(
                """
                <div style="border: 1px solid #3399ff; border-radius: 8px; padding: 10px; background-color: #f0f8ff;">
                <strong>💡 Хотите подсказку? </strong> Нажмите кнопку ниже, чтобы получить интересные комбинации переменных для визуализации.
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<div style='margin-top: -5px;'></div>", unsafe_allow_html=True)
            if st.button("✨ Рекомендовать комбинации переменных"):
                df_info = f"Набор переменных: {', '.join(df.columns)}"
                suggestion = suggest_visualization_combinations(df_info)
                st.info(f"📌 Рекомендации от ИИ:\n\n{suggestion}")

        st.markdown("---")

        # 🎨 Настройка графика
        st.markdown("#### 🎨 Настройка графика")
        chart_type = st.selectbox(
            "Тип графика",
            ["Автоматически", "Гистограмма", "Круговая диаграмма", "Точечный график", "Boxplot", "Bar-график", "Лайнплот"]
        )

        st.markdown("---")

        # 🔍 Фильтрация данных
        st.markdown("#### 🔍 Фильтрация данных")
        filters = {}
        for col in [x, y] if y else [x]:
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val, max_val = float(df[col].min()), float(df[col].max())
                selected = st.slider(f"Фильтр по {col}", min_val, max_val, (min_val, max_val))
                filters[col] = selected

        st.markdown("---")

        # 📌 Дополнительные опции
        st.markdown("#### 📌 Дополнительные опции")
        top_n = None
        if st.checkbox("Показать только top-N категорий"):
            top_n = st.slider("Выберите N", 3, 30, 10)

        st.markdown("---")

        # 📈 Построение графика
        st.markdown("#### 📈 Построение графика")
        fig = plot_data_visualizations(
            df,
            x=x,
            y=y,
            top_n=top_n,
            numeric_filters=filters,
            chart_type=chart_type
        )

        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⚠️ Невозможно построить график с выбранными параметрами.")

    else:
        st.warning("📥 Сначала загрузите данные.", icon="📎")

# === Предсказание модели ===
elif st.session_state["page"] == "Предсказание модели":
    st.title("🔬 Обучение модели и предсказание")

    st.markdown("---")

    with st.expander("ℹ️ Описание раздела"):
        st.markdown("""
        Получите **рекомендации** по выбору целевой переменной и оптимальной модели для обучения, 
        основываясь на ваших данных (и, при наличии, описании целей из раздела «Загрузка данных»). 

        **Доступные модели:**
        - Decision Tree
        - Logistic Regression
        - Neural Network
        - Random Forest
        - Gradient Boosting
        - SVM
        - KNN

        Если не уверены в выборе — воспользуйтесь автоматическим режимом.
        """)

    if "df" in st.session_state:
        df = st.session_state["df"]

        # 💡 Блок с рекомендацией переменной и модели
        st.markdown("#### 💡 Рекомендация от ИИ")
        with st.container():
            st.markdown(
                """
                <div style="border: 1px solid #3399ff; border-radius: 8px; padding: 12px; background-color: #f0f8ff; margin-bottom: 10px;">
                <strong>🤖 Нужна помощь?</strong><br>
                Нажмите кнопку ниже, чтобы получить целевую переменную и подходящую модель от ИИ на основе ваших данных.
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("💡 Рекомендовать переменную и модель", key="recommendation"):
                prompt_target = (
                    f"На основе следующих характеристик данных: {context.get('data_summary', 'Сводка данных отсутствует')}. "
                    "Рекомендуй одну или несколько переменную, которая оптимально подойдёт в качестве целевой для предсказания, "
                    "и максимально кратко опиши, почему именно эта переменная является наилучшей для данной задачи. "
                    "Также укажи, какая модель обеспечит высокую точность и быструю работу, и объясни это простыми словами (максимально коротко)."
                )
                recommended_target = get_chatgpt_response(prompt_target)
                st.info(f"📌 Рекомендация от ИИ:\n\n{recommended_target}")

        st.markdown("---")

        # 🎯 Выбор целевой переменной и модели
        st.markdown("#### 🎯 Выбор целевой переменной и модели")
        target = st.selectbox("Целевая переменная", df.columns, key="target_select")

        model_type = st.selectbox(
            "Модель для обучения",
            ["Выберите модель", "Decision Tree", "Logistic Regression", "Neural Network", 
             "Random Forest", "Gradient Boosting", "SVM", "KNN"],
            key="model_select"
        )

        if st.button("🤖 Автовыбор модели", key="auto_model"):
            model_type = "Neural Network"
            st.success("Модель 'Neural Network' выбрана автоматически.")

        st.markdown("---")

        # 🚀 Запуск обучения модели (запускается только при явном нажатии)
        st.markdown("#### 🚀 Обучение модели")
        if st.button("🚀 Начать обучение модели", key="start_training"):
            if model_type == "Выберите модель":
                st.info("Пожалуйста, выберите подходящую модель.")
            else:
                try:
                    metrics, model, X_test, y_test, y_pred = train_model(df, target, model_type)
                    if metrics is not None:
                        st.subheader("📋 Метрики модели")
                        st.info(summarize_metrics(metrics["weighted avg"]))
                        with st.expander("📑 Полный отчет о метриках"):
                            st.json(metrics)

                        st.subheader("📊 Предсказания модели")
                        plot_predictions(y_test, y_pred)

                        # Сохраняем результаты для последующего использования
                        st.session_state.update({
                            "X_test": X_test,
                            "y_pred": y_pred,
                            "model_type": model_type,
                            "target": target,
                            "metrics": metrics  # сохраняем метрики для комментария от ИИ
                        })
                    else:
                        st.error(
                            "⚠️ Ошибка при обучении. Проверьте, что данные корректны и содержат хотя бы два класса в целевой переменной."
                        )
                except Exception:
                    st.error(
                        "⚠️ Произошла ошибка при обучении. Убедитесь, что данные очищены и не содержат критических проблем."
                    )
        else:
            st.info("Выберите параметры и нажмите '🚀 Начать обучение модели'.")

        st.markdown("---")

        # 🧠 Комментарий от ИИ по метрикам
        if "metrics" in st.session_state:
            if st.button("🧠 Комментарий от ИИ по метрикам", key="ai_comment"):
                if "ai_commentary" not in st.session_state:
                    st.session_state["ai_commentary"] = summarize_metrics(st.session_state["metrics"]["weighted avg"])
                st.info(st.session_state["ai_commentary"])
        else:
            st.info("После обучения модели вы сможете увидеть комментарий от ИИ по метрикам.")

    else:
        st.warning("📥 Сначала загрузите данные.", icon="📎")



# === Разъяснение результатов ===
elif st.session_state["page"] == "Разъяснение результатов (с ИИ)":
    st.title("💬 Разъяснение результатов (с ИИ)")
    st.markdown("---")

    st.markdown(
        """
        Хотите понять, **почему модель предсказала именно так**, что значат метрики или как найти ошибки в данных?  
        Просто выберите вопрос ниже или напишите свой — и получите разъяснение в формате чата.
        """
    )

    suggested_questions = [
        "Что означают метрики модели и как их можно улучшить?",
        "Как интерпретировать результаты предсказаний для бизнеса?",
        "Какие проблемы с данными могут повлиять на модель и как их устранить?",
        "Как ошибки в данных влияют на прогноз и что с этим делать?"
    ]

    with st.container():
        st.markdown("#### 💡 Частые вопросы")
        selected_question = st.radio("", suggested_questions, key="radio_question")

        if st.button("📥 Использовать выбранный вопрос"):
            st.session_state["chosen_question"] = selected_question

    st.markdown("#### 💬 Ваш вопрос")
    user_input = st.text_area(
        "",
        value=st.session_state.get("chosen_question", ""),
        placeholder="Например: почему модель выбрала именно такой класс для клиента?",
        height=100,
        label_visibility="collapsed"
    )

    # Имитируем чат
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("🚀 Отправить"):
        question = user_input.strip()
        if question:
            with st.spinner("ИИ думает..."):
                answer = continue_chat(question)
                # Сохраняем в историю
                st.session_state.chat_history.append(("🧑‍💻 Вы", question))
                st.session_state.chat_history.append(("🤖 ИИ", answer))
        else:
            st.warning("⚠️ Пожалуйста, введите или выберите вопрос.")

    # Отображаем историю чата
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 💬 История диалога")
        for speaker, message in st.session_state.chat_history:
            st.markdown(f"**{speaker}:** {message}")


if st.session_state.get("page") == "Документация":
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            readme_content = f.read()
        st.markdown(readme_content, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Документация не найдена. Убедитесь, что файл README.md существует в проекте.")


# Футер внизу страницы (автор)
# Постоянная надпись внизу лево, вне зависимости от содержимого
st.markdown("""
    <style>
        .bottom-right {
            position: fixed;
            right: 15px;
            bottom: 10px;
            font-size: 0.75em;
            color: #333333;
            z-index: 9999;
        }
    </style>
    <div class="bottom-right">© Created by Rahimov M.A.</div>
""", unsafe_allow_html=True)
