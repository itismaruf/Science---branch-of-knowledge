import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import plotly.express as px
import re
from AI_helper import get_chatgpt_response

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    conversion_log = []

    for col in df.columns:
        original_dtype = df[col].dtype

        if original_dtype == "object":
            # Удаляем лишние пробелы и запятые
            df[col] = df[col].astype(str).str.replace(",", ".").str.strip()

            # Пробуем привести к числу
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
                conversion_log.append(f"{col}: object → float (успешно)")
            except:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    success_rate = df[col].notnull().mean()
                    if success_rate > 0.9:
                        conversion_log.append(f"{col}: object → float (по успешности {success_rate:.0%})")
                    else:
                        df[col] = df[col].astype(str)
                        conversion_log.append(f"{col}: остался как текст (только {success_rate:.0%} чисел)")
                except:
                    df[col] = df[col].astype(str)
                    conversion_log.append(f"{col}: остался как текст (неподдающийся формат)")
        else:
            conversion_log.append(f"{col}: {original_dtype} (без изменений)")

    # Сохраняем лог преобразования типов
    st.session_state["conversion_log"] = conversion_log

    # 📊 Формируем базовую информацию о датасете
    base_info = {
        "Количество строк": df.shape[0],
        "Количество столбцов": df.shape[1],
        "Пропущенные значения": int(df.isnull().sum().sum()),
        "Дубликаты": int(df.duplicated().sum()),
        "Числовые признаки": len(df.select_dtypes(include=["number"]).columns),
        "Категориальные признаки": len(df.select_dtypes(include=["object"]).columns),
    }

    st.session_state["base_info"] = base_info
    st.subheader("📊 Базовая информация о данных")
    for key, value in st.session_state["base_info"].items():
        st.markdown(f"- **{key}:** {value}")

    return df


def summarize_data(df):
    summary = f"""
### ℹ️ Информация о датасете

- 🔢 **Размер:** {df.shape[0]} строк, {df.shape[1]} столбцов
- 📄 **Типы данных:**
{df.dtypes.to_string()}

- 🧩 **Пропуски:**
{df.isnull().sum()[df.isnull().sum() > 0].to_string() if df.isnull().sum().sum() > 0 else 'Нет пропусков'}

- 🧪 **Пример данных:**
{df.head(3).to_markdown()}
"""
    return summary


def summarize_columns_for_gpt(df):
    summary = []
    for col in df.columns:
        null_pct = round(df[col].isnull().mean() * 100, 1)
        examples = df[col].dropna().astype(str).unique()[:3]
        summary.append({
            "column": col,
            "type": str(df[col].dtype),
            "nulls": null_pct,
            "examples": list(examples)
        })
    return summary


def ask_gpt_smart_cleaning(summary):
    prompt = (
        "На основе этой информации укажи, какие столбцы нужно заполнить и чем "
        "(mean, median или mode), а какие не трогать. Формат:\n\n"
        "- Заполнить: age: mean, city: mode\n"
        "- Не трогать: country, code\n\n"
        "Только список, без лишних слов.\n\n"
        f"{summary}"
    )
    return get_chatgpt_response(prompt)  # тут вызывается модель


def apply_gpt_cleaning(df, gpt_response):
    import re
    to_fill = {}
    do_not_touch = []

    # Извлечение рекомендаций GPT
    fill_part = re.search(r"- Заполнить:\s*(.*?)(?:\n|$)", gpt_response)
    if fill_part:
        for item in fill_part.group(1).split(','):
            parts = item.strip().split(':')
            if len(parts) == 2:
                col, method = parts[0].strip(), parts[1].strip()
                if col in df.columns:
                    to_fill[col] = method

    notouch_part = re.search(r"- Не трогать:\s*(.*)", gpt_response)
    if notouch_part:
        do_not_touch = [x.strip() for x in notouch_part.group(1).split(',') if x.strip() in df.columns]

    # Обработка и лог
    cleaning_log = []

    for col in df.columns:
        if col in to_fill:
            method = to_fill[col]
            if df[col].isnull().sum() > 0:
                try:
                    if method == "mean":
                        df[col].fillna(df[col].mean(), inplace=True)
                        cleaning_log.append(f"{col}: заполнено (mean)")
                    elif method == "median":
                        df[col].fillna(df[col].median(), inplace=True)
                        cleaning_log.append(f"{col}: заполнено (median)")
                    elif method == "mode":
                        mode_val = df[col].mode()
                        if not mode_val.empty:
                            df[col].fillna(mode_val[0], inplace=True)
                            cleaning_log.append(f"{col}: заполнено (mode)")
                        else:
                            cleaning_log.append(f"{col}: не удалось заполнить (пустой mode)")
                    else:
                        cleaning_log.append(f"{col}: неизвестный метод заполнения ({method})")
                except Exception as e:
                    cleaning_log.append(f"{col}: ошибка при заполнении → {str(e)}")
            else:
                cleaning_log.append(f"{col}: пропусков нет, не требуется заполнение")
        elif col in do_not_touch:
            cleaning_log.append(f"{col}: оставлен без изменений")
        else:
            if df[col].isnull().sum() > 0:
                cleaning_log.append(f"{col}: содержит пропуски, но не указан в инструкциях")
            else:
                cleaning_log.append(f"{col}: без пропусков")

    # Сохраняем лог в сессию для отображения
    st.session_state["cleaning_log"] = cleaning_log
    return cleaning_log


# Анализ качества данных

def analyze_data_quality(df: pd.DataFrame) -> str:
    total_rows = len(df)
    total_cols = len(df.columns)
    missing_total = df.isnull().sum().sum()
    duplicated_rows = df.duplicated().sum()
    dtypes_summary = df.dtypes.value_counts().to_dict()

    dtype_text = ", ".join([f"{v} столбцов с типом {k}" for k, v in dtypes_summary.items()])

    summary = f"""
✅ В вашем наборе данных {total_rows} строк и {total_cols} столбцов.

🧼 Очистка данных:
- Было удалено {duplicated_rows} дублирующих строк.
- Пропущенных значений: {'нет' if missing_total == 0 else missing_total}.

🔎 Типы данных: {dtype_text}.

🎉 Данные готовы к дальнейшему анализу и обучению модели.
"""
    return summary


# Отчёт об изменениях после очистки

def generate_data_cleaning_report(df_original, df_cleaned):
    report = []

    rows_before = df_original.shape[0]
    rows_after = df_cleaned.shape[0]
    dups_removed = df_original.duplicated().sum()
    rows_removed = rows_before - rows_after

    report.append(f"- 🧾 Исходных строк: **{rows_before}**, после очистки: **{rows_after}**")
    report.append(f"- ❌ Удалено дубликатов: **{dups_removed}**")
    report.append(f"- 🧹 Всего удалено строк: **{rows_removed}** (дубликаты + строки с >50% NaN)")

    nan_percent = df_original.isnull().mean().round(2) * 100
    if nan_percent[nan_percent > 0].empty:
        report.append("- ✅ Пропущенных значений до очистки не было.")
    else:
        report.append("### 🔍 Пропущенные значения до очистки:")
        report.append(nan_percent[nan_percent > 0].to_markdown())

    report.append("### 📦 Типы данных после очистки:")
    report.append(df_cleaned.dtypes.to_frame("Тип").to_markdown())

    unique_vals = df_cleaned.nunique()
    report.append("### 🧬 Уникальные значения по столбцам:")
    report.append(unique_vals.to_frame("Количество уникальных").to_markdown())

    return "\n\n".join(report)


def show_data_issues(issues_dict):
    st.subheader("🛠 Качество данных:")
    for k, v in issues_dict.items():
        st.markdown(f"**{k}:**")
        st.dataframe(v)


# 3. Визуализация данных
def plot_data_visualizations(df, x, y=None, top_n=None, numeric_filters=None, chart_type="Автоматически"):
    try:
        if y and x == y:
            st.warning("Вы выбрали одинаковые столбцы для осей X и Y. Пожалуйста, выберите разные переменные.")
            return None

        if numeric_filters:
            for col, (min_val, max_val) in numeric_filters.items():
                df = df[df[col].between(min_val, max_val)]

        if top_n:
            if not pd.api.types.is_numeric_dtype(df[x]):
                top_categories_x = df[x].value_counts().nlargest(top_n).index
                df = df[df[x].isin(top_categories_x)]
            if y and not pd.api.types.is_numeric_dtype(df[y]):
                top_categories_y = df[y].value_counts().nlargest(top_n).index
                df = df[df[y].isin(top_categories_y)]

        x_is_num = pd.api.types.is_numeric_dtype(df[x])
        y_is_num = pd.api.types.is_numeric_dtype(df[y]) if y else False

        # --- Обработка ручного выбора ---
        if chart_type != "Автоматически":
            st.info(f"Выбранный тип графика: **{chart_type}**")
            if chart_type == "Гистограмма":
                return px.histogram(df, x=x, nbins=30, title=f'Histogram: {x}')
            elif chart_type == "Круговая диаграмма":
                counts = df[x].value_counts()
                if top_n:
                    counts = counts.nlargest(top_n)
                return px.pie(values=counts.values, names=counts.index, title=f'Pie Chart: {x}')
            elif chart_type == "Точечный график" and y:
                return px.scatter(df, x=x, y=y, title=f'Scatter: {x} vs {y}')
            elif chart_type == "Boxplot" and y:
                return px.box(df, x=x, y=y, title=f'Boxplot: {x} vs {y}')
            elif chart_type == "Bar-график" and y:
                return px.bar(df, x=x, y=y, title=f'Bar: {x} vs {y}')
            else:
                st.warning("Для выбранного типа графика необходима вторая переменная Y.")
                return None

        # --- Автоматический режим ---
        if y:
            if x_is_num and y_is_num:
                st.info(f"График: Scatter — числовая взаимосвязь `{x}` и `{y}`")
                return px.scatter(df, x=x, y=y, title=f'Scatter: {x} vs {y}')

            elif y_is_num and not x_is_num:
                if df[x].nunique() <= 5:
                    st.info(f"График: Bar — среднее значение `{y}` по `{x}`")
                    agg_df = df.groupby(x)[y].mean().reset_index()
                    return px.bar(agg_df, x=x, y=y, title=f'Bar: {x} vs {y}')
                else:
                    st.info(f"График: Boxplot — распределение `{y}` по `{x}`")
                    return px.box(df, x=x, y=y, title=f'Boxplot: {x} vs {y}')

            elif not x_is_num and not y_is_num:
                st.info(f"График: Bar — категории `{x}` и цвет по `{y}`")
                return px.histogram(df, x=x, color=y, barmode="group", title=f'Bar: {x} by {y}')

            else:
                return px.bar(df, x=x, y=y, title=f'Bar: {x} vs {y}')
        else:
            if x_is_num:
                st.info(f"График: Histogram — распределение значений `{x}`")
                return px.histogram(df, x=x, nbins=30, title=f'Histogram: {x}')
            else:
                counts = df[x].value_counts()
                if top_n:
                    counts = counts.nlargest(top_n)
                st.info(f"График: Pie — распределение категорий по `{x}`")
                return px.pie(values=counts.values, names=counts.index, title=f'Pie Chart: {x}')

    except Exception as e:
        st.error(f"Ошибка визуализации: {e}")
        return None
    

# Обучение модели
def train_model(df, target_column, model_type):
    df = df.copy()
    st.info("🚀 Начинаем обучение модели...")

    # Разделим признаки и целевую переменную
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Кодирование категориальных признаков
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Проверка на пропущенные значения
    if X.isnull().any().any() or y.isnull().any():
        st.error(
            "⚠️ Обнаружены пропущенные значения в данных. "
            "Модель не может быть обучена, пока они не будут обработаны. "
            "Пожалуйста, попробуйте:\n"
            "- воспользоваться функцией умной очистки,\n"
            "- или выбрать другую целевую переменную,\n"
            "- или предварительно заполнить/удалить пропуски вручную."
        )
        return None, None, None, None, None

    # Масштабирование признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Кодирование целевой переменной, если нужно
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y.astype(str))

    # Разделение на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Инициализация модели
    if model_type == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_type == "Logistic Regression":
        model = LogisticRegression()
    elif model_type == "Neural Network":
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    else:
        st.error("❌ Неподдерживаемый тип модели.")
        return None, None, None, None, None

    # Обучение модели
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Вычисление метрик
    metrics = classification_report(y_test, y_pred, output_dict=True)
    st.success("✅ Модель обучена успешно!")

    return metrics, model, X_test, y_test, y_pred

# 5. Визуализация предсказаний
def plot_predictions(y_test, y_pred):
    st.subheader("📊 Матрица ошибок (Confusion Matrix)")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истинное значение")
    st.pyplot(fig)