import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

import plotly.express as px
import re
from AI_helper import get_chatgpt_response

def load_data(uploaded_file):
    import re

    filename = uploaded_file.name.lower()
    try:
        if filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            raise ValueError("Неподдерживаемый формат файла. Используйте CSV, XLSX или XLS.")
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")
        raise

    conversion_log = []

    def looks_like_number(s):
        """Проверка: выглядит ли строка как число"""
        s = s.strip().replace(",", ".")
        return bool(re.match(r"^-?\d+(\.\d+)?$", s))

    for col in df.columns:
        original_dtype = df[col].dtype
        if original_dtype == "object":
            # Удалим пробелы и заменим запятые на точки
            df[col] = df[col].astype(str).str.strip().str.replace(",", ".")

            # Проверим, сколько значений выглядят как числа
            is_number_like = df[col].apply(looks_like_number)
            success_rate = is_number_like.mean()

            if success_rate > 0.9:
                try:
                    df[col] = pd.to_numeric(df[col], errors="raise")
                    conversion_log.append(f"{col}: object → float (успешно, {success_rate:.0%} чисел)")
                except Exception:
                    df[col] = df[col].astype(str)
                    conversion_log.append(f"{col}: остался как текст (ошибка при преобразовании)")
            else:
                conversion_log.append(f"{col}: остался как текст ({success_rate:.0%} числовых строк)")
        else:
            conversion_log.append(f"{col}: {original_dtype} (без изменений)")

    st.session_state["conversion_log"] = conversion_log

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
    for key, value in base_info.items():
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

    # Извлечение рекомендаций GPT для заполнения
    fill_part = re.search(r"- Заполнить:\s*(.*?)(?:\n|$)", gpt_response)
    if fill_part:
        for item in fill_part.group(1).split(','):
            parts = item.strip().split(':')
            if len(parts) == 2:
                col, method = parts[0].strip(), parts[1].strip()
                if col in df.columns:
                    to_fill[col] = method

    # Извлечение рекомендаций GPT для оставления без изменений
    notouch_part = re.search(r"- Не трогать:\s*(.*)", gpt_response)
    if notouch_part:
        do_not_touch = [x.strip() for x in notouch_part.group(1).split(',') if x.strip() in df.columns]

    cleaning_log = []

    # Проходим по всем колонкам и применяем инструкции или дефолтную очистку
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
            # Если для колонки не указаны инструкции, и в ней есть пропуски, применяем дефолтное заполнение
            if df[col].isnull().sum() > 0:
                # Автоматически очищаем колонку по умолчанию
                result = default_cleaning(df, col)
                cleaning_log.append(f"{col}: содержит пропуски, не указан в инструкциях -> {result}")
            else:
                cleaning_log.append(f"{col}: без пропусков")
    
    # Сохраняем лог очистки в session_state
    st.session_state["cleaning_log"] = cleaning_log
    return cleaning_log


def default_cleaning(df, column):
    """
    Очищает колонку по умолчанию:
    - Если числовая – заполняет пропуски средним.
    - Если категориальная – заполняет наиболее частым значением (mode).
    """
    import pandas as pd
    if pd.api.types.is_numeric_dtype(df[column]):
        df[column].fillna(df[column].mean(), inplace=True)
        return "очищено (mean)"
    else:
        mode_val = df[column].mode()
        if not mode_val.empty:
            df[column].fillna(mode_val[0], inplace=True)
            return "очищено (mode)"
        else:
            return "очищение не выполнено (пустой mode)"


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

        # Проверка, является ли столбец ID или уникальным
        if df[x].nunique() == len(df[x]):
            st.warning("Вы выбрали уникальный идентификатор (например, ID). Для этого типа данных визуализация не имеет смысла.")
            return None

        # --- Обработка ручного выбора ---
        if chart_type != "Автоматически":
            st.info(f"Выбранный тип графика: **{chart_type}**")
            if chart_type == "Гистограмма":
                if x_is_num:
                    return px.histogram(
                        df,
                        x=x,
                        nbins=30,
                        title=f'Histogram: {x}',
                        color_discrete_sequence=px.colors.qualitative.Vivid  # Исправлено для работы с категориальными цветами
                    )
                else:
                    return px.histogram(
                        df,
                        x=x,
                        title=f'Histogram: {x}',
                        color=x,
                        color_discrete_sequence=px.colors.qualitative.Safe
                    )
            elif chart_type == "Круговая диаграмма":
                counts = df[x].value_counts()
                if top_n:
                    counts = counts.nlargest(top_n)
                return px.pie(
                    values=counts.values,
                    names=counts.index,
                    title=f'Pie Chart: {x}',
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
            elif chart_type == "Точечный график" and y:
                if not x_is_num:
                    color_param = x
                elif not y_is_num:
                    color_param = y
                else:
                    color_param = None
                if color_param:
                    return px.scatter(
                        df,
                        x=x,
                        y=y,
                        title=f'Scatter: {x} vs {y}',
                        color=color_param,
                        color_discrete_sequence=px.colors.qualitative.Dark2
                    )
                else:
                    return px.scatter(
                        df,
                        x=x,
                        y=y,
                        title=f'Scatter: {x} vs {y}',
                        color_continuous_scale=px.colors.sequential.Inferno
                    )
            elif chart_type == "Boxplot" and y:
                if not x_is_num:
                    return px.box(
                        df,
                        x=x,
                        y=y,
                        title=f'Boxplot: {x} vs {y}',
                        color=x,
                        color_discrete_sequence=px.colors.qualitative.Pastel2
                    )
                else:
                    return px.box(
                        df,
                        x=x,
                        y=y,
                        title=f'Boxplot: {x} vs {y}'
                    )
            elif chart_type == "Bar-график" and y:
                if not x_is_num:
                    return px.bar(
                        df,
                        x=x,
                        y=y,
                        title=f'Bar: {x} vs {y}',
                        color=x,
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                else:
                    return px.bar(
                        df,
                        x=x,
                        y=y,
                        title=f'Bar: {x} vs {y}'
                    )
            elif chart_type == "Лайнплот" and x and y:
                return px.line(
                    df,
                    x=x,
                    y=y,
                    title=f'Line Plot: {x} vs {y}',
                    markers=True,
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
            else:
                st.warning("Для выбранного типа графика необходима вторая переменная Y.")
                return None

        # --- Автоматический режим ---
        if y:
            if x_is_num and y_is_num:
                st.info(f"График: Scatter — числовая взаимосвязь `{x}` и `{y}`")
                return px.scatter(
                    df, x=x, y=y,
                    title=f'Scatter: {x} vs {y}',
                    color_continuous_scale=px.colors.sequential.Inferno
                )
            elif y_is_num and not x_is_num:
                if df[x].nunique() <= 5:
                    st.info(f"График: Bar — среднее значение `{y}` по `{x}`")
                    agg_df = df.groupby(x)[y].mean().reset_index()
                    return px.bar(
                        agg_df,
                        x=x,
                        y=y,
                        title=f'Bar: {x} vs {y}',
                        color=x,
                        color_discrete_sequence=px.colors.qualitative.Safe
                    )
                else:
                    st.info(f"График: Boxplot — распределение `{y}` по `{x}`")
                    return px.box(
                        df,
                        x=x,
                        y=y,
                        title=f'Boxplot: {x} vs {y}',
                        color=x,
                        color_discrete_sequence=px.colors.qualitative.Pastel2
                    )
            elif not x_is_num and not y_is_num:
                st.info(f"График: Bar — категории `{x}` с разделением по `{y}`")
                return px.histogram(
                    df,
                    x=x,
                    color=y,
                    barmode="group",
                    title=f'Bar: {x} by {y}',
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
            else:
                return px.bar(
                    df,
                    x=x,
                    y=y,
                    title=f'Bar: {x} vs {y}',
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
        else:
            if x_is_num:
                st.info(f"График: Histogram — распределение значений `{x}`")
                return px.histogram(
                    df,
                    x=x,
                    nbins=30,
                    title=f'Histogram: {x}',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
            else:
                counts = df[x].value_counts()
                if top_n:
                    counts = counts.nlargest(top_n)
                st.info(f"График: Pie — распределение категорий по `{x}`")
                return px.pie(
                    values=counts.values,
                    names=counts.index,
                    title=f'Pie Chart: {x}',
                    color_discrete_sequence=px.colors.qualitative.Bold
                )

    except Exception as e:
        st.error(f"Ошибка визуализации: {e}")
        return None




# Обучение модели с тщательной проверкой данных
def train_model(df, target_column, model_type):
    df = df.copy()
    st.info("🚀 Начинаем обучение модели...")

    # Проверка целевой переменной
    if target_column not in df.columns:
        st.error("❌ Целевая переменная не найдена в данных.")
        return None, None, None, None, None

    if df[target_column].nunique() < 2:
        st.error(
            "❌ Целевая переменная должна иметь как минимум два уникальных значения. "
            "Пожалуйста, выберите другой столбец."
        )
        return None, None, None, None, None

    # Разделение признаков и целевой переменной
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Исключение явно идентификаторных столбцов (например, ID)
    for col in X.columns:
        if X[col].nunique() == len(X[col]):
            st.warning(f"⚠️ Столбец '{col}' выглядит как уникальный идентификатор и будет исключён из обучения.")
            X = X.drop(columns=[col])

    # Кодирование категориальных признаков
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Проверка на пропущенные значения
    if X.isnull().any().any() or y.isnull().any():
        st.error(
            "⚠️ Обнаружены пропущенные значения в данных. "
            "Модель не может быть обучена, пока они не будут обработаны. "
            "Пожалуйста, воспользуйтесь функцией умной очистки или предобработкой данных."
        )
        return None, None, None, None, None

    # Масштабирование признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Кодирование целевой переменной, если это необходимо
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y.astype(str))

    # Разделение на обучающую и тестовую выборки (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Инициализация модели на основе выбранного типа
    if model_type == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=200, random_state=42)
    elif model_type == "Neural Network":
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    elif model_type == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "Support Vector Machine":
        model = SVC(probability=True, random_state=42)
    elif model_type == "K Nearest Neighbors":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
    else:
        st.error("❌ Неподдерживаемый тип модели.")
        return None, None, None, None, None

    # Обучение модели
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Вычисление метрик классификации
    metrics = classification_report(y_test, y_pred, output_dict=True)
    st.success("✅ Модель обучена успешно!")

    return metrics, model, X_test, y_test, y_pred

def plot_predictions(y_test, y_pred):
    st.subheader("📊 Анализ предсказаний")
    
    # Визуализация распределения истинных и предсказанных значений
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Распределение истинных значений**")
        fig_true, ax_true = plt.subplots()
        sns.histplot(y_test, kde=False, color="green", ax=ax_true, bins=10)
        ax_true.set_title("Истинные значения")
        ax_true.set_xlabel("Классы")
        ax_true.set_ylabel("Количество")
        st.pyplot(fig_true)

    with col2:
        st.markdown("**Распределение предсказанных значений**")
        fig_pred, ax_pred = plt.subplots()
        sns.histplot(y_pred, kde=False, color="blue", ax=ax_pred, bins=10)
        ax_pred.set_title("Предсказанные значения")
        ax_pred.set_xlabel("Классы")
        ax_pred.set_ylabel("Количество")
        st.pyplot(fig_pred)

    # Краткий вывод метрик
    st.markdown("**Ключевые метрики классификации:**")
    accuracy = (y_test == y_pred).mean() * 100
    st.success(f"Точность модели: {accuracy:.2f}%")
