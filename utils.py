import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
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
        "Только список, без лишних слов!!! (не надо Практические рекомендации!!!) Строго отвечай по формату не надо чтото обясниить внутри формата.\n\n"
        f"{summary}"
    )
    return get_chatgpt_response(prompt)


        
def remove_outliers_iqr(df):
    """
    Удаляет выбросы для числовых столбцов в DataFrame по методу IQR.
    
    Для каждого числового столбца вычисляет первый (Q1) и третий (Q3) квартиль, 
    а затем интерквартильный размах (IQR = Q3 - Q1). Строки, где значение столбца 
    выходит за пределы [Q1 - 1.5*IQR, Q3 + 1.5*IQR], исключаются из DataFrame.

    Возвращает:
      - Обновлённый DataFrame
      - Количество удалённых строк (разница между исходным и финальным количеством строк)
    """
    import pandas as pd
    initial_rows = df.shape[0]
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    # Для корректного удаления выбросов применяем фильтрацию для каждой колонки итерационно
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    final_rows = df.shape[0]
    removed_rows = initial_rows - final_rows
    return df, removed_rows


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
                    color_discrete_sequence=px.colors.qualitative.Bold
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
    
def suggest_visualization_combinations(df_info):
    """
    Рекомендует интересные комбинации переменных для визуализации на основе информации о наборах данных.
    
    Используя описание набора переменных (например, список столбцов), сформируйте 2-3 рекомендации для визуализации 
    (например, варианты для осей X и Y), которые могут показать важные закономерности. 
    Ответ должен быть кратким, понятным и без сложных терминов, по одному варианту на строку.
    """
    prompt = (
        f"На основе следующей информации о переменных: {df_info}. "
        "Предложи 2-3 интересные комбинации для визуализации, указав какие переменные использовать для оси X и Y, "
        "чтобы можно было увидеть важные закономерности в данных. "
        "Ответ дай кратко, по одному варианту на строку."
    )
    return get_chatgpt_response(prompt)




# Предсказание модели
def train_model(df, target_column, model_type):
    df = df.copy()
    st.info("🚀 Начинаем обучение модели...")

    # Проверка наличия целевой переменной в данных
    if target_column not in df.columns:
        st.error("❌ Целевая переменная не найдена в данных.")
        return None, None, None, None, None

    # Проверка уникальности значений целевой переменной
    unique_vals = df[target_column].nunique()
    if unique_vals < 2:
        st.error(
            "❌ Целевая переменная должна иметь как минимум два уникальных значения. "
            "Если вы используете бинарную классификацию (например, значения 0 и 1), "
            "убедитесь, что выбранный столбец содержит оба класса."
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
    st.success("✅ Модель обучена успешно! минутку...")

    return metrics, model, X_test, y_test, y_pred

def suggest_optimal_model(df_info):
    """
    Рекомендует оптимальную модель для предсказания на основе характеристик данных.
    Учитывает требование: высокая точность предсказаний и скорость работы.
    Возвращает ответ одним словом (название модели) с кратким пояснением.
    """
    prompt = (
    f"На основе следующих характеристик данных: {df_info}. "
    "Объясни результаты предсказания модели так, чтобы человек без знаний в статистике и анализе данных понял. "
    "Просто расскажи, что значат полученные метрики, как их интерпретировать и что они говорят о качестве предсказаний. "
    "Дай объяснение без сложных терминов, коротко и ясно."
    )

    return get_chatgpt_response(prompt)


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
