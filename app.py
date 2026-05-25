import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Настройка страницы
st.set_page_config(page_title="РГР: Air Quality Prediction", layout="wide")

# 2. Функция загрузки моделей и скалера
@st.cache_resource
def load_models():
    with open('models_data.pkl', 'rb') as f:
        return pickle.load(f)

data = load_models()

# 3. Навигация в боковой панели
page = st.sidebar.selectbox("Выберите раздел", [
    "О разработчике", 
    "Датасет и EDA", 
    "Визуализации", 
    "Предсказание (Инференс)"
])

# СТРАНИЦА 1: О РАЗРАБОТЧИКЕ
if page == "О разработчике":
    st.title("Расчетно-графическая работа")
    st.error("**Тема РГР:** Разработка Web-приложения (дашборда) для инференса моделей ML и анализа данных качества атмосферного воздуха")
    
    col_photo, col_info = st.columns([1, 2])
    
    with col_photo:
        st.image("photo.jpg", use_container_width=True)
        
    with col_info:
        st.markdown("""
        * **ФИО:** Лейба Виктория Витальевна.
        * **Учебная группа:** ФИТ-241
        * **Используемый стек технологий:** Python, Streamlit, Scikit-Learn, CatBoost, Git
        """)

# СТРАНИЦА 2: DATASET & EDA
elif page == "Датасет и EDA":
    st.title("Описание предметной области и предобработки данных")
    
    st.markdown("""
    ### 1. Предметная область
    Данный датасет содержит результаты мониторинга качества атмосферного воздуха в итальянском городе за период с марта 2004 по февраль 2005 года. 
    Сбор данных осуществлялся автоматическим газоанализатором со встроенными химическими мультисенсорными датчиками оксидов металлов. 
    
    ### 2. Описание признаков (переменных)
    * **CO(GT)** — Истинная концентрация угарного газа в воздухе (**Целевая переменная для регрессии**).
    * **PT08.S1** — Отклик сенсора оксида олова (CO).
    * **C6H6(GT)** — Концентрация бензола.
    * **PT08.S2** — Отклик сенсора неметановых углеводородов.
    * **NOx(GT)** — Концентрация оксидов азота.
    * **PT08.S3** — Отклик сенсора оксида вольфрама (NOx).
    * **NO2(GT)** — Концентрация диоксида азота.
    * **PT08.S4** — Отклик сенсора оксида вольфрама (O3).
    * **PT08.S5** — Отклик сенсора оксида индия.
    * **T** — Температура воздуха.
    * **RH** — Относительная влажность воздуха.
    * **AH** — Абсолютная влажность.

    ### 3. Особенности предобработки данных
    1. **Обработка пропусков:** В исходном датасете пропущенные значения были закодированы числом `-200`. Все такие аномалии были удалены и заменены медианными значениями по соответствующим признакам.
    2. **Масштабирование:** Так как признаки имеют разный порядок величин (отклик сенсора ~1500, а влажность ~1.0), выполнена стандартизация с помощью `StandardScaler`, приводящая данные к $M=0$ и $\sigma=1$.
    """)
    
    df = pd.read_csv('AirQuality_cleaned.csv')
    st.write("### Разведочный анализ (EDA): Первые 5 строк очищенного датасета")
    st.dataframe(df.head())
    
    st.write("### Основные статистические характеристики")
    st.write(df.describe())

# СТРАНИЦА 3: ВИЗУАЛИЗАЦИИ
elif page == "Визуализации":
    st.title("Визуальный анализ зависимостей")
    df = pd.read_csv('AirQuality_cleaned.csv')
    
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    with row1_col1:
        st.write("#### 1. Матрица корреляций (Heatmap)")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.heatmap(df.select_dtypes(include=[np.number]).corr(), cmap='coolwarm', ax=ax1, cbar=False)
        st.pyplot(fig1)
    
    with row1_col2:
        st.write("#### 2. Распределение целевой переменной CO(GT) (Histplot)")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(df['CO(GT)'], kde=True, color='green', ax=ax2)
        st.pyplot(fig2)
        
    with row2_col1:
        st.write("#### 3. Зависимость CO от Бензола C6H6 (Scatterplot)")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x='C6H6(GT)', y='CO(GT)', alpha=0.5, color='orange', ax=ax3)
        st.pyplot(fig3)
        
    with row2_col2:
        st.write("#### 4. Анализ выбросов целевого признака CO(GT) (Boxplot)")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        # Используем проверенную колонку CO(GT)
        sns.boxplot(x=df['CO(GT)'], color='skyblue', ax=ax4)
        st.pyplot(fig4)

# СТРАНИЦА 4: ИНФЕРЕНС (ПРЕДСКАЗАНИЕ)
elif page == "Предсказание (Инференс)":
    st.title("Инференс моделей машинного обучения")
    
    input_mode = st.radio("Выберите способ подачи данных для моделей:", [
        "Ручной ввод через интерактивные виджеты", 
        "Пакетная загрузка файла *.csv"
    ])
    
    if input_mode == "Ручной ввод через интерактивные виджеты":
        st.write("### Отрегулируйте параметры сенсоров и метеоусловия:")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            s1 = st.slider("PT08.S1 (Отклик сенсора CO)", 600, 2000, 1000)
            c6h6 = st.slider("C6H6 (Концентрация бензола, мг/м³)", 0.1, 60.0, 10.0)
            s2 = st.slider("PT08.S2 (Отклик сенсора углеводородов)", 400, 2000, 900)
        with c2:
            nox = st.slider("NOx (Концентрация оксидов азота, ppb)", 2.0, 1500.0, 150.0)
            s3 = st.slider("PT08.S3 (Отклик сенсора оксида вольфрама)", 300, 2700, 800)
            no2 = st.slider("NO2 (Диоксид азота, мг/м³)", 2.0, 340.0, 100.0)
        with c3:
            s4 = st.slider("PT08.S4 (Отклик сенсора O3)", 500, 2800, 1400)
            s5 = st.slider("PT08.S5 (Отклик сенсора оксида индия)", 200, 2500, 1000)
            temp = st.slider("T (Температура воздуха, °C)", -10.0, 50.0, 20.0)
            rh = st.slider("RH (Относительная влажность, %)", 0.0, 100.0, 50.0)
            ah = st.slider("AH (Абсолютная влажность, г/м³)", 0.1, 2.3, 1.0)

        # Подготовка вектора
        input_data = np.array([[s1, c6h6, s2, nox, s3, no2, s4, s5, temp, rh, ah]])
        
        if st.button("Рассчитать прогноз по введенным данным"):
            input_scaled = data['scaler'].transform(input_data)
            results = {}
            
            # ML1: Polynomial
            input_poly = data['poly'].transform(input_scaled)
            results['ML1: Polynomial Regression'] = data['models']['ML1'].predict(input_poly)[0]
            
            # Остальные ансамбли и нейросеть
            for m_name in ['ML2', 'ML3', 'ML4', 'ML5', 'ML6']:
                pred = data['models'][m_name].predict(input_scaled)
                results[m_name] = pred[0] if isinstance(pred, np.ndarray) else pred
            
            st.write("### Итоговые результаты прогнозирования концентрации CO:")
            
            res_list = []
            for name, val in results.items():
                status = "Норма" if val < 3.0 else "Повышенное загрязнение"
                res_list.append({
                    "Модель / Алгоритм": name,
                    "Прогноз концентрации CO (мг/м³)": f"{max(0, val):.3f} мг/м³",
                    "Экологический статус": status
                })
            
            st.table(pd.DataFrame(res_list))

    else:
        st.write("### Загрузите файл параметров в формате *.csv для пакетного прогноза")
        uploaded_file = st.file_uploader("Выберите файл *.csv", type=["csv"])
        
        if uploaded_file is not None:
            uploaded_df = pd.read_csv(uploaded_file)
            st.write(" Загруженные входные данные (первые строки):")
            st.dataframe(uploaded_df.head())
            
            # Быстрая валидация: проверка, что в файле ровно 11 нужных колонок
            if uploaded_df.shape[1] >= 11:
              if st.button("Запустить пакетный инференс по файлу"):
                    # 1. Оставляем только числовые колонки (отбрасываем Date/Time)
                    numeric_cols = uploaded_df.select_dtypes(include=[np.number])
                    
                    # 2. Убираем целевую переменную CO(GT), если она есть в файле
                    if 'CO(GT)' in numeric_cols.columns:
                        numeric_cols = numeric_cols.drop(columns=['CO(GT)'])
                    
                    # 3. Берем ровно 11 оставшихся признаков (сенсоры и метеоусловия)
                    raw_features = numeric_cols.iloc[:, :11].values
                    scaled_features = data['scaler'].transform(raw_features)
                    
                    # Считаем лучшей моделью ML3 (CatBoost)
                    predictions = data['models']['ML3'].predict(scaled_features)
                    
                    output_df = uploaded_df.copy()
                    output_df['Прогноз CO (CatBoost, мг/м³)'] = [f"{max(0, x):.2f} мг/м³" for x in predictions]
                    
                    st.success("Пакетный расчет успешно завершен!")
                    st.dataframe(output_df)
            else:
                st.error("Ошибка валидации: Файл должен содержать как минимум 11 столбцов с признаками датчиков!")