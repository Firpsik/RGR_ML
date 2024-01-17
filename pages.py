import streamlit as st
import seaborn as sns
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.metrics import accuracy_score, rand_score

data = pd.read_csv('C:/Venv/notebooks/my_card_transdata.csv', sep=";")
data = data.drop(columns=["Unnamed: 0"])
X = data.drop('fraud', axis=1)
y = data['fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def load_models():
    model1 = pickle.load(open('SVM.pkl', 'rb'))
    model2 = pickle.load(open('kmeans.pkl', 'rb'))
    model3 = pickle.load(open('Stacking.pkl', 'rb'))
    model4 = pickle.load(open('GradientBoosting.pkl', 'rb'))
    model5 = pickle.load(open('Bagging.pkl', 'rb'))
    model6 = load_model('Neural.h5')
    return model1, model2, model3, model4, model5, model6

st.markdown("""
<style>
h1 {
    color: #0e1123;
}
</style>
""", unsafe_allow_html=True)


# Функции для каждой страницы
def info_page():
    st.title("Информация о разработчике:")

    st.header("Тема РГР")
    st.write("Разработка Web-приложения для инференса моделей ML и анализа данных")

    st.header("Фотография разработчика")
    st.image("photo.jpg", width=333)  
    st.header("Контактная информация")
    st.write("ФИО: Козлов Дмитрий Владимирович")
    st.write("Номер учебной группы: МО-221")


def dataset_info_page():
    st.title("Информация о наборе данных")

    st.markdown("""
    ## Описание датасета "my_card_transdata.csv"

    **Описание:**
    Данный датасет содержит табличные данные с различными показаниями покупок. Включает следующие столбцы:

    - **distance_from_home**: Расстояние от дома, на котором была совершена транзакция, в километрах.
    - **distance_from_last_transaction**: Расстояние от места последней транзакции до места текущей, в километрах.
    - **ratio_to_median_purchase_price**: Отношение цены текущей покупки к медианной цене покупок, безразмерный коэффициент.
    - **repeat_retailer**: Индикатор повторной покупки у того же ритейлера (истина если да, ложь если нет).
    - **used_chip**: Индикатор использования микросхемы во время транзакции (истина если использовалась, ложь если нет).
    - **used_pin_number**: Индикатор использования ПИН-кода во время транзакции (истина если использовался, ложь если нет).
    - **online_order**: Индикатор выполнения заказа через интернет (истина если заказ онлайн, ложь если заказ оффлайн).
    - **fraud**: Признак мошеннической транзакции (истина если транзакция мошенническая, ложь если транзакция легальная).

    **Особенности предобработки данных:**
    - Обработка пропущенных значений.
    - Кодирование категориальных переменных.
    """)


def data_visualization_page():
  st.header("Гистограммы")
  columns = ['distance_from_home','ratio_to_median_purchase_price','distance_from_last_transaction']

  for col in columns:
      plt.figure(figsize=(8, 6))
  
      xlim = (data[col].min(), data[col].mean())
      
      sns.histplot(data.sample(5000)[col], bins=1000)
      plt.title(f'Гистограмма для {col}')
      plt.xlim(xlim)
      st.pyplot(plt)


def ml_prediction_page():
    st.title("Предсказания моделей машинного обучения")

    uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv")

    if uploaded_file is None:
        st.subheader("Введите данные:")

        input_data = {}
        feature_names = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price', 'repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']
        for feature in feature_names:
            input_data[feature] = st.number_input(feature, min_value=0.0, value=10.0)

        if st.button('Сделать предсказание'):
            model1, model2, model3, model4, model5, model6 = load_models()

            input_df = pd.DataFrame([input_data])

            st.write("Входные данные:", input_df)

            # Делаем предсказания
            prediction_ml1 = model1.predict(input_df)
            prediction_ml2 = model2.predict(input_df)
            prediction_ml3 = model3.predict(input_df)
            prediction_ml4 = model4.predict(input_df)
            prediction_ml5 = model5.predict(input_df)
            prediction_ml6 = (model6.predict(input_df) > 0.5).astype(int)

            # Вывод результатов
            st.success(f"Результат предсказания SVC: {prediction_ml1}")
            st.success(f"Результат предсказания kmeans.pkl: {prediction_ml2}")
            st.success(f"Результат предсказания Stacking: {prediction_ml3}")
            st.success(f"Результат предсказания GradientBoosting: {prediction_ml4}")
            st.success(f"Результат предсказания Bagging: {prediction_ml5}")
            st.success(f"Результат предсказания Neural: {prediction_ml6}")
    else:
        try:
            model1=pickle.load(open('C:/Venv/notebooks/SVM.pkl', 'rb'))
            model2=pickle.load(open('C:/Venv/notebooks/kmeans.pkl', 'rb'))
            model3=pickle.load(open('C:/Venv/notebooks/Stacking.pkl', 'rb'))
            model4=pickle.load(open('C:/Venv/notebooks/GradientBoosting.pkl', 'rb'))
            model5=pickle.load(open('C:/Venv/notebooks/Bagging.pkl', 'rb'))
            model6 = load_model('C:/Venv/notebooks/Neural.h5')

            # Делаем предсказания на тестовых данных
            predictions_ml1 = model1.predict(X_test)
            predictions_ml2  = model2.fit_predict(X_test)
            predictions_ml3 = model3.predict(X_test)
            predictions_ml4 = model4.predict(X_test)
            predictions_ml5 = model5.predict(X_test)
            predictions_ml6 = model6.predict(X_test).round()

            accuracy_ml1 = accuracy_score(y_test, predictions_ml1)
            accuracy_ml2 = round(rand_score(y_test, predictions_ml2))
            accuracy_ml3 = accuracy_score(y_test, predictions_ml3)
            accuracy_ml4 = accuracy_score(y_test, predictions_ml4)
            accuracy_ml5 = accuracy_score(y_test, predictions_ml5)
            accuracy_ml6 = accuracy_score(y_test, predictions_ml6)

            st.success(f"Точность SVC: {accuracy_ml1}")
            st.success(f"Точность Kmeans: {accuracy_ml2}")
            st.success(f"Точность Stacking: {accuracy_ml3}")
            st.success(f"Точность GradienBoosting: {accuracy_ml4}")
            st.success(f"Точность Bagging: {accuracy_ml5}")
            st.success(f"Точность Neural: {accuracy_ml6}")

        except Exception as e:
            st.error(f"Произошла ошибка при чтении файла: {e}")

pages = {
    "Информация о разработчике": info_page,
    "Информация о наборе данных": dataset_info_page,
    "Визуализации данных": data_visualization_page,
    "Предсказание модели ML": ml_prediction_page
}

st.sidebar.title("Перемещение")
page = st.sidebar.selectbox("Выберите страницу:", list(pages.keys()))

pages[page]()