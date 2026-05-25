import pandas as pd
import numpy as np
import joblib  # Используем joblib вместо pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
import os

# 1. Загрузка данных
df = pd.read_csv('AirQuality_cleaned.csv')
X = df[['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']]
y = df['CO(GT)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение (оставил все модели)
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train_scaled)
ml1 = LinearRegression().fit(X_poly_train, y_train)

ml2 = GradientBoostingRegressor(random_state=42).fit(X_train_scaled, y_train)
ml3 = CatBoostRegressor(verbose=0, random_state=42).fit(X_train_scaled, y_train)
ml4 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_scaled, y_train)
ml5 = StackingRegressor(estimators=[('rf', ml4), ('gb', ml2)], final_estimator=LinearRegression()).fit(X_train_scaled, y_train)
ml6 = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42).fit(X_train_scaled, y_train)

# 2. Сериализация с сжатием через joblib
save_data = {
    'scaler': scaler,
    'poly': poly,
    'models': {'ML1': ml1, 'ML2': ml2, 'ML3': ml3, 'ML4': ml4, 'ML5': ml5, 'ML6': ml6}
}

# compress=3 обеспечивает хороший баланс размера и скорости
joblib.dump(save_data, 'models_data.pkl', compress=3)

print(f"Файл сохранен. Размер: {os.path.getsize('models_data.pkl') / (1024 * 1024):.2f} МБ")