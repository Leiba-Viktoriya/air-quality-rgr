import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
import pickle
import os

# 1. Загрузка данных
df = pd.read_csv('AirQuality_cleaned.csv')

# Выбираем признаки
X = df[['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']]
y = df['CO(GT)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models_report = {}

# ML1: Полиномиальная регрессия
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train_scaled)
ml1 = LinearRegression()
ml1.fit(X_poly_train, y_train)
models_report['ML1_Polynomial'] = r2_score(y_test, ml1.predict(poly.transform(X_test_scaled)))

# ML2: Gradient Boosting
ml2 = GradientBoostingRegressor(random_state=42)
ml2.fit(X_train_scaled, y_train)
models_report['ML2_GradientBoosting'] = r2_score(y_test, ml2.predict(X_test_scaled))

# ML3: CatBoost
ml3 = CatBoostRegressor(verbose=0, random_state=42)
ml3.fit(X_train_scaled, y_train)
models_report['ML3_CatBoost'] = r2_score(y_test, ml3.predict(X_test_scaled))

# ML4: Random Forest
ml4 = RandomForestRegressor(n_estimators=100, random_state=42)
ml4.fit(X_train_scaled, y_train)
models_report['ML4_RandomForest'] = r2_score(y_test, ml4.predict(X_test_scaled))

# ML5: Stacking 
estimators = [('rf', ml4), ('gb', ml2)]
ml5 = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
ml5.fit(X_train_scaled, y_train)
models_report['ML5_Stacking'] = r2_score(y_test, ml5.predict(X_test_scaled))

# ML6: MLPRegressor (Нейросеть)
ml6 = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
ml6.fit(X_train_scaled, y_train)
models_report['ML6_MLP'] = r2_score(y_test, ml6.predict(X_test_scaled))

# 2. Сериализация
save_data = {
    'scaler': scaler,
    'poly': poly,
    'models': {
        'ML1': ml1, 'ML2': ml2, 'ML3': ml3, 
        'ML4': ml4, 'ML5': ml5, 'ML6': ml6
    }
}

with open('models_data.pkl', 'wb') as f:
    pickle.dump(save_data, f)

print("Метрики R2 для моделей:")
for name, score in models_report.items():
    print(f"{name}: {score:.4f}")
    save_data = {
    'scaler': scaler,
    'poly': poly,
    'models': {
        'ML1': ml1, 'ML2': ml2, 'ML3': ml3, 
        'ML4': ml4, 'ML5': ml5, 'ML6': ml6
    }
}

# Используем протокол сжатия pickle (protocol=4 или 5)
with open('models_data.pkl', 'wb') as f:
    pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

# Проверка размера
file_size = os.path.getsize('models_data.pkl') / (1024 * 1024)
print(f"Размер файла: {file_size:.2f} МБ")
