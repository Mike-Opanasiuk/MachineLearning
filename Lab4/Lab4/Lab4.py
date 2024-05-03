
# 1. Імпорт бібліотек та модулів
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Встановлення випадкових параметрів для відтворюваності результатів
np.random.seed(80)

# 2. Завантаження даних з диска
XY = pd.read_csv("region4a.csv", delimiter=";")

# 3. Розділення змінних на впливаючі фактори та змінну -відгук
import statsmodels.api as sm
# Визначення факторів (всі колонки, крім колонки з змінною-відгуком eps)
X = XY.drop(columns=['eps'])
# Додати константу до матриці X
X = sm.add_constant(X)
# Визначення залежної змінної
y = XY['eps']

# 4. Видалення одного фактора згідно з варіантом
X0 = X.drop(columns=['t1'])
# print(X0.head()) # перегляд факторів

# 5. Задання розміру навчальної вибірки - 75% від всіх даних
samplesize = int(0.75 * XY.shape[0])
# Розбивка даних на навчальну та тестову вибірки
index = np.random.choice(XY.index, size=samplesize, replace=False)
train = XY.loc[index]
test = XY.drop(index)
# Розділення змінних у впливаючі фактори та змінну відгук
x_train = train.iloc[:, :-1] # впливаючі фактори
y_train = train.iloc[:, -1] # змінна відгук
x_test = test.iloc[:, :-1] # впливаючі фактори тестової вибірки
y_test = test.iloc[:, -1] # змінна відгук тестової вибірки
# print(x_train.head()) # перегляд факторів
# print(y_train.head()) # перегляд відгуку
# Scaler
scaler = StandardScaler()

# 6. Створення екземплярів моделей

models = [
    LinearRegression(),
    Ridge(alpha=10),
    Lasso(alpha=0.1)
]
# Словник з назвами моделей
model_names = {
    "LinearRegression": "Лінійна регресія",
    "Ridge": "Ридж регресія",
    "Lasso": "Лассо регресія"
}

# 7. Навчання моделей
for model in models:
    model.fit(x_train, y_train)
    coefs = model.coef_
    model_name = model_names[type(model).__name__]
    print(f"Коефіцієнти для {model_name}:")
    print(np.around(model.coef_, decimals=2)) # Форматований вивід

# 8. Помилки моделей на навчальній вибірці
for model in models:
    mse = mean_squared_error(y_train, model.predict(x_train))
    # r2 = r2_score(y_train, model.predict(x_train))
    model_name = model_names[type(model).__name__]
    print(f"Test RMSE of {model} = {np.sqrt(mse):.2f}")
    
# 9. Помилки моделей на контрольній вибірці
for model in models:
    mse = mean_squared_error(y_test, model.predict(x_test))
    # r2 = r2_score(y_test, model.predict(x_test))
    model_name = model_names[type(model).__name__]
    print(f"Test RMSE of {model} = {np.sqrt(mse):.2f}")
    
# 10. Розрахунок фактора CV RMSE
# Фактор CV RMSE 

# CV Scores using train data
for model in models:
    CV = KFold(n_splits=4, shuffle=True, random_state=1)
    # Pipeline
    model_pipe = Pipeline([
        ('scaler', scaler),
        ('model', model)
    ])
    model_rmse = cross_val_score(
        model_pipe,
        x_train,
        y_train,
        cv=CV,
        scoring='neg_root_mean_squared_error',
        error_score='raise'
    )
    print(f"CV RMSE of {model} = {-np.mean(model_rmse):.2f}")
# print("\n")
# Порівняйте CV RMSE для різних моделей і зробіть висновок щодо впливу 
# регуляризації на модель LinearRegression    

# 11. Розрахунок CV RMSE для тестових даних
# RMSE Score of test data
for model in models:
    model = model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print(f"Test RMSE of {model} = {np.sqrt(mean_squared_error(prediction,y_test)):.2f}")
    # Порівняйте CV RMSE для різних моделей і зробіть висновок щодо впливу 
    # регуляризації на модель LinearRegression
    
# 12. Візуалізація залишків моделі
import matplotlib.pyplot as plt
# RMSE Score of test data
for model in models:
    model = model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print(f"Test RMSE of {model} = {np.sqrt(mean_squared_error(prediction,y_test)):.2f}")

# Візуалізація залишків
# for model in models:
#     # Отримання залишків
#     residuals = model.predict(x_test) - y_test
#     # Гістограма залишків
#     plt.subplot(1, len(models), models.index(model) + 1)
#     plt.hist(residuals, bins=20)
#     plt.title(f'Model {type(model).__name__}')
#     plt.show()
# Зробіть висновок щодо відповідності залишків моделей до нормального закону 
# розподілу

# 13. Візуалізація прогнозів
import matplotlib.pyplot as plt
# CV Scores using train data
for model in models:
    CV = KFold(n_splits=4, shuffle=True, random_state=1)
    # Pipeline
    model_pipe = Pipeline([
    ('scaler', scaler),
    ('model', model)
    ])
    # Навчання моделі
    model_pipe.fit(x_train, y_train)
    # Прогнози моделі
    y_pred_train = model_pipe.predict(x_train)
    # Побудова графіка
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(x_train)), train.iloc[:, -1], color='black', 
    label='Actual')
    plt.plot(np.arange(len(x_train)), y_pred_train, label='Predicted', 
    color='red', linestyle='--')
    plt.title(f"Predicted values for {model.__class__.__name__}")
    plt.xlabel("Sample Number")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.show()
