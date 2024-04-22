# 1. Імпорт бібліотек та модулів
import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split

# 2. Завантаження даних з диска
XY = pd.read_csv("region5a.csv")

# 3. Розділення на фактори та змінну відгук
from sklearn.model_selection import train_test_split
X = XY.iloc[:, :12]
y = XY.iloc[:, 12]
# Додати константу до матриці X
X = sm.add_constant(X)

# 4. Видалення одного фактора згідно з варіантом
X0 = X.drop(columns=['t1'])
print(X0.head()) # перегляд факторів

# 5. Ініціалізація моделі лінійної регресії
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# 6. Утворення складок K-Fold
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
# Визначення кількості складок
k_folds = 5
# Ініціалізація об'єкту KFold для розділення даних на складки
kf = KFold(n_splits=k_folds)
# kf = KFold(n_splits=k_folds, shuffle=True)
# Створення списків для збереження результатів
mse_scores = []
r2_scores = []

# 7. Виконання K-fold Cross-Validation
for i, (train_index, test_index) in enumerate(kf.split(X), 1):
# for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # Навчання моделі на навчальних даних
    model.fit(X_train, y_train)
    # Передбачення на тестових даних
    y_pred = model.predict(X_test)
    # Обчислення середньоквадратичної помилки
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
    # Обчислення R-квадрат
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)
    print(f"Складка {i}: Кількість елементів контрольної вибірки: {len(test_index)}")
    print("СКв помилка (MSE):", mse.round(2))
    print("R-квадрат:", r2.round(2))
    print()
    
# 8. Усереднення результатів оцінювання моделей
avg_mse = np.mean(mse_scores)

avg_r2 = np.mean(r2_scores)
print("Середня середньоквадратична помилка:", avg_mse.round(2))
print("Середнє R-квадрат:", avg_r2.round(2))