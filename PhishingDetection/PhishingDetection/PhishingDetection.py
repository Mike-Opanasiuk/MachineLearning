import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Завантаження даних з файлу CSV
data = pd.read_csv('Sample.csv')

# Розділення на колонки категорійних та числових даних
categorical_cols = ['URL', 'Domain', 'TLD', 'Title']  # Категорійні колонки
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()  # Числові колонки

# Імп'ютер для числових даних, заміна NaN на медіану
imputer_num = SimpleImputer(strategy='median')
data[numerical_cols] = imputer_num.fit_transform(data[numerical_cols])

print("Числові дані після заміни медіаною:")
print(data[numerical_cols].head())

# Імп'ютер для категорійних даних, заміна NaN на найчастіше значення
imputer_cat = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = imputer_cat.fit_transform(data[categorical_cols])

print("Категорійні дані після заміни найчастішим значенням:")
print(data[categorical_cols].head())

# Вилучення мітки з числових колонок, якщо вона там присутня
if 'label' in numerical_cols:
    numerical_cols.remove('label')

# One-hot кодування категорійних даних
encoder = OneHotEncoder()
encoded_categorical_data = encoder.fit_transform(data[categorical_cols]).toarray()
categorical_feature_names = encoder.get_feature_names_out(categorical_cols)

# Об'єднання імен числових та категорійних ознак
all_feature_names = numerical_cols + list(categorical_feature_names)

# Нормалізація числових даних
scaler = StandardScaler()
scaled_numerical_data = scaler.fit_transform(data[numerical_cols])

# Об'єднання нормалізованих числових даних та закодованих категорійних даних
processed_data = np.concatenate([scaled_numerical_data, encoded_categorical_data], axis=1)

# Розділення даних на тренувальні та тестові набори
X_train, X_test, y_train, y_test = train_test_split(processed_data, data['label'], test_size=0.2, random_state=42)

# Налаштування моделі ласо регресії для вибору ознак
lasso = LassoCV(cv=5, random_state=42, max_iter=1000000)
lasso.fit(X_train, y_train)

# Отримання і візуалізація важливості ознак
feature_importance = np.abs(lasso.coef_)
important_features_indices = np.where(feature_importance > 0)[0]
important_features_names = [all_feature_names[i] for i in important_features_indices]

plt.figure(figsize=(10, 8))
plt.bar(important_features_names, feature_importance[important_features_indices])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance via Lasso Regression')
plt.xticks(rotation=90)
plt.show()

# Тренування основної моделі ласо регресії
model = Lasso(alpha=0.01, random_state=42)
model.fit(X_train, y_train)

# Оцінка моделі на тренувальному та тестовому наборах
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print("Training score:", train_score)
print("Testing score:", test_score)

# Розрахунок і виведення метрик для оцінки якості моделі
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)

# Налаштування параметрів моделі за допомогою GridSearchCV
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(Lasso(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(-grid_search.best_score_))

model = grid_search.best_estimator_
test_score = model.score(X_test, y_test)
print("Test set score: {:.2f}".format(test_score))

# Візуалізація порівняння фактичних і передбачених значень
plt.figure(figsize=(14, 7))
indices = range(len(y_test))
sorted_indices = np.argsort(y_test)
sorted_y_test = np.array(y_test)[sorted_indices]
sorted_y_pred_test = y_pred[sorted_indices]

plt.plot(indices, sorted_y_test, label='Actual', color='blue', marker='o')
plt.plot(indices, sorted_y_pred_test, label='Predicted', color='red', linestyle='--', marker='x')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Comparison of Actual and Predicted Values')
plt.legend()
plt.show()
