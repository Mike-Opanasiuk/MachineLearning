import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt


print("################################################### Завантаження даних ###################################################")

data = pd.read_csv('Sample.csv')
# Заповнення пропущених даних (NaN)
from sklearn.impute import SimpleImputer

# Розділення на колонки категорійних та числових даних
categorical_cols = ['URL', 'Domain', 'TLD', 'Title']  # Категорійні колонки
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()  # Усі числові колонки

# Ініціалізація імп'ютера, що замінює NaN на медіану кожної колонки
imputer = SimpleImputer(strategy='median')
data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

# Ініціалізація імп'ютера, що замінює NaN на найчастіше значення кожної колонки
imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = imputer.fit_transform(data[categorical_cols])

if 'label' in numerical_cols:
    numerical_cols.remove('label')

encoder = OneHotEncoder()
encoded_categorical_data = encoder.fit_transform(data[categorical_cols]).toarray();
categorical_feature_names = encoder.get_feature_names_out(categorical_cols)
# Об'єднуємо імена числових ознак із генерованими категорійними ознаками
all_feature_names = numerical_cols + list(categorical_feature_names)

scaler = StandardScaler()
scaled_numerical_data = scaler.fit_transform(data[numerical_cols])

processed_data = np.concatenate([scaled_numerical_data, encoded_categorical_data], axis=1)
X_train, X_test, y_train, y_test = train_test_split(processed_data, data['label'], test_size=0.2, random_state=42)

print("################################################### Вибір ознак (Feature selection) ###################################################")

# Налаштування моделі ласо регресії
lasso = LassoCV(cv=5, random_state=42, max_iter=1000000)

# Тренування моделі на тренувальному наборі даних
lasso.fit(X_train, y_train)

# Отримання важливості ознак
feature_importance = np.abs(lasso.coef_)

# Визначення індексів для ознак, які модель вважає важливими
important_features_indices = np.where(feature_importance > 0)[0]
important_features_names = [all_feature_names[i] for i in important_features_indices]

# Візуалізація важливості ознак
plt.figure(figsize=(10, 8))
plt.bar(important_features_names, feature_importance[important_features_indices])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance via Lasso Regression')
plt.xticks(rotation=90)
plt.show()

important_features_indices, important_features_names


print("################################################### Навчання моделі ###################################################")

from sklearn.linear_model import Lasso

# Встановлення моделі з вибраною альфа, яка може бути визначена через крос-валідацію або досвід
model = Lasso(alpha=0.01, random_state=42) # was 0.1

# Тренування моделі на тренувальному наборі
model.fit(X_train, y_train)

# Оцінка моделі на тренувальному наборі
train_score = model.score(X_train, y_train)

# Оцінка моделі на тестувальному наборі
test_score = model.score(X_test, y_test)

print("Training score:", train_score)
print("Testing score:", test_score)


print("################################################### Оцінка моделі ###################################################")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Передбачення значень на тестових даних
y_pred = model.predict(X_test)
y_pred_test = model.predict(X_test)

# Розрахунок метрик
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)

print("################################################### Налаштування моделі ###################################################")

from sklearn.model_selection import GridSearchCV

# Визначення параметрів для тестування
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# Налаштування GridSearchCV
grid_search = GridSearchCV(Lasso(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)

# Тренування GridSearchCV
grid_search.fit(X_train, y_train)

# Вивід найкращих параметрів і найкращого результату
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(-grid_search.best_score_))

# Використання найкращої моделі для оцінки на тестових даних
model = grid_search.best_estimator_
test_score = model.score(X_test, y_test)
print("Test set score: {:.2f}".format(test_score))

print("################################################### Тестування моделі ###################################################")

plt.figure(figsize=(14, 7))

# Впорядкування тестових значень та передбачень за індексом
indices = range(len(y_test))
sorted_indices = np.argsort(y_test)
sorted_y_test = np.array(y_test)[sorted_indices]
sorted_y_pred_test = y_pred_test[sorted_indices]

# Фактичні дані
plt.plot(indices, sorted_y_test, label='Actual', color='blue', marker='o')

# Передбачені дані
plt.plot(indices, sorted_y_pred_test, label='Predicted', color='red', linestyle='--', marker='x')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Comparison of Actual and Predicted Values')
plt.legend()
plt.show()