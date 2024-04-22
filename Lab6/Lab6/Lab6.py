# 1. Імпорт бібліотек та модулів
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Завантаження даних з диска
XY = pd.read_csv("region5b.csv")

# 2. Розділення на фактори та змінну відгук
from sklearn.model_selection import train_test_split
X = XY.iloc[:, :12]
y = XY.iloc[:, 12]
# print(X)
# print(y)
# Додати константу до матриці X
X = sm.add_constant(X)

# 3. Видалення одного фактора згідно з варіантом
X0 = X.drop(columns=['t1'])
print(X0.head()) # перегляд факторів

# 4. Ініціалізація моделі логістичної регресії
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)

# 5. Метод перехресної перевірки (4 складки)
from sklearn.model_selection import KFold
# Визначення кількості складок
k_folds = 4
# Ініціалізація об'єкту KFold для розділення даних на складки
kf = KFold(n_splits=k_folds)
# kf = KFold(n_splits=k_folds, shuffle=True)
# Ініціалізація змінних для усереднення оцінок класифікації
avg_auc = np.array([])
avg_sensitivity = np.array([])
avg_specificity = np.array([])
avg_accuracy = np.array([])


# 6. Реалізація моделі та оцінка її точності

# Виконання K-fold Cross-Validation
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score

for i, (train_index, test_index) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # Навчання моделі на навчальних даних
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    train_auc = roc_auc_score(y_train, y_train_pred)
    # Передбачення на тестових даних
    y_pred = model.predict(X_test)
    # Оцінка моделі на тестовому наборі
    test_auc = roc_auc_score(y_test, y_pred)
    print("Test AUC:", test_auc.round(2))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    # Розрахунок Sensitivity, Specificity та Accuracy
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(y_test, y_pred)
    print("AUC:", test_auc.round(2))
    print("Sensitivity:", sensitivity.round(2))
    print("Specificity:", specificity.round(2))
    print("Accuracy:", round(accuracy, 2))
    # Збереження значень характеристик для поточної складки
    avg_auc = np.append(avg_auc, test_auc)
    avg_sensitivity = np.append(avg_sensitivity, sensitivity)
    avg_specificity = np.append(avg_specificity, specificity)
    avg_accuracy = np.append(avg_accuracy, accuracy)
    
# 7. Усереднені оцінки якості бінарного класифікатора

# Усереднення результатів K-fold Cross-Validation
avg_auc = np.mean(avg_auc)
avg_sensitivity = np.mean(avg_sensitivity)
avg_specificity = np.mean(avg_specificity)
avg_accuracy = np.mean(avg_accuracy)
print("Average AUC:", avg_auc.round(2))
print("Average Sensitivity:", avg_sensitivity.round(2))
print("Average Specificity:", avg_specificity.round(2))
print("Average Accuracy:", avg_accuracy.round(2))

# 8. Побудова ROC-кривої
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# Для навчальних даних
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_pred)
roc_auc_train = auc(fpr_train, tpr_train)
# Для тестових даних
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred)
roc_auc_test = auc(fpr_test, tpr_test)

plt.figure()
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label='Test ROC curve (area = %0.2f)' % roc_auc_test)
plt.plot(fpr_train, tpr_train, color='green', lw=2, label='Train ROC curve (area = %0.2f)' % roc_auc_train)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()