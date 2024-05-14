# Імпорт бібліотек та модулів
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import KFold

file = "region5b.csv"
XY = pd.read_csv(file)

# Розділення на фактори та змінну відгук
X = XY.iloc[:, :12]
y = XY.iloc[:, 12]

# Видалення одного фактора згідно з варіантом
X = X.drop(columns=['t1'])
print(X.head()) # перегляд факторів

# Ініціалізація моделі дерева рішень з глибиною дерева 2
model = DecisionTreeClassifier(max_depth=4, random_state=0)

# Підготовка до крос-валідації моделі
k_folds = 4
kf = KFold(n_splits=k_folds)

# Ініціалізація змінних для усереднення результатів
avg_auc = np.array([])
avg_sensitivity = np.array([])
avg_specificity = np.array([])
avg_accuracy = np.array([])

# Перехресна крос-валідація моделі за методом K-Fold
for i, (train_index, test_index) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    
    # Створення фігури для дерева
    plt.figure(i)
    tree.plot_tree(model, filled=True, fontsize=10)
    plt.title(f"Decision Tree for Fold {i}")
    plt.savefig(f"tree_fold_{i}.png")  # Збереження дерева в файл
    plt.close()
    
    y_pred = model.predict(X_test)
    test_auc = roc_auc_score(y_test, y_pred)
    
    # Розрахунок Sensitivity, Specificity та Accuracy
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Збереження значень характеристик для поточної складки
    avg_auc = np.append(avg_auc, test_auc)
    avg_sensitivity = np.append(avg_sensitivity, sensitivity)
    avg_specificity = np.append(avg_specificity, specificity)
    avg_accuracy = np.append(avg_accuracy, accuracy)

# Усереднення результатів K-fold Cross-Validation
avg_auc = np.mean(avg_auc)
avg_sensitivity = np.mean(avg_sensitivity)
avg_specificity = np.mean(avg_specificity)
avg_accuracy = np.mean(avg_accuracy)

# Виведення усереднених результатів
print("Average AUC:", avg_auc.round(2))
print("Average Sensitivity:", avg_sensitivity.round(2))
print("Average Specificity:", avg_specificity.round(2))
print("Average Accuracy:", avg_accuracy.round(2))

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Ініціалізація списків для збереження міток та прогнозів
all_y_test = np.array([])
all_y_pred_proba = np.array([])

# Код крос-валідації з попереднього прикладу
for i, (train_index, test_index) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    # Збереження істинних міток та ймовірностей для класу 1
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    all_y_test = np.append(all_y_test, y_test)
    all_y_pred_proba = np.append(all_y_pred_proba, y_pred_proba)

# Побудова ROC-кривої
fpr, tpr, thresholds = roc_curve(all_y_test, all_y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Налаштування глибини дерева
from sklearn.model_selection import cross_val_score

# Змінна для зберігання результатів точності
depth_accuracy = []

# Пробуємо різні значення глибини дерева
for depth in range(1, 7):
    model = DecisionTreeClassifier(max_depth=depth, random_state=0)
    # Виконуємо крос-валідацію та обчислюємо середню точність
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    depth_accuracy.append(scores.mean())

# Відображення графіка залежності точності від глибини дерева
plt.figure()
plt.plot(range(1, 7), depth_accuracy, marker='o', linestyle='-', color='b')
plt.xlabel('Depth of Tree')
plt.ylabel('Average Accuracy')
plt.title('Impact of Tree Depth on Model Accuracy')
plt.grid(True)
plt.show()

# Визначення оптимальної глибини
optimal_depth = range(1, 7)[depth_accuracy.index(max(depth_accuracy))]
print(f"The optimal depth of the tree is: {optimal_depth}")

# Розрахунок важливості ознак
# Ініціалізація і навчання моделі дерева рішень з оптимальною глибиною
model = DecisionTreeClassifier(max_depth=1, random_state=0)
model.fit(X, y)

# Отримання важливості ознак
feature_importances = model.feature_importances_

# Створення DataFrame для зручного відображення
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Сортування ознак за важливістю в порядку спадання
features_df = features_df.sort_values(by='Importance', ascending=False)

# Виведення таблиці важливостей ознак
print(features_df)
