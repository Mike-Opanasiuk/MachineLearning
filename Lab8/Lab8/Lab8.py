import numpy as np
from numpy import interp
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

# Завантаження даних
file = "region5b.csv"
XY = pd.read_csv(file)
X = XY.iloc[:, :12].drop(columns=['t1'])
y = XY.iloc[:, 12]

# Налаштування крос-валідації
k_folds = 4
kf = KFold(n_splits=k_folds)

# Параметри С для перевірки
C_values = [None, 1, 10, 100, 1000]
results = []

# Виконання крос-валідації для кожного значення С
for C in C_values:
    avg_auc = []
    avg_sensitivity = []
    avg_specificity = []
    avg_accuracy = []
    roc_curves = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Створення і навчання моделі SVM
        model = SVC(kernel='linear', C=C if C is not None else 1, probability=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba_test = model.decision_function(X_test)

        # Оцінка моделі
        test_auc = roc_auc_score(y_test, y_proba_test)
        avg_auc.append(test_auc)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        avg_sensitivity.append(tp / (tp + fn))
        avg_specificity.append(tn / (tn + fp))
        avg_accuracy.append(accuracy_score(y_test, y_pred))
        fpr, tpr, _ = roc_curve(y_test, y_proba_test)
        roc_curves.append((fpr, tpr))

    # Усереднення і збереження результатів
    mean_tpr = np.mean([interp(np.linspace(0, 1, 100), fpr, tpr) for fpr, tpr in roc_curves], axis=0)
    results.append({
        'C': 'default' if C is None else C,
        'AUC': np.mean(avg_auc),
        'Sensitivity': np.mean(avg_sensitivity),
        'Specificity': np.mean(avg_specificity),
        'Accuracy': np.mean(avg_accuracy),
        'Mean FPR': np.linspace(0, 1, 100),
        'Mean TPR': mean_tpr
    })

# Виведення результатів і побудова ROC-кривих
plt.figure(figsize=(10, 8))
for result in results:
    plt.plot(result['Mean FPR'], result['Mean TPR'], label=f'C={result["C"]} (AUC = {result["AUC"]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for different C values')
plt.legend(loc="lower right")
plt.show()

for result in results:
    print(f'C={result["C"]}: AUC={result["AUC"]:.2f}, Sensitivity={result["Sensitivity"]:.2f}, Specificity={result["Specificity"]:.2f}, Accuracy={result["Accuracy"]:.2f}')
