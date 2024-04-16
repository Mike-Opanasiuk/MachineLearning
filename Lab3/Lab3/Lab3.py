# 1. Import libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm

lineWidth = 90;
np.random.seed(80);

def printLine():
    print('*' * lineWidth)

# 2. Data download

XY = pd.read_csv("region4a.csv", delimiter=";")
XY.info()
printLine();

# 3. Statistical data analysis

pd.options.display.float_format = '{:,.2f}'.format
print(XY.describe())
printLine();

# 4. Separation of variables into influencing factors and feedback variable

X = XY.drop(columns=['eps'])
X = sm.add_constant(X)
y = XY['eps']

# 5. One factor removal depending on variant number (t1)

X0 = X.drop(columns=['t1'])
print(X.head())
printLine();

# 6. Building a linear regression model with all factors

model = sm.OLS(y, X0).fit()
print(model.summary())
printLine();

# 7. Removal of insignificant factors (t3, t5, t7 remains)

X1 = X0.drop(columns=['t2', 't4', 't6', 't8', 't9',  'R10', 'R20', 'R30']);
model1 = sm.OLS(y, X1).fit()
print(model1.summary())

# 8. Removal of insignificant factors (t5, t7 remains)

X2 = X1.drop(columns=['t3'])
model2 = sm.OLS(y, X2).fit()
print(model2.summary())

# 9. Build of new data frame

XY2 = X2
XY2['eps'] = XY['eps']
print(XY2.head())
printLine();

# 10. Splitting the data into a training sample and a control sample

samplesize = int(0.75 * XY2.shape[0])

index = np.random.choice(XY2.index, size=samplesize, replace=False)
train = XY2.loc[index]
test = XY2.drop(index)

x_train = train.iloc[:, :-1] 
y_train = train.iloc[:, -1] 
x_test = test.iloc[:, :-1] 
y_test = test.iloc[:, -1] 

# 11. Construction of the model and evaluation of its accuracy

from sklearn.metrics import mean_squared_error
model = LinearRegression()
model.fit(x_train, y_train)

train_pred = model.predict(x_train)
test_pred = model.predict(x_test)

train_rmse = mean_squared_error(y_train, train_pred, squared=False)
test_rmse = mean_squared_error(y_test, test_pred, squared=False)

absolute_percentage_errors1 = np.abs((y_train - train_pred) / y_train)
absolute_percentage_errors2 = np.abs((y_test - test_pred) / y_test)

train_mape = np.mean(absolute_percentage_errors1)
test_mape = np.mean(absolute_percentage_errors2)

print("Study RMSE:{:.2f}".format(train_rmse))
print("Test RMSE:{:.2f}".format(test_rmse))
print("Study MAPE:{:.2f}".format(train_mape))
print("Test MAPE:{:.2f}".format(test_mape))
printLine();

# 12. Correspondence plot for the study sample

plt.figure(figsize=(10, 6))
plt.scatter(y_train, train_pred, color='blue', label='Prediction (study sample)')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 
color='red', linestyle='--')
plt.xlabel('Real data')
plt.ylabel('Predicted data')
plt.title('Prediction for study sample')
plt.legend()
plt.show()

# 13. Fact-prediction dot chart

plt.plot(np.arange(len(x_train)), train.iloc[:, -1], label='Actual')
plt.plot(np.arange(len(x_train)), train_pred, label='Predicted', linestyle='--')
plt.legend()
plt.show()

# 14. Correspondence plot for the test sample

plt.figure(figsize=(10, 6))
plt.scatter(y_test, test_pred, color='green', label='Prediction (test sample)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
color='red', linestyle='--')
plt.xlabel('Real data')
plt.ylabel('Predicted data')
plt.title('Prediction for test sample')
plt.legend()
plt.show()

# 15. Fact-prediction dot chart

plt.plot(np.arange(len(x_test)), test.iloc[:, -1], label='Actual')
plt.plot(np.arange(len(x_test)), test_pred, label='Predicted', linestyle='--')
plt.legend()
plt.show()
