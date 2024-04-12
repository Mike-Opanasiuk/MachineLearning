import pandas as pd

# Task 1

data = pd.read_csv('iris.csv')

missing_values = data.isnull().sum()
print("Missing values in the dataset are: ")

if(missing_values.sum() == 0):
    print("There are no missing values in the dataset\n")
else:
    print(missing_values)
    
pd.options.display.float_format = '{:,.2f}'.format
print(data.describe())

# Task 2

print("\n\nTop 5 rows before standardization: ")
print(data.head(5)) 
from sklearn.preprocessing import StandardScaler
# Initialize the scaler
scaler = StandardScaler()
# Standardize numerical columns
data[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']] = scaler.fit_transform(data[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']])
print("\nTop 5 rows after standardization: ")
print(data.head(5))