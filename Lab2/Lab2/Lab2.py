import matplotlib.pyplot as plt;
import pandas as pd;
import seaborn as sns

# Task 1

data = pd.read_csv('iris.csv')
data['Species'].value_counts().plot(kind='bar')
plt.show()

# Task 2

data = pd.read_csv('titanic_train.csv')

data_male = data[data['Sex'] == 'male']

sns.histplot(data_male['Age'], kde=True, bins=30)

plt.title('Detailed age distribution function for men')

plt.show()

# Task 3

data = pd.read_csv('iris.csv')

data_setosa = data[data['Species'] == 'Iris-setosa']

sns.histplot(data_setosa['SepalWidth'], kde=True, bins=30)

plt.title('Detailed SepalWidth distribution function for the setosa species')

plt.show()

# Task 4

data = pd.read_csv('iris.csv')

data_versicolor = data[data['Species'] == 'Iris-versicolor']

sns.scatterplot(x='PetalLength', y='PetalWidth', data=data_versicolor)

plt.title('Two-dimensional scatter plot for PetalLength and PetalWidth (Species=versicolor)')
plt.xlabel('PetalLength')
plt.ylabel('PetalWidth')

plt.show()