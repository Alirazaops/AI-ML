import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Data
dataset = pd.read_csv(r"/Users/ali/Downloads/Salary_Data.csv")

x = dataset.iloc[:, :-1]
y = dataset.iloc[:,-1]

dataset.isnull().sum()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=0)


from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train, y_train)

y_pred = regression.predict(x_test)


plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regression.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

m_coef = regression.coef_
print(m_coef)

c_intercept = regression.intercept_
print(c_intercept)

y_12 = m_coef * 12 + c_intercept
print(y_12)