# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('monthlyexp vs incom.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'green')
plt.title('Monthly exp vs income (Linear Regression)')
plt.xlabel('Monthly exp')
plt.ylabel('Income')
plt.show()

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'green')
plt.title('Monthly exp vs income (Polynomial Regression)')
plt.xlabel('Monthly exp')
plt.ylabel('Income')
plt.show()

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 1)
regressor.fit(X, y)


# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'green')
plt.title('Monthly exp vs income  (Decision Tree Regression)')
plt.xlabel('Montlhy exp')
plt.ylabel('Income')
plt.show()

x =lin_reg.predict([[4]])
print('linear reg=',x)

# Predicting a new result with Polynomial Regression
b =lin_reg_2.predict(poly_reg.fit_transform([[3]]))
print('Polynomial reg=',b)

# Predicting a new result
c = regressor.predict([[3]])
print('Decision tree=',c)