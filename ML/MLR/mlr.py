import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"/Users/ali/github/AI-ML/ML/MLR/Investment.csv")

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]


X = pd.get_dummies(X, dtype=int)


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

y_pred = regression.predict(X_test)

m = regression.coef_
print(m)

c = regression.intercept_
print(c)


X = np.append(arr=np.full((50,1), 42467).astype(int), values=X, axis=1)


import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,4,5]]

#OrdinaryLeatSquares

regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()


import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,5]]

#OrdinaryLeatSquares

regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()


import statsmodels.api as sm
X_opt = X[:,[0,1,2,3]]

#OrdinaryLeatSquares

regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()

import statsmodels.api as sm
X_opt = X[:,[0,1,3]]

#OrdinaryLeatSquares

regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()

import statsmodels.api as sm
X_opt = X[:,[0,1]]

#OrdinaryLeatSquares

regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()

bias = regression.score(X_train, y_train)
print(bias)

variance = regression.score(X_test, y_test)
variance
    
    
    
    
    
    
    
    
    
    
    
    
    
    

