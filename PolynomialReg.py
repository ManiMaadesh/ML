# -*- coding: utf-8 -*-
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting linregmodel
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X,y)

#fitting ploynomialregmodel

from sklearn.preprocessing import PolynomialFeatures
polyreg=PolynomialFeatures(degree=4)
Xpoly=polyreg.fit_transform(X)
linreg2=LinearRegression()
linreg2.fit(Xpoly,y)

#visuaizing linreg results
plt.scatter(X,y,color='red')
plt.plot(X,linreg.predict(X),color='blue')
plt.title('truth or bluff(linreg)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

#visulizing polyreg results
xgrid=np.arange(min(X),max(X),0.1)
xgrid=xgrid.reshape((len(xgrid),1))#higher res graphs
plt.scatter(X,y,color='red')
plt.plot(xgrid,linreg2.predict(polyreg.fit_transform(xgrid)),color='blue')
plt.title('truth or bluff(polyreg)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

#predicting new result with linreg
linreg.predict(6.5)
#predicting new result with polyreg
linreg2.predict(polyreg.fit_transform(6.5))











