import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting test set results
ypred=regressor.predict(X_test)

#visualizing the results for training set
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title('Sal vs exp(training set')
plt.xlabel('yearsofexp')
plt.ylabel('salary')
plt.show()


#visualizing the results for test set
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title('Sal vs exp(test set')
plt.xlabel('yearsofexp')
plt.ylabel('salary')
plt.show()