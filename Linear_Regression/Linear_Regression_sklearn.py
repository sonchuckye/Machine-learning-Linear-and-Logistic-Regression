from sklearn import datasets, linear_model
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

def cost_function(predictions, train_output): 
        cost = np.mean((train_output - predictions) ** 2) 
        return cost
## Dataset 1
# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

## Dataset 2
#url = 'https://media.geeksforgeeks.org/wp-content/uploads/20240320114716/data_for_lr.csv'
#data = pd.read_csv(url)

# Drop the missing values
#data = data.dropna()

# training dataset and labels
#train_input = np.array(data.x[0:500]).reshape(500, 1)
#train_output = np.array(data.y[0:500]).reshape(500, 1)

# valid dataset and labels
#test_input = np.array(data.x[500:700]).reshape(199, 1)
#test_output = np.array(data.y[500:700]).reshape(199, 1)

#X = train_input
#y = train_output
#---------------------------------------------------------------------------------------------

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)
print("Linear Regression Coef:", regr.coef_)

# Predict
x0 = X
y0 = regr.coef_[0,0] + x0*regr.coef_[0,1]
print("MSE =", cost_function(y, y0))
# Drawing the fitting line 
plt.plot(X.T, y.T, 'ro')     # data 
plt.plot(x0, y0)               # the fitting line
plt.plot(152, 52, 'bs')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()