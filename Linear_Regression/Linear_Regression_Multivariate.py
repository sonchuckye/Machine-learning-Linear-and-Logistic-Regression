from sklearn import datasets, linear_model
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def cost_function(predictions, train_output): 
        cost = np.mean((train_output - predictions) ** 2) 
        return cost
## Dataset 1
datapath = 'C:\\Users\\Admin\\Desktop\\Trí tuệ nhân tạo\\Bài tập lớn\\Linear_Regression\\multiple_linear_regression_dataset.csv'
data = pd.read_csv(datapath)
x1 = np.array(data.age)
x2 = np.array(data.experience)
X = np.array([x1,x2]).T
#print(data)
#print(x)
y = np.array(data.income).T

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=True) # fit_intercept = True for calculating the bias
regr.fit(X, y)
print("Linear Regression Coef:", regr.coef_)
print("Linear Regression intercept:", regr.intercept_)
print("Accuracy:", regr.score(X, y))

# Predict
x0 = X
y0 = regr.intercept_ + x0@regr.coef_
print("MSE =", cost_function(y, y0))

# Drawing the fitting line
X1, X2 = np.meshgrid(x1, x2)
X_surfaces = np.array([X1.ravel(), X2.ravel()]).T
Y = regr.predict(X_surfaces).reshape(X1.shape)

fig = plt.figure(figsize=(9, 7))
ax = plt.axes(projection='3d')

ax.scatter(x1, x2, y, color='red', marker='o')
ax.plot_surface(X1, X2, Y, cmap='viridis', edgecolor='black', alpha=0.6)  # Adjust colormap as desired

ax.set_xlabel('Age')
ax.set_ylabel('Experience')
ax.set_zlabel('Income')
ax.view_init(20, 10)

ax.set_title('Best fit plane')

plt.show()