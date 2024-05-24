import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
np.random.seed(2)

X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

clf = linear_model.LogisticRegression().fit(X, y)
print(clf.predict(np.array([np.linspace(0,6,20)]).T))
print("Probablity = 0, Probablity = 1")
print(clf.predict_proba(np.array([np.linspace(0,6,20)]).T))
print("Độ chính xác:", clf.score(X, y))

X0 = X[y == 0]
y0 = y[y == 0]
X1 = X[y == 1]
y1 = y[y == 1]
plt.plot(X0, y0, 'ro', markersize = 8)
plt.plot(X1, y1, 'bs', markersize = 8)

xx = np.array([np.linspace(0, 6, 1000)]).T
yy = clf.predict_proba(xx)
plt.axis([-2, 8, -1, 2])
plt.plot(xx, yy[:,1], 'g-', linewidth = 2)
#plt.plot(threshold, .5, 'y^', markersize = 8)
plt.xlabel('Studying hours')
plt.ylabel('Predicted probability of pass')
plt.show()