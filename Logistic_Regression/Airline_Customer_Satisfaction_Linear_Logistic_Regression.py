# Standard operational package imports.
import numpy as np
import pandas as pd

# Important imports for preprocessing, modeling, and evaluation.
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics

# Visualization package imports.
import matplotlib.pyplot as plt
import seaborn as sns
df_original = pd.read_csv("C:\\Users\\Admin\\Desktop\\Trí tuệ nhân tạo\\Bài tập lớn\\Logistic_Regression\\Invistico_Airline.csv")
df_original.head(n = 10)
#print("Số lượng dữ liệu bị thiếu:\n", df_original.isnull().sum())
df_subset = df_original.dropna(axis=0).reset_index(drop = True) # drop dữ liệu NULL

df_subset = df_subset.astype({"Inflight entertainment": float})
df_subset['satisfaction'] = OneHotEncoder(drop='first').fit_transform(df_subset[['satisfaction']]).toarray()
df_subset.head(10)
#print(df_subset)

#X = df_subset[["Inflight entertainment", "Flight Distance"]]
X = df_subset[["Inflight entertainment"]]
y = df_subset["satisfaction"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
#print(X_train.shape)

clf = LogisticRegression().fit(X_train,y_train)
print("Logistic Regression Coef:", clf.coef_)
print("Logistic Regression Intercept:", clf.intercept_)
print("Độ chính xác của mô hình hồi quy Logistic (tính bằng%):", clf.score(X_test, y_test))
sns.regplot(x="Inflight entertainment", y="satisfaction", data=df_subset, logistic=True, ci=None)
plt.show()