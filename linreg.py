# Linear regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error, r2_score

# Import data and normalize it
normalize = lambda x : (x.values - np.mean(x.values) ) / np.std(x.values) 
data = pd.read_csv("data.txt", sep="\t")
ndata = pd.DataFrame()

mean_rent = np.mean(data["rent"])
std_rent = np.std(data["rent"])

for col in data.columns:
	if not "ward" in col:
		kwargs = { col: normalize(data[col])}
		ndata = ndata.assign(**kwargs)
# print(ndata.head())

# Fit linear regression
X = ndata[ndata.columns.difference(["ward", "rent"])]
X_train = X[:-200]
X_test = X[-200:]

y = ndata["rent"]
y_train = y[:-200]
y_test = y[-200:]
print(X_test.shape, y_test.shape)

linmodel = lm.LinearRegression(fit_intercept=True)
linmodel.fit(X_train, y_train)
y_pred = linmodel.predict(X_test)

print('Coefficients: \n', linmodel.coef_)
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))
print('Intercept score: %.2f' % linmodel.intercept_)

# Plot outputs
plt.xlabel("Real rent")
plt.ylabel("Predicted rent")
plt.scatter(data["rent"][-200:], (y_pred * std_rent) + mean_rent, color='black')

# plt.xticks(())
# plt.yticks(())

plt.show()