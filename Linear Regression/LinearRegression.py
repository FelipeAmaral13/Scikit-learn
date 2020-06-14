import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split



dataset = pd.read_csv("fatherEson.csv")
dataset.head()

X = dataset['Father'].values.reshape(-1,1)
y = dataset['Son'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)




plt.scatter(x, y, color = 'blue')
plt.plot(x, lin_reg.predict(X), color = 'red', linewidth=5)
plt.title('Father Height vs Son Height')
plt.xlabel('Father Height')
plt.ylabel('Son Height')
plt.show()

lin_reg_ridge = Ridge(alpha=0.2)
lin_reg_ridge.fit(X_train, y_train)

plt.scatter(x, y, color = 'blue')
plt.plot(x, lin_reg_ridge.predict(X), color = 'red', linewidth=5)
plt.title('Father Height vs Son Height')
plt.xlabel('Father Height')
plt.ylabel('Son Height')
plt.show()
