from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split

import mglearn

import matplotlib.pyplot as plt


X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lr = LinearRegression()
lr.fit(X_train, y_train)

print(f'lr.coef: {lr.coef_}\nlr.intercept_: {lr.intercept_}')
print(f'Train score: {lr.score(X_train, y_train)}\nTest score: {lr.score(X_test, y_test)}')

ridge = Ridge()
ridge_01 = Ridge(alpha=0.1)
ridge_10 = Ridge(alpha=10)

ridge.fit(X_train, y_train)
ridge_01.fit(X_train, y_train)
ridge_10.fit(X_test, y_test)

print(f'lr.coef: {ridge.coef_}\nlr.intercept_: {ridge.intercept_}')
print(f'Train score: {ridge.score(X_train, y_train)}\nTest score: {ridge.score(X_test, y_test)}')

plt.plot(ridge.coef_, 's', label='Гребневая регрессия alpha=1')
plt.plot(ridge_10.coef_, '^', label='Гребневая регрессия alpha=10')
plt.plot(ridge_01.coef_, 'v', label='Гребневая регрессия alpha=0.1')
plt.plot(lr.coef_, 'o', label='Линейная регрессия')

plt.xlabel('Индекс коэффициента')
plt.ylabel('Оценка коэффициента')

plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()

plt.show()

mglearn.plots.plot_ridge_n_samples()
plt.show()

