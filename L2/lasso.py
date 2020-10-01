from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split

import mglearn

import matplotlib.pyplot as plt

import numpy as np


X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

ridge_01 = Ridge(alpha=0.1)
ridge_01.fit(X_train, y_train)

print(f'Train cccuracy: {ridge_01.score(X_train, y_train)}')
print(f'Test cccuracy: {ridge_01.score(X_test, y_test)}')

lasso = Lasso()
lasso.fit(X_train, y_train)

print(f'Train cccuracy: {lasso.score(X_train, y_train)}')
print(f'Test cccuracy: {lasso.score(X_test, y_test)}')
print(f'Correct features: {np.sum(lasso.coef_ != 0)}')

lasso_001 = Lasso(alpha=0.01, max_iter=100000)
lasso_001.fit(X_train, y_train)

print(f'Train cccuracy: {lasso_001.score(X_train, y_train)}')
print(f'Test cccuracy: {lasso_001.score(X_test, y_test)}')
print(f'Correct features: {np.sum(lasso_001.coef_ != 0)}')

lasso_00001 = Lasso(alpha=0.0001, max_iter=100000)
lasso_00001.fit(X_train, y_train)

print(f'Train cccuracy: {lasso_00001.score(X_train, y_train)}')
print(f'Test cccuracy: {lasso_00001.score(X_test, y_test)}')
print(f'Correct features: {np.sum(lasso_00001.coef_ != 0)}')

plt.plot(lasso.coef_, 's', label='Лассо alpha=1')
plt.plot(lasso_001.coef_, '^', label='Лассо alpha=0.01')
plt.plot(lasso_00001.coef_, 'v', label='Лассо alpha=0.0001')
plt.plot(ridge_01.coef_, 'o', label='Гребневая регрессия alpha=0.1')

plt.xlabel('Индекс коэффициента')
plt.ylabel('Оценка коэффициента')

plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)

plt.show()
