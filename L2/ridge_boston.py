from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

import mglearn


X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y)

ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)

print(f'lr.coef: {ridge.coef_}\nlr.intercept_: {ridge.intercept_}')
print(f'Train score: {ridge.score(X_train, y_train)}\nTest score: {ridge.score(X_test, y_test)}')
