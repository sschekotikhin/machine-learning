from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import mglearn


X, y = mglearn.datasets.make_wave(n_samples=60)

print(type(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

print(f'lr.coef: {lr.coef_}\nlr.intercept_: {lr.intercept_}')
print(f'Train score: {lr.score(X_train, y_train)}\nTest score: {lr.score(X_test, y_test)}')
