import mglearn

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print(f'Train-set size: {len(X_train)}')

# training
kns = KNeighborsClassifier(n_neighbors=1)
kns.fit(X_train, y_train)

predicted = kns.predict(X_test)
print(f'Expected: {y_test}, predicted: {predicted}')
print(f'Accuracy: {kns.score(X_test, y_test)}')
