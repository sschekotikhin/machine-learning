from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt


cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
  cancer.data,
  cancer.target,
  stratify=cancer.target,
  random_state=66
)

accuracy = { 'train': [], 'test': [] }
settings = range(1, 15)

for n_neighbors in settings:
  kns = KNeighborsClassifier(n_neighbors=n_neighbors)
  kns.fit(X_train, y_train)

  accuracy['train'].append(kns.score(X_train, y_train))
  accuracy['test'].append(kns.score(X_test, y_test))

plt.plot(settings, accuracy['train'], label='training accuracy')
plt.plot(settings, accuracy['test'], label='test accuracy')

plt.ylabel('accuracy')
plt.xlabel('neighbors')
plt.legend()

plt.show()
