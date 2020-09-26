import mglearn

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

import numpy as np

import matplotlib.pyplot as plt


X, y = mglearn.datasets.make_wave(n_samples=150)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# accuracy = []

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)

for n_neighbors, ax in zip([1, 3, 9], axes):
  knr = KNeighborsRegressor(n_neighbors=n_neighbors)
  knr.fit(X_train, y_train)

  ax.plot(line, knr.predict(line))
  ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
  ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
  ax.set_title(
    "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
      n_neighbors, knr.score(X_train, y_train),
      knr.score(X_test, y_test)
    )
  )
  ax.set_xlabel("Признак")
  ax.set_ylabel("Целевая переменная")
  axes[0].legend(["Прогнозы модели", "Обучающие данные/ответы","Тестовые данные/ответы"], loc="best")

plt.show()

# for i in range(1, 20):
#   knr = KNeighborsRegressor(n_neighbors=i)
#   knr.fit(X_train, y_train)

#   predicted = knr.predict(X_test)
#   accuracy.append(knr.score(X_test, y_test))

# for index, acc in enumerate(accuracy):
#   print(f'neighbors: {index}, accuracy: {acc}')
