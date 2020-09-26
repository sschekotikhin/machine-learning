from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
from pandas.plotting import scatter_matrix

import mglearn

import numpy as np

# import matplotlib.pyplot as plt


iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
  iris_dataset['data'],
  iris_dataset['target'],
  random_state=0
)

# iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# grr = scatter_matrix(
#   iris_dataframe,
#   c=y_train,
#   figsize=(15, 15),
#   marker='o',
#   hist_kwds={'bins': 20},
#   s=60,
#   alpha=.8,
#   cmap=mglearn.cm3
# )

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)

y_pred = knn.predict(X_test)
print(y_pred)

print(knn.score(X_test, y_test))
