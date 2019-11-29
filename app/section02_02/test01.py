import sklearn as sk
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

print(sk.__version__)

iris_dataset = load_iris()
print("iris_dataset key: {}".format(iris_dataset.keys()))
print(iris_dataset['data'])
print(iris_dataset['data'].shape)
print(iris_dataset['feature_names'])
print(iris_dataset['target'])
print(iris_dataset['target_names'])
print(iris_dataset['DESCR'])

train_input, test_input, train_label, test_label \
    = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.25, random_state=42)

print("shape of train input: {}".format(train_input.shape))
print("shape of test input: {}".format(test_input.shape))
print("shape of train label: {}".format(train_label.shape))
print("shape of test label: {}".format(test_label.shape))


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_input, train_label)

import numpy as np

predict_label = knn.predict(test_input)
print(predict_label)

print(np.mean(predict_label == test_label))


