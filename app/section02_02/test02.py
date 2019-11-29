from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np

iris_dataset = load_iris()

train_input, test_input, train_label, test_label \
    = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.25, random_state=42)

k_means = KMeans(n_clusters=3)
k_means.fit(train_input)

print(k_means.labels_)

print("0 cluster:", train_label[k_means.labels_ == 0])
print("1 cluster:", train_label[k_means.labels_ == 1])
print("2 cluster:", train_label[k_means.labels_ == 2])

predict_cluster = k_means.predict(test_input)

print(predict_cluster)

np_arr = np.array(predict_cluster)
np_arr[np_arr == 0], np_arr[np_arr == 1], np_arr[np_arr == 2] = 3, 4, 5

np_arr[np_arr == 3] = 0
np_arr[np_arr == 4] = 1
np_arr[np_arr == 5] = 2
predict_label = np_arr.tolist()
print(predict_label)

print(np.mean(predict_label == test_label))
