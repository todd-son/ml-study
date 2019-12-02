import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import seaborn as sns

data_path = "/Users/todd/word2vec-nlp-tutorial/"
train_data = pd.read_csv(data_path + "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

print(train_data.head())

# 파일크기 확인
print()
print("파일 크기: ")

for file in os.listdir(data_path):
    if 'tsv' in file:
        print(file.ljust(30), round(os.path.getsize(data_path + file) / (1024 * 1024), 2), "MB")

print()
print("전체 학습 데이터 개수: ", len(train_data))

print()
train_length = train_data['review'].apply(len)
train_length.head()

plt.figure(figsize=(12, 5))
plt.hist(train_length, bins=200, alpha=0.5, color='r', label='word')
plt.yscale('log', nonposy='clip')
plt.xlabel("Length of review")
plt.ylabel("Number of review")
plt.show()

plt.boxplot(train_length, labels=['count'], showmeans=True)
plt.show()

print()
print('리뷰 길이 최대값: {}'.format(np.max(train_length)))
print('리뷰 길이 최소값: {}'.format(np.min(train_length)))
print('리뷰 길이 평균값: {}'.format(np.average(train_length)))
print('리뷰 길이 표준편차: {}'.format(np.std(train_length)))
print('리뷰 길이 중간값: {}'.format(np.median(train_length)))
print('리뷰 길이 제1사분위: {}'.format(np.percentile(train_length, 25)))
print('리뷰 길이 제3사분위: {}'.format(np.percentile(train_length, 75)))

cloud = WordCloud(width=800, height=600).generate(" ".join(train_data['review']))
plt.imshow(cloud)
plt.show()

fig, axe = plt.subplots(ncols=1)
fig.set_size_inches(6, 3)
sns.countplot(train_data['sentiment'])
plt.show()

print("긍정 리뷰 개수: {}".format(train_data['sentiment'].value_counts()[1]))
print("부정 리뷰 개수: {}".format(train_data['sentiment'].value_counts()[0]))
