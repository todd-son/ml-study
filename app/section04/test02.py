import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_path = "/Users/todd/word2vec-nlp-tutorial/"

train_data = pd.read_csv(data_path + "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
train_word_counts = train_data['review'].apply(lambda x: len(x.split(' ')))
print(train_word_counts.head())

plt.figure(figsize=(12, 5))
plt.hist(train_word_counts, bins=200, alpha=0.5, color='r', label='word')
# plt.yscale('log', nonposy='clip')
plt.xlabel("Length of review")
plt.ylabel("Number of review")
plt.show()

print()
print('리뷰 길이 최대값: {}'.format(np.max(train_word_counts)))
print('리뷰 길이 최소값: {}'.format(np.min(train_word_counts)))
print('리뷰 길이 평균값: {}'.format(np.average(train_word_counts)))
print('리뷰 길이 표준편차: {}'.format(np.std(train_word_counts)))
print('리뷰 길이 중간값: {}'.format(np.median(train_word_counts)))
print('리뷰 길이 제1사분위: {}'.format(np.percentile(train_word_counts, 25)))
print('리뷰 길이 제3사분위: {}'.format(np.percentile(train_word_counts, 75)))

qmarks = np.mean(train_data['review'].apply(lambda x: '?' in x))
fullstop = np.mean(train_data['review'].apply(lambda x: '.' in x))
capital_first = np.mean(train_data['review'].apply(lambda x: x[0].isupper()))
capitals = np.mean(train_data['review'].apply(lambda x: max([y.isupper() for y in x])))
numbers = np.mean(train_data['review'].apply(lambda x: max([y.isdigit() for y in x])))

print()
print('물음표가 있는 질문: {}'.format(qmarks))
print('마침표가 있는 질문: {}'.format(fullstop))
print('첫 글자가 대문자인 질문: {}'.format(capital_first))
print('대문자가 있는 질문: {}'.format(capitals))
print('숫자가 있는 질문: {}'.format(numbers))
