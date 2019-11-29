import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS


def directory_data(directory: str):
    data = {"review": []}
    for file_path in os.listdir(directory):
        with open(os.path.join(directory, file_path), 'r') as file:
            print("file_path", file_path)
            data["review"].append(file.read())

    return pd.DataFrame.from_dict(data)


data_set = tf.keras.utils.get_file(
    fname="imdb.tar.gz",
    origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    extract=True
)

pos_dir = os.path.join(os.path.dirname(data_set), "aclImdb", "train", "pos")
pos_df = directory_data(pos_dir)
pos_df["sentiment"] = 1

neg_dir = os.path.join(os.path.dirname(data_set), "aclImdb", "train", "neg")
neg_df = directory_data(neg_dir)
neg_df["sentiment"] = 0

train_df = pd.concat([pos_df, neg_df])

print(pos_df.head())
print(neg_df.head())
print(train_df.head())

reviews = list(train_df['review'])

tokenized_reviews = [r.split() for r in reviews]
print(tokenized_reviews[0])

review_len_by_token = [len(t) for t in tokenized_reviews]
print(review_len_by_token[0])

review_len_by_eumjeol = [len(s.replace(' ', '')) for s in reviews]
print(review_len_by_eumjeol[0])

plt.figure(figsize=(12, 5))
plt.hist(review_len_by_token, bins=50, alpha=0.5, color='r', label='word')
plt.yscale('log', nonposy='clip')
plt.title('Review Length Histogram')
plt.xlabel('Review Length')
plt.ylabel('Number of Reviews')
plt.show()

print('문장 최대 길이 : {}'.format(np.max(review_len_by_token)))
print('문장 최소 길이 : {}'.format(np.min(review_len_by_token)))
print('문장 평균 길이 : {}'.format(np.mean(review_len_by_token)))
print('문장 길이 표준편차 : {}'.format(np.std(review_len_by_token)))
print('문장 중간 길이 : {}'.format(np.median(review_len_by_token)))
print('문장 1사분위 길이 : {}'.format(np.percentile(review_len_by_token, 25)))
print('문장 3사분위 길이 : {}'.format(np.percentile(review_len_by_token, 75)))

wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', width=800, height=600).generate(' '.join(train_df['review']))
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

sentiment = train_df['sentiment'].value_counts()
fig, axe = plt.subplots(ncols=1)
fig.set_size_inches(6, 3)
sns.countplot(train_df['sentiment'])
plt.show()


