import pandas as pd
import logging
import numpy as np
from gensim.models import word2vec, Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

data_path = "/Users/todd/word2vec-nlp-tutorial/"
train_clean_data = "train_clean.csv"

train_data = pd.read_csv(data_path + train_clean_data)

reviews = list(train_data['review'])
sentiments = list(train_data['sentiment'])

sentences = []
for review in reviews:
    sentences.append(review.split())

print(sentences[0])

num_features = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3

model: Word2Vec = word2vec.Word2Vec(sentences,
                                    workers=num_workers,
                                    size=num_features,
                                    min_count=min_word_count,
                                    window=context,
                                    sample=downsampling)

model_name = "300features_40minwords_10context"
model.save(model_name)

print(model.wv.index2word)



def get_features(words, model: Word2Vec, num_features):
    feature_vector = np.zeros((num_features), dtype=np.float32)

    num_words = 0

    index2word_set = set(model.wv.index2word)

    for w in words:
        if w in index2word_set:
            num_words += 1
            feature_vector = np.add(feature_vector, model[w])

    feature_vector = np.divide(feature_vector, num_words)
    return feature_vector

def get_dataset(reviews, model, num_features):
    dataset = list()

    for s in reviews:
        dataset.append(get_features(s, model, num_features))

    review_feature_vecs = np.stack(dataset)

    return review_feature_vecs

test_data_vecs = get_dataset(sentences, model, num_features)

print(test_data_vecs[0])

X = test_data_vecs
y = np.array(sentiments)

x_train, x_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

lgs = LogisticRegression(class_weight='balanced')
lgs.fit(x_train, y_train)

print("Accuracy: {}".format(lgs.score(x_eval, y_eval)))
