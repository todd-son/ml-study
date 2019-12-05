import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data_path = "/Users/todd/word2vec-nlp-tutorial/"
train_clean_data = "train_clean.csv"
train_data = pd.read_csv(data_path + train_clean_data, header=0, delimiter=",", quoting=3)

print(train_data.head())
print(len(train_data))
print(train_data.keys())

reviews = list(train_data['review'])
sentiments = list(train_data['sentiment'])

vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char", sublinear_tf=True, ngram_range=(1, 3), max_features=5000)

X = vectorizer.fit_transform(reviews)
y = np.array(sentiments)

x_train, x_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

lgs = LogisticRegression(class_weight='balanced')
lgs.fit(x_train, y_train)

print("Accuracy: {}".format(lgs.score(x_eval, y_eval)))

test_clean_data = "test_clean.csv"
test_data = pd.read_csv(data_path + test_clean_data, header=0, delimiter=',', quoting=3)

test_data_vecs = vectorizer.fit_transform(test_data['review'])
test_prediceted = lgs.predict(test_data_vecs)
print(test_prediceted)

ids = test_data['id']
answer_dataset = pd.DataFrame({'id': ids, 'sentiment': test_prediceted})
answer_dataset.to_csv(data_path + 'lgs_tfidf_answer.csv', index=False, quoting=3)
