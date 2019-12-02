import re
import pandas as pd
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

data_path = "/Users/todd/word2vec-nlp-tutorial/"
train_data = pd.read_csv(data_path + "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

review = train_data['review'][0]

print(review)

review_text = BeautifulSoup(review, "html.parser").get_text()

print(review_text)

review_text = re.sub("[^a-zA-z]", " ", review_text)

print(review_text)

review_text = review_text.lower()

print(review_text)

stop_words = set(stopwords.words('english'))
words = review_text.split()
words = [w for w in words if not w in stop_words]
review_text = ' '.join(words)
print(words)
print(review_text)
