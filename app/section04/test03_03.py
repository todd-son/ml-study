import re

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer


def preprocessing(review: str):
    review_text = BeautifulSoup(review, "html.parser").get_text()
    review_text = re.sub("[^a-zA-z]", " ", review_text)
    review_text = review_text.lower()
    stop_words = set(stopwords.words('english'))
    words = review_text.split()
    words = [w for w in words if not w in stop_words]
    return ' '.join(words)

data_path = "/Users/todd/word2vec-nlp-tutorial/"
test_data = pd.read_csv(data_path + "testData.tsv", header=0, delimiter="\t", quoting=3)

clean_test_data = []

for review in test_data['review']:
    clean_test_data.append(preprocessing(review))

clean_test_df = pd.DataFrame({'review': clean_test_data, 'id': test_data['id']})

MAX_SEQUENCE_LENGTH = 17

tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_test_data)
text_sequences = tokenizer.texts_to_sequences(clean_test_data)
test_inputs = pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

test_id = np.array(test_data['id'])

np.save(open(data_path + 'test_input.npy', 'wb'), test_inputs)
np.save(open(data_path + 'test_id.npy', 'wb'), test_id)

clean_test_df.to_csv(data_path + 'test_clean.csv', index=False)


