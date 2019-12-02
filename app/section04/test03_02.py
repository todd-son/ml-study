import re
import json
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
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
train_data = pd.read_csv(data_path + "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

clean_train_data = []

for review in train_data['review']:
    clean_train_data.append(preprocessing(review))

print(clean_train_data[0])

clean_train_df = pd.DataFrame({'review': clean_train_data, 'sentiment': train_data['sentiment']})

tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_train_data)
text_sequences = tokenizer.texts_to_sequences(clean_train_data)

print(text_sequences[0])

word_vocab = tokenizer.word_index
print(word_vocab)

print("전체 단어 개수: {}".format(len(word_vocab)))

data_configs = {'vocab': word_vocab, 'vocab_size': len(word_vocab) + 1}

MAX_SEQUENCE_LENGTH = 174

train_inputs = pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

print('Shape of train data: ', train_inputs.shape)

train_labels = np.array(train_data['sentiment'])
print('Shape of train label: ', train_labels.shape)

np.save(open(data_path + 'train_input.npy', 'wb'), train_inputs)
np.save(open(data_path + 'train_label.npy', 'wb'), train_labels)

clean_train_df.to_csv(data_path + 'train_clean.csv', index=False)
json.dump(data_configs, open(data_path + 'data_configs.json', 'w'), ensure_ascii=False)







