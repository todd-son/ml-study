from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import nltk

sentence = "Natural language processing (NLP) is a subfield of computer science," \
           "information engineering, and artificial intelligence concerned with the interactions between computers " \
           "and human (natural) languages, in particular how to program computers to process and analyze large " \
           "amounts of natural language data. Challenges in natural language " \
           "processing frequently involve speech recognition."

print(word_tokenize(sentence))
print(sent_tokenize(sentence))
