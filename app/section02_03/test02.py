import spacy

nlp = spacy.load('en')

sentence = "Natural language processing (NLP) is a subfield of computer science," \
           "information engineering, and artificial intelligence concerned with the interactions between computers " \
           "and human (natural) languages, in particular how to program computers to process and analyze large " \
           "amounts of natural language data. Challenges in natural language " \
           "processing frequently involve speech recognition."

doc = nlp(sentence)

word_tokenized_sentence = [token.text for token in doc]
sentence_tokenized_list = [sent.text for sent in doc.sents]
print(word_tokenized_sentence)
print(sentence_tokenized_list)
