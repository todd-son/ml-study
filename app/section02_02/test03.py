from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

text_data = ['나는 배가 고프다',
             '내일 점심 뭐먹지',
             '내일 공부 해야겠다',
             '점심 먹고 공부 해야지']

count_vectorizer = CountVectorizer()
count_vectorizer.fit(text_data)
print(count_vectorizer.vocabulary_)

sentence = [text_data[0]]

print(count_vectorizer.transform(sentence).toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(text_data)

print(tfidf_vectorizer.vocabulary_)

sentence = [text_data[3]]
print(tfidf_vectorizer.transform(text_data).toarray())
