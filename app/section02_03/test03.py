from konlpy.tag import Okt

okt = Okt()

text = "한글 자연어 처리는 재밌다 이제부터 열심히 해야지ㅎㅎㅎ"

print(okt.morphs(text))
print(okt.morphs(text, stem=True))
print(okt.nouns(text))
print(okt.phrases(text))

print(okt.pos(text))
print(okt.pos(text, join=True))
