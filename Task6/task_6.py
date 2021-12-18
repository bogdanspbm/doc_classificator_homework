import pymorphy2
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec

nltk.download('punkt')
nltk.download('stopwords')

data = []
morph = pymorphy2.MorphAnalyzer()

# Read Text
with open("data/news_train.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        vec = []
        category, topic, content = line.split("\t")
        vec.append(category)
        vec.append(topic)
        vec.append(content)
        data.append(vec)

# Tokenize

data_tokens = []

for vec in data:
    vec_tokenize = [vec[0]]
    sentences = sent_tokenize(vec[1])
    vec_tokenize.append([word_tokenize(sent) for sent in sentences])
    sentences = sent_tokenize(vec[1])
    vec_tokenize.append([word_tokenize(sent) for sent in sentences])
    data_tokens.append(vec_tokenize)

# Process tokens

counter = 0
token_limit = 5


for vec in data_tokens:
    for i in range(1,2):
        sentence = vec[i]
        for tokens in sentence:
            for token in tokens:
                print(morph.parse(token))
                counter += 1
                if counter > token_limit:
                    break
            if counter > token_limit:
                break
        if counter > token_limit:
            break
    if counter > token_limit:
        break

# Fill all words for train

words_arr = []
for vec in data_tokens:
    for i in range(1,2):
        sentence = vec[i]
        for tokens in sentence:
            for token in tokens:
                parsed_token = morph.parse(token)[0].word # Take the first instance
                if parsed_token not in stopwords.words('russian'):
                    words_arr.append(parsed_token)
                else:
                    pass
                    #print("Word in stopwords")

print(words_arr)
word2vec = Word2Vec(words_arr, min_count=2)
vocabulary = word2vec.wv.index_to_key
print(vocabulary)