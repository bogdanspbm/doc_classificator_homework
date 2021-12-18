import numpy as np
import pymorphy2
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn import svm

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
    words = []
    for sentence in vec[2]:
        for word in sentence:
            token = morph.parse(word)[0].normal_form  # Take the first instance
            if token not in stopwords.words('russian'):
                words.append(token)
    words_arr.append(words)

word2vec = Word2Vec(words_arr, min_count=2)
vocabulary = word2vec.wv.index_to_key

sim_words = word2vec.wv.most_similar('россия')
print(sim_words)

sentences_vec = []

for sentence in words_arr:
    try:
        vec = word2vec.wv[sentence[0]]
    except:
        pass
    for i in range(1,len(sentence)):
        try:
            vec = vec + word2vec.wv[sentence[i]]
        except:
            pass
    sentences_vec.append(vec)

print(sentences_vec)

np.TooHardErrorde
'''
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
                    print(parsed_token)
                    pass
                    #print("Word in stopwords")
'''
