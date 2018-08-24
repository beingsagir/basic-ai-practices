import os
import pandas as pd
import nltk
import gensim
from gensim import corpora, models, similarities

print("Hello Gensim")
os.chdir("/Users/sagir/Documents/data/ai/deeplearning")
df = pd.read_csv('jokes.csv')

x = df['Question'].values.tolist()
y = df['Answer'].values.tolist()

corpus = x + y

token_corpus = [nltk.word_tokenize(sent) for sent in corpus]

#-----------------------------------------------
#For trainnig and creating the model
#-----------------------------------------------
#model = gensim.models.Word2Vec(token_corpus, min_count = 1, size = 32)

#-----------------------------------------------
#For Saving the model
#-----------------------------------------------
#model.save('jokeModelSaved')

model = gensim.models.Word2Vec.load('jokeModelSaved')

print(model.most_similar('Hi'))
#model.most_similar([vector])







