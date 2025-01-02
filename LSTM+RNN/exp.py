import nltk
nltk.download('gutenberg')

from nltk.corpus import gutenberg
import pandas as pd

data = gutenberg.raw('shakespeare-hamlet.txt')
with open('hamlet.txt','w') as file:
    file.write(data)


import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

with open('hamlet.txt','r') as file:
    text = file.read().lower()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words =  len(tokenizer.word_index) + 1

print(tokenizer.word_index)

input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequence([line])[0]
    for i in range(1,len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)


