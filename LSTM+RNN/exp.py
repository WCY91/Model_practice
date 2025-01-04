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


max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences,maxlen = max_sequence_len,padding='pre'))

import tensorflow as tf
x,y=input_sequences[:,:-1],input_sequences[:,-1]

y= tf.keras.utils.to_categorical(y,num_classes = total_words)

x_train,x_test ,y_train,y_test = train_test_split(x,y,test_size=0.2)

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(mointor = 'val_loss',patience = 3, restore_best_weights=True)

# define model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Embedding,LSTM,Dropout,GRU


# LSTM
model = Sequential()
model.add(Embedding(total_words,100,input_length = max_sequence_len - 1))
model.add(LSTM(150,return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words,activation="softmax"))

model.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()


# GRU
gru_model = Sequential()
gru_model.add(Embedding(total_words,100,input_length=max_sequence_len-1))
gru_model.add(GRU(150,return_sequences = True))
gru_model.add(Dropout(0.2))
gru_model.add(GRU(100))
gru_model.add(Dense(total_words,activation='softmax'))

gru_model.compile(optimizer ='adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])
history=model.fit(x_train,y_train,epochs=50,validation_data=(x_test,y_test),verbose=1,callbacks=[early_stopping])
gru_history =gru_model.fit(x_train,y_train,epochs=50,validation_data=(x_test,y_test),verbose=1,callbacks=[early_stopping])

def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  
    token_list = pad_sequences([token_list],maxlen = max_sequence_len-1,padding='pre')
    predicted = model.predict(token_list,verbose=0)
    predict_word_index = np.argmax(predicted,1)
    for word , index in tokenizer.word_index.items():
        if index == predict_word_index:
            return word
    
    return None

input_text="To be or not to be"
print(f"Input text:{input_text}")
max_sequence_len=model.input_shape[1]+1
next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
print(f"Next Word PRediction:{next_word}")


gru_next_word=predict_next_word(gru_model,tokenizer,input_text,max_sequence_len)
print(f"Next Word GRU PRediction:{gru_next_word}")

model.save("next_word_lstm.h5")
gru_model.save("next_word_gru.h5")
## Save the tokenizer
import pickle
with open('tokenizer.pickle','wb') as handle:
    pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)

input_text="  Barn. Last night of all,When yond same"
print(f"Input text:{input_text}")
max_sequence_len=model.input_shape[1]+1
next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
print(f"Next Word PRediction:{next_word}")