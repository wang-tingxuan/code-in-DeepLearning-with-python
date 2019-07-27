# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:03:40 2019

@author: Thinkpad
"""

from keras.datasets import imdb
(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)
#word_index=imdb.get_word_index()
#reverse_word_index=dict([(value,key) for key,value in word_index.items()])
#decoded_review=' '.join(
#        [reverse_word_index.get(i-3,'?') for i in train_data[0]])

import numpy as np
def vectorize_sequences(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1.
    return results

x_train=vectorize_sequences(train_data)
x_test=vectorize_sequences(test_data)

y_train=np.asarray(train_labels).astype('float32')
y_test=np.asarray(test_labels).astype('float32')

from keras import layers
from keras import models

model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

layers0=model.layers[0]
layers1=model.layers[1]
layers2=model.layers[2]

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

x_val=x_train[:10000]
partial_x_train=x_train[10000:]

y_val=y_train[:10000]
partial_y_train=y_train[10000:]

history=model.fit(partial_x_train,
                  partial_y_train,
                  epochs=4,
                  batch_size=512,
                  validation_data=[x_val,y_val]
                  )
#history_dict=history.history
#
#import matplotlib.pyplot as plt
#
#loss_values=history_dict['loss']
#val_loss_values=history_dict['val_loss']
#
#epochs=range(1,len(loss_values)+1)
#
#plt.plot(epochs,loss_values,'bo',label='Training_loss')
#plt.plot(epochs,val_loss_values,'b',label='Validation_loss')
#plt.title('training and validation loss')
#plt.xlabel('epochs')
#plt.ylabel('loss')
#plt.legend()
#plt.show()

results=model.evaluate(x_test,y_test)