# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:43:05 2017

@author: kesh1_000
"""



import pandas as pd
import numpy as np


import matplotlib.pyplot as plt

import keras
from keras.models import Sequntial
from keras.layers import Dense, Activation



def getInputData(filename):
    #Takes a csv file with headers: id, z, phi, eta, val, r, zl
    #Returns a 3D array with values z, phi, r to put into a network(as input)
    
    initData = pd.read_csv(filename)
    print(initData)
    
    #Remove id, eta, val, and z
    refData = initData.drop(['id', 'eta', 'val', 'zl'], axis=1).values
    return refData
# HELLO THERE! (Daniel)

def show_losses( histories ):
    plt.figure(figsize=(10,10))
    #plt.ylim(bottom=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Error by Epoch')
    colors=[]
    do_acc=False
    for label,loss in histories:
        color = tuple(np.random.random(3))
        colors.append(color)
        l = label
        vl= label+" validation"
        if 'acc' in loss.history:
            l+=' (acc %2.4f)'% (loss.history['acc'][-1])
            do_acc = True
        if 'val_acc' in loss.history:
            vl+=' (acc %2.4f)'% (loss.history['val_acc'][-1])
            do_acc = True
        plt.plot(loss.history['loss'], label=l, color=color)
        if 'val_loss' in loss.history:
            plt.plot(loss.history['val_loss'], lw=2, ls='dashed', label=vl, color=color)


    plt.legend()
    plt.yscale('log')
    plt.show()
    if not do_acc: return
    plt.figure(figsize=(10,10))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    for i,(label,loss) in enumerate(histories):
        color = colors[i]
        if 'acc' in loss.history:
            plt.plot(loss.history['acc'], lw=2, label=label+" accuracy", color=color)
        if 'val_acc' in loss.history:
            plt.plot(loss.history['val_acc'], lw=2, ls='dashed', label=label+" validation accuracy", color=color)
    plt.legend(loc='lower right')
    plt.show()

data = getInputData('file_o_stuff3.csv')
print(data)

train_data = data
target_data = outData

simple = Sequntial()
simple.add(Dense(32, input_dim=np.unique(tracks).size*4, activation='relu'))
simple.add(Dense(3, kernel_initializer='uniform'))
simple.add(Activation('softmax'))
simple.compile(loss='categorical_crossentropy', optimizer='sgd', metrics['accuracy'])

simple.summary()

hist = simple.fit( train_data,
                  target_data,
                  epochs=10,
                  batch_size=5,
                  verbose=1,
                  validation_split=0.2,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min'),
                            ModelCheckpoint(filepath='simple.h5', verbose=0)]
                  )

show_losses([("cat x entropy", hist)])