"""
    @organization: Fermilab
    @author: Daniel Zurawski
    @author: Keshav Kapoor
    
    Created on Fri June 16 10:43:05 2017
"""

import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequntial
from keras.layers import Dense, Activation

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

def get_input_data(dataframe):
    """ Retrieve the input data in the correct format from 'dataframe'.
        @param  dataframe :: pd.DataFrame
                    A data frame with headers: (id, z, phi, eta, val, r, zl).
        @return np.array 2-D
                    A matrix with columns of data: (r, phi, z)
                    
    """
    return dataframe[["r", "phi", "z"]].values

def get_output_data(dataframe):
    """ Retrieve the output data in the correct format form 'dataframe'.
        @param  dataframe :: pd.DataFrame
                    A data frame with headers: (id, z, phi, eta, val, r, zl).
        @return np.array 2-D
                    A matrix where a given column at index i is the probability
                    vector of which track the particle at index i belongs to.
    """
    IDs     = dataframe[["id"]].values[:,0] # The track id's for each hit.
    uniques = np.unique(IDs) # The unique track id's.
    matrix  = np.zeros((IDs.size, uniques.size)) # The probability matrix.

    # Create a way to map track id number to an index in the probability matrix.
    ID2row = dict() 
    for col, ID in enumerate(uniques):
        ID2row[ID] = col
     
    # Populate the probability matrix with 100% certainty values.
    for col, ID in enumerate(IDs):
        matrix[col, ID2row[ID]] = 1

    return matrix

def main():
    filename  = ('file_o_stuff3.csv')
    dataframe = pd.read_csv(filename)
    
    train_data  = get_input_data (dataframe)
    target_data = get_output_data(dataframe)

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

print("=== The program is starting. ===")
main()
print("=== The program is ending. ===")

