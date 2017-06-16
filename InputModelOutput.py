"""
    @organization: Fermilab
    @author: Daniel Zurawski
    @author: Keshav Kapoor
    
    Created on Fri June 16 10:43:05 2017
"""

import pandas as pd
import numpy as np

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
    tracks  = dataframe[["id"]].values[:,0]
    uniques = np.unique(tracks)
    matrix  = np.zeros((tracks.size, uniques.size))
    
    track2row = dict() 
    for i, track in enumerate(uniques):
        track2row[track] = i
     
    for i, track in enumerate(tracks):
        matrix[i, track2row[track]] = 1
        
    return matrix

def main():
    filename  = ("linear_data_1.csv")
    dataframe = pd.read_csv(filename)
    in_data   = get_input_data (dataframe)
    out_data  = get_output_data(dataframe)
    
    print(out_data.shape)

print("=== The program is starting. ===")
main()
print("=== The program is ending. ===")