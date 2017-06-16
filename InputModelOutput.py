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
    IDs     = dataframe[["id"]].values[:,0] # The track id's for each hit.
    uniques = np.unique(IDs) # The unique track id's.
    matrix  = np.zeros((IDs.size, uniques.size)) # The probability matrix.
    
    # Create a way to map track id number to an index
    # in the probability matrix.
    ID2row = dict() 
    for col, ID in enumerate(uniques):
        ID2row[ID] = col
     
    # Populate the probability matrix with 100% certainty values.
    for col, ID in enumerate(IDs):
        matrix[col, ID2row[ID]] = 1

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