# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:43:05 2017

@author: kesh1_000
"""

import pandas as pd
import numpy as np

def getInputData(filename):
    #Takes a csv file with headers: id, z, phi, eta, val, r, zl
    #Returns a 3D array with values z, phi, r to put into a network(as input)
    
    initData = pd.read_csv(filename)
    print(initData)
    
    #Remove id, eta, val, and z
    refData = initData.drop(['id', 'eta', 'val', 'zl'], axis=1).values
    return refData

data = getInputData("file_o_stuff3.csv")
print(data)