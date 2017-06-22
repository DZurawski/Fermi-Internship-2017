# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 13:12:17 2017

@author: kesh1_000
"""

import pandas as pd
import numpy as np

filename = ('linear_data_5k.csv')
dataframe = pd.read_csv(filename)
az = np.add(dataframe.iloc[:,'z'], dataframe.iloc[:,'zl'])
print(az)