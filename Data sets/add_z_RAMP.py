# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 16:49:14 2017

@author: kesh1_000
"""

import pandas as pd
import numpy as np

filename = ("public_train_100MeV.csv")
allData = pd.read_csv(filename)
allData["act_z"] = (np.zeros(allData["x"].size))
byEvent = allData.groupby("event_id")
print(byEvent)