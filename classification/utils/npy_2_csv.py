# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 14:36:22 2023

@author: Harikrishnan
"""
import numpy as np
import pandas as pd

data_hammer = list()
for i in range(1,11):
    arr = np.load("predictions_hammer/predictions_hammer_" + str(i) + "/contact_pts.npy",allow_pickle=True)
    df = arr.tolist()[-1]
    dataset = pd.DataFrame(df)
    dataset.to_csv("predictions_hammer/predictions_hammer_" + str(i) +".csv",index = False)