# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:46:26 2023

@author: Harikrishnan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_hammer = list()
for i in range(1,11):
    arr = np.load("predictions_bowl/predictions_bowl_" + str(i) + "/contact_pts.npy",allow_pickle=True)
    df = arr.tolist()[-1]
    data_hammer.append(df)
    
data = np.concatenate(data_hammer, axis=0)

dataset_hammer = pd.DataFrame(data)

dataset_hammer[3] = "bowl"

column_names = ["x","y","z","object"]

dataset_hammer.columns = column_names

dataset_hammer["sinx"] = np.sin(np.pi*dataset_hammer["x"])
dataset_hammer["cosx"] = np.cos(np.pi*dataset_hammer["x"])
dataset_hammer["siny"] = np.sin(np.pi*dataset_hammer["y"])
dataset_hammer["cosy"] = np.cos(np.pi*dataset_hammer["y"])
dataset_hammer["sinz"] = np.sin(np.pi*dataset_hammer["z"])
dataset_hammer["cosz"] = np.cos(np.pi*dataset_hammer["z"])

#if(dataset_hammer.isna().sum().sum())!=0:
    
    
dataset_hammer.to_csv("bowl_df.csv",index = False)
