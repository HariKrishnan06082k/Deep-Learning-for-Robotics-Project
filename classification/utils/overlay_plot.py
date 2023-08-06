import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


PC_DIR = "npy/skillet.npy"
CP_DIR = "predictions_skillet/contact_pts.npy"

df = pd.read_csv("hammer_1.csv")

c_point = df.iloc[:,:3].values

xc = c_point[:,0:1]
yc = c_point[:,1:2]
zc = c_point[:,2:3]

fig = plt.figure(figsize=(12,7))
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')    

ax.scatter(-0.068746306,  -0.01845745, 0.01614229, c='black', marker='*', lw=0)
ax.scatter(xc,yc,zc, c='red', marker='.', lw=0,alpha=0.005)

ax.set_xlabel("Z")
ax.set_ylabel("X")
ax.set_zlabel("Y")

plt.show()