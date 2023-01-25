import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os.path import join as join
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))

learning_rate = [0.01, 0.05]
nodes_list = [1024, 512]
drop_list = [0.3, 0.5]

df = pd.read_csv(join(script_dir, "gender_cv.csv"))
mat = df["Mean Loss"]
mat = mat.to_numpy()

X, Y, Z = np.meshgrid(learning_rate, nodes_list, drop_list)
a = np.array([[mat[0], mat[4]],
              [mat[2], mat[6]]])
b = np.array([[mat[1], mat[5]],
              [mat[3], mat[7]]])
W = np.array([a, b])
U = W.ravel()
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.view_init(-135, 50)
ax.scatter3D(X, Y, Z, c=U, alpha=0.8)
ax.set_xlabel('Kernel standard deviation')
ax.set_ylabel('Nodes')
ax.set_zlabel('Dropout rate')
plt.show()

