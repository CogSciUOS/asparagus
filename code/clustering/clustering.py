import numpy as np 
import matplotlib.pyplot as plt 

from mpl_toolkits import mplot3d

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# get file path
length = './spaces/m_length_space.npy'
width = './spaces/m_width_space.npy'
hollow = './spaces/m_hollow_space.npy'
blume = './spaces/m_blume_space.npy'
violet = './spaces/m_violet_space.npy'
rust_body = './spaces/m_rost_body_space.npy'
bended = './spaces/m_bended_space.npy'

files = [length,width,hollow,blume,violet,rust_body,bended,hollow]
classes = ["length","width","hollow","flower","violet","bended","rust_body","rust_head"]

legend_elements = [Patch([0], [0], color='b', label='feature present'),
                   Patch([0], [0], color='orange', label='feature not present')]
# create figure
fig, axes = plt.subplots(4,2,figsize=(30,10))
fig.subplots_adjust(hspace=0.7)
fig.legend(handles = legend_elements)
fig.suptitle("Feature-wise scatterplots in 2D sub-space", fontsize=15, fontweight='bold')
count = 0
for i in range(4):
    for j in range(2):
        # load the corresponding eigenspace
        space = np.load(files[count])
        # get the relevant data for 2D plotting aka the first 2 PCs
        xdata_0 = space[:200,0] # first PC feature present
        ydata_0 = space[:200,1] # second PC feature present
        xdata_1 = space[200:,0] # first PC feature not present
        ydata_1 = space[200:,1] # second PC feature not present

        axes[i,j].scatter(xdata_0, ydata_0, marker = ".", color = 'b')
        axes[i,j].set_title(str(classes[count]), {'fontweight':'bold'})
        axes[i,j].set_xlabel("PC1")
        axes[i,j].set_ylabel("PC2")
        axes[i,j].scatter(xdata_1, ydata_0, marker = ".", color = 'orange')
        count += 1
fig.delaxes(axes[3,1])
plt.show()


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(xdata_0, ydata_0, zdata_0, c = cdata_0, cmap = 'Reds')
# ax.scatter3D(xdata_1, ydata_1, zdata_1, c = cdata_1, cmap = 'Greens')
# plt.show()