import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.pyplot import cm
import numpy as np
import random

def create_fig():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    return fig, ax

def plot_show():
    plt.show()

def plot_vessel_portions(portions, fig, ax):
    n = len(portions)
    random.shuffle(portions)
    color = iter(cm.turbo(np.linspace(0, 1 ,n)))
    for vessel in portions:
        c = next(color)
        ax.plot3D(vessel.coords[:,0], vessel.coords[:,1], vessel.coords[:,2], c = c)
