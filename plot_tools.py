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

def plot_vessel_portions(portions, bifurcations = None, connectivity = None, fig = None, ax = None):
    if fig == None and ax == None:
        fig, ax = create_fig()
        show_plot = True
    n = len(portions)
    random.shuffle(portions)
    color = iter(cm.turbo(np.linspace(0, 1 ,n)))
    for vessel in portions:
        c = next(color)
        ax.plot3D(vessel.coords[:,0], vessel.coords[:,1], vessel.coords[:,2], c = c)

    if bifurcations is not None:
        plot_bifurcations(bifurcations, connectivity, fig, ax)

    if show_plot:
        plot_show()

def plot_bifurcations(bifurcations, connectivity, fig, ax):
    for i in range(0, bifurcations.shape[0]):
        if np.linalg.norm(connectivity[i,:]) > 0:
            if (np.where(connectivity[i,:] == 2)[0].shape[0] == 1):
                ax.scatter3D(bifurcations[i,0], bifurcations[i,1], bifurcations[i,2], color = 'red');
            elif (np.where(connectivity[i,:] > 2)[0].shape[0] == 1):
                ax.scatter3D(bifurcations[i,0], bifurcations[i,1], bifurcations[i,2], color = 'blue');
            else:
                ax.scatter3D(bifurcations[i,0], bifurcations[i,1], bifurcations[i,2], color = 'green');
