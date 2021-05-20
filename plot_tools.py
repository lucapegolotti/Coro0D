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
    # random.shuffle(portions)
    color = iter(cm.turbo(np.linspace(0, 1 ,n)))
    for i in range(0,n):
        c = next(color)
        ax.plot3D(portions[i].coords[:,0],
                  portions[i].coords[:,1],
                  portions[i].coords[:,2],
                  label=str(i),
                  c = c)
        ncoords = portions[i].coords.shape[0]
        ax.text(portions[i].coords[int(ncoords/2),0],
                portions[i].coords[int(ncoords/2),1],
                portions[i].coords[int(ncoords/2),2],
                str(i),
                color='black',
                fontsize = 7)

    if bifurcations is not None:
        plot_bifurcations(bifurcations, connectivity, fig, ax)

    if show_plot:
        plot_show()

def plot_vessel_portions_highlight(portions, index_highlight, bifurcations = None, connectivity = None, fig = None, ax = None):
    show_plot = False
    if fig == None and ax == None:
        fig, ax = create_fig()
        show_plot = True
    n = len(portions)
    # random.shuffle(portions)
    for i in range(0, n):
        if i == index_highlight:
            ax.plot3D(portions[i].coords[:,0], portions[i].coords[:,1], portions[i].coords[:,2], c = 'red')
        else:
            ax.plot3D(portions[i].coords[:,0], portions[i].coords[:,1], portions[i].coords[:,2], c = 'black')

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

def plot_solution(solutions, times, portions, portion_index, variable_name):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_vessel_portions_highlight(portions, portion_index, fig = fig, ax = ax1)
    if variable_name == 'Pin':
        variable_index = 0
    if variable_name == 'Pout':
        variable_index = 1
    if variable_name == 'Q':
        variable_index = 2
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(times, solutions[portion_index * 3 + variable_index,:])
    ax2.set_title(variable_name + ", portion :" + str(portion_index))
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel(variable_name)
    plot_show()
