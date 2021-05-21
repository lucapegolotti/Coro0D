import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.pyplot import cm
import numpy as np
import random
from matplotlib import animation
import mpl_toolkits.mplot3d.axes3d as p3

def create_fig():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    return fig, ax

def plot_show():
    plt.show()

def plot_vessel_portions(portions, bifurcations = None, connectivity = None, fig = None, ax = None, color = None):
    show_plot = False
    if fig == None and ax == None:
        fig, ax = create_fig()
        show_plot = True
    n = len(portions)
    lines = []
    if color == None:
        color = iter(cm.turbo(np.linspace(0, 1 ,n)))
    for i in range(0,n):
        if color == None:
            c = next(color)
        else:
            c = color
        lines.append(ax.plot3D(portions[i].coords[:,0],
                     portions[i].coords[:,1],
                     portions[i].coords[:,2],
                     label=str(i),
                     c = c))
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

    return lines

def plot_vessel_portions_highlight(portions, index_highlight, bifurcations = None, connectivity = None, fig = None, ax = None):
    show_plot = False
    if fig == None and ax == None:
        fig, ax = create_fig()
        show_plot = True
    n = len(portions)
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

def plot_solution(solutions, times, t0, T, portions, portion_index, variable_name):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_vessel_portions_highlight(portions, portion_index, fig = fig, ax = ax1)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.set_zticklabels([])
    scale = 1
    if variable_name == 'Pin':
        variable_index = 0
        scale = 1 / 1333.2 # we convert the pressure back to mmHg
        unit = ' [mmHg]'
    if variable_name == 'Pout':
        variable_index = 1
        scale = 1 / 1333.2
        unit = ' [mmHg]'
    if variable_name == 'Q':
        variable_index = 2
        unit = ' [mL/s]'
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(times, solutions[portion_index * 3 + variable_index,:] * scale)
    ax2.set_title(variable_name + ", portion: " + str(portion_index))
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel(variable_name + unit)
    ax2.set_xlim([t0, T])
    return fig, ax1, ax2

def show_animation(solutions, times, t0, portions, variable_name, resample, inlet_index = None):
    nportions = len(portions)
    if inlet_index == None:
        fig = plt.figure()
        ax1 = p3.Axes3D(fig)
    else:
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    ax1.grid(False)
    ax1.axis('off')

    # we keep only the solutions from t0 on
    indices = np.where(times >= t0)[0]
    times = times[indices]
    solutions = solutions[0:nportions*3,indices]

    times = times[::resample]

    variables = solutions[:3*nportions,::resample]
    if variable_name == 'Pin':
        selectvariables = variables[0::3,:] / 1333.2
        units = ' [mmHg]'
    elif variable_name == 'Pout':
        selectvariables = variables[1::3,:] / 1333.2
        units = ' [mmHg]'
    elif variable_name == 'Q':
        selectvariables = variables[2::3,:]
        units = ' [mL/s]'

    minv = np.min(selectvariables)
    maxv = np.max(selectvariables)

    def update(num, ax, times, selectvariables, lines, minv, maxc, timestamp):
        nlines = len(lines)
        for i in range(0, nlines):
            lines[i][0].set_color(cm.jet((selectvariables[i,num] - minv)/(maxv - minv)))
        timestamp.set_text('t = ' + "{:.2f}".format(times[num]) + " s")

    def update_dual(num, ax, times, selectvariables, lines, minv, maxc, timestamp, dot, inflow):
        nlines = len(lines)
        for i in range(0, nlines):
            lines[i][0].set_color(cm.jet((selectvariables[i,num] - minv)/(maxv - minv)))
        timestamp.set_text('t = ' + "{:.2f}".format(times[num]) + " s")

        dot.set_data(times[num],inflow[num])

    N = times.shape[0]

    lines = plot_vessel_portions(portions, fig = fig, ax = ax1, color = 'black')
    timestamp = ax1.text2D(0.05, 0.95, 't = ' + "{:.2f}".format(times[0]) + " s", transform=ax1.transAxes)
    # trick to display the colorbar
    p = ax1.scatter([],[],[],c = [], cmap=plt.cm.jet)
    p.set_clim(minv, maxv)

    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.set_zticklabels([])

    cbaxes = fig.add_axes([0.01, 0.2, 0.015, 0.6])
    cbar = fig.colorbar(p, cax=cbaxes, shrink = 0.7)
    cbar.set_label(variable_name + units, rotation=90, labelpad = 20)

    if inlet_index == None:
        anim = animation.FuncAnimation(fig, update, N,
                                       fargs=(ax1,
                                              times,
                                              selectvariables,
                                              lines,
                                              minv,
                                              maxv,
                                              timestamp),
                                       interval = 10,
                                       blit=False)
    else:
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(times, solutions[inlet_index * 3 + 0,::resample] / 1333.2)
        ax2.set_xlim([times[0], times[-1]])
        ax2.set_xlabel('t [s]')
        ax2.set_ylabel('Pin [mmHg]')
        dot, = ax2.plot(times[0], solutions[inlet_index * 3 + 0,0] / 1333.2,'ro')
        anim = animation.FuncAnimation(fig, update_dual, N,
                                       fargs=(ax1, times,
                                              selectvariables,
                                              lines, minv,
                                              maxv, timestamp,
                                              dot,
                                              solutions[inlet_index * 3 + 0,::resample] / 1333.2),
                                       interval = 10,
                                       blit=False)

    plot_show()
    # anim.save('simulation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
