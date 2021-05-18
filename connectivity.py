from parse import *
from plot_tools import *
import numpy as np
from numpy import linalg
from constants import *

# we need to look for the position of the bifurcations
def build_slices(portions):
    fig, ax = create_fig()
    newportions = []
    bifurcations = find_bifurcations(portions)
    for portion in portions:
        curportions = portion.break_at_points(bifurcations, tol)
        for curportion in curportions:
            slicedportions, joints = curportion.limit_length(maxlength)
            newportions += slicedportions
            bifurcations = np.vstack([bifurcations, joints])

    bifurcations = simplify_bifurcations(bifurcations)
    bifurcations = simplify_bifurcations(bifurcations)
    bifurcations, connectivity = build_connectivity(newportions, bifurcations)

    plot_vessel_portions(newportions, fig, ax)
    for i in range(0, bifurcations.shape[0]):
        if np.linalg.norm(connectivity[i,:]) > 0:
            if (np.where(connectivity[i,:] == 2)[0].shape[0] == 1):
                ax.scatter3D(bifurcations[i,0], bifurcations[i,1], bifurcations[i,2], color = 'red');
            elif (np.where(connectivity[i,:] > 2)[0].shape[0] == 1):
                ax.scatter3D(bifurcations[i,0], bifurcations[i,1], bifurcations[i,2], color = 'blue');
            else:
                ax.scatter3D(bifurcations[i,0], bifurcations[i,1], bifurcations[i,2], color = 'green');
    plot_show()

# code: 1 inlet node, -1 outlet node, 2 global input, 3,4,..., outlet nodes
def build_connectivity(portions, bifurcations):
    nportions = len(portions)
    nbifurcations = len(bifurcations)

    connectivity = np.zeros([nbifurcations, nportions])

    for bifindex in range(0, nbifurcations):
        for porindex in range(0, nportions):
            index = portions[porindex].find_closest_point(bifurcations[bifindex], tol)
            if index != -1:
                # this is the case where the bifurcation is at the inlet
                # of the current portion
                if index < portions[porindex].coords.shape[0] / 2:
                    connectivity[bifindex, porindex] = 1
                else:
                    connectivity[bifindex, porindex] = -1

    indexglobaloutlet = 3
    # add global inlet and outlet
    for porindex in range(0, nportions):
        bifone = np.where(connectivity[:,porindex] == 1)
        # then, this portion has a global inlet
        if bifone[0].shape[0] == 0:
            bifurcations = np.vstack([bifurcations, portions[porindex].coords[0,:]])
            newconn = np.zeros([1, nportions])
            newconn[0, porindex] = 2
            connectivity = np.vstack([connectivity, newconn])

        bifminusone = np.where(connectivity[:,porindex] == -1)
        if bifminusone[0].shape[0] == 0:
            bifurcations = np.vstack([bifurcations, portions[porindex].coords[-1,:]])
            newconn = np.zeros([1, nportions])
            newconn[0, porindex] = indexglobaloutlet
            connectivity = np.vstack([connectivity, newconn])
            indexglobaloutlet += 1

    return bifurcations, connectivity

def find_bifurcations(portions):
    nportions = len(portions)
    joints = []
    for ipor in range(0,nportions):
        curpor = portions[ipor]
        curcoords = curpor.coords;
        for jpor in range(0,nportions):
            if jpor != ipor:
                firstcoords = portions[jpor].coords[0,:]
                diff = np.subtract(curcoords, firstcoords)
                magn = np.sqrt(np.square(diff[:,0]) + np.square(diff[:,1]) + np.square(diff[:,2]))

                minmagn = np.amin(magn)
                if (minmagn < tol):
                    index = np.where(magn == minmagn)
                    joints.append(curcoords[index])

    joints = np.array(joints).squeeze()

    return simplify_bifurcations(joints)

def simplify_bifurcations(bifurcations):
    nbifurcations = bifurcations.shape[0]

    indices = list(range(0, nbifurcations))
    clusters = []
    for ibif in range(0, nbifurcations):
        if indices[ibif] != None:
            simindices = []
            for jbif in range(0, nbifurcations):
                if indices[jbif] != None:
                    if np.linalg.norm(bifurcations[ibif,:] - bifurcations[jbif,:]) < tol:
                        simindices.append(jbif)

            indices[ibif] = None
            cluster = bifurcations[ibif,:]
            for simindex in simindices[1:]:
                cluster += bifurcations[simindex,:]
                indices[simindex] = None

            cluster /= len(simindices)
            clusters.append(cluster)


    return np.array(clusters).squeeze()
