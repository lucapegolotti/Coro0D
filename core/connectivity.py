import numpy as np
from numpy import linalg


# we need to look for the position of the bifurcations
def build_slices(portions, tol, maxlength):
    newportions = []
    bifurcations = find_bifurcations(portions, tol)
    for portion in portions:
        curportions = portion.break_at_points(bifurcations, tol)
        for curportion in curportions:
            slicedportions, joints = curportion.limit_length(tol, maxlength)
            newportions += slicedportions
            if bifurcations.shape[0] > 0:
                bifurcations = np.vstack([bifurcations, joints])
            else:
                bifurcations = joints

    bifurcations = simplify_bifurcations(bifurcations, tol)
    bifurcations, connectivity = build_connectivity(newportions, bifurcations, tol)

    return newportions, bifurcations, connectivity


# code: 1 inlet node, -1 outlet node, 2 global input, 3,4,..., outlet nodes
def build_connectivity(portions, bifurcations, tol):
    nportions = len(portions)
    nbifurcations = len(bifurcations)

    connectivity = np.zeros([nbifurcations, nportions])

    for bifindex in range(nbifurcations):
        for porindex in range(nportions):
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
    for porindex in range(nportions):

        bifone = np.where(connectivity[:, porindex] == 1)
        # then, this portion has a global inlet
        if bifone[0].shape[0] == 0:
            bifurcations = np.vstack([bifurcations, portions[porindex].coords[0, :]])
            newconn = np.zeros([1, nportions])
            newconn[0, porindex] = 2
            connectivity = np.vstack([connectivity, newconn])

        bifminusone = np.where(connectivity[:, porindex] == -1)
        if bifminusone[0].shape[0] == 0:
            bifurcations = np.vstack([bifurcations, portions[porindex].coords[-1, :]])
            newconn = np.zeros([1, nportions])
            newconn[0, porindex] = indexglobaloutlet
            connectivity = np.vstack([connectivity, newconn])
            indexglobaloutlet += 1

    return bifurcations, connectivity


def find_bifurcations(portions, tol):
    nportions = len(portions)
    joints = []
    for ipor in range(nportions):
        curpor = portions[ipor]
        curcoords = curpor.coords
        for jpor in range(nportions):
            if jpor != ipor:
                firstcoords = portions[jpor].coords[0, :]
                diff = np.subtract(curcoords, firstcoords)
                magn = np.sqrt(np.square(diff[:, 0]) + np.square(diff[:, 1]) + np.square(diff[:, 2]))

                minmagn = np.amin(magn)
                if minmagn < tol:
                    index = np.where(magn == minmagn)
                    joints.append(curcoords[index])

    joints = np.array(joints).squeeze()

    return simplify_bifurcations(joints, tol)


def simplify_bifurcations(bifurcations, tol):
    nbifurcations = bifurcations.shape[0]

    indices = list(range(nbifurcations))
    clusters = []
    for ibif in range(nbifurcations):
        if indices[ibif] is not None:
            simindices = []
            for jbif in range(nbifurcations):
                if indices[jbif] is not None:
                    if np.linalg.norm(bifurcations[ibif, :] - bifurcations[jbif, :]) < tol:
                        simindices.append(jbif)

            indices[ibif] = None
            cluster = bifurcations[ibif, :]
            for simindex in simindices[1:]:
                cluster += bifurcations[simindex, :]
                indices[simindex] = None

            cluster /= len(simindices)
            clusters.append(cluster)

    return np.array(clusters).squeeze()
