import numpy as np
from numpy import linalg


# we need to look for the position of the bifurcations and stenoses
def build_slices(portions,
                 stenoses_map, threshold_metric, min_stenoses_length, autodetect_stenoses,
                 tol, maxlength, inlet_name):
    newportions = []
    bifurcations = find_bifurcations(portions, tol)
    if autodetect_stenoses:
        stenoses = find_stenoses_automatically(portions,
                                               threshold=threshold_metric, threshold_length=min_stenoses_length)
    else:
        stenoses = find_stenoses(portions, stenoses_map)
    breakpoints = np.vstack((stenoses, bifurcations)) if stenoses is not None else bifurcations

    for portion in portions:
        curportions = portion.break_at_points(breakpoints, tol)
        identify_stenoses(curportions, stenoses, 1e-16)
        for (index, curportion) in enumerate(curportions):
            slicedportions, joints = curportion.limit_length(tol, maxlength)

            newportions += slicedportions
            breakpoints = np.vstack([breakpoints, joints]) if breakpoints.shape[0] > 0 else joints

    identify_stenoses(newportions, stenoses, tol)
    breakpoints = simplify_bifurcations(breakpoints, tol)
    breakpoints, connectivity = build_connectivity(newportions, breakpoints, tol, inlet_name)

    return newportions, breakpoints, connectivity


# code: 1 inlet node, -1 outlet node, 2 global input, 3,4,..., outlet nodes
def build_connectivity(portions, bifurcations, tol, inlet_name):
    nportions = len(portions)
    nbifurcations = len(bifurcations)

    connectivity = np.zeros([nbifurcations, nportions])

    for bifindex in range(nbifurcations):
        for porindex in range(nportions):
            index = portions[porindex].find_closest_point(bifurcations[bifindex], tol)
            if index != -1:
                # this is the case where the bifurcation is at the inlet of the current portion
                val = 1 if not portions[porindex].isStenotic else 0.5
                if index < portions[porindex].coords.shape[0] / 2:
                    connectivity[bifindex, porindex] = val
                else:
                    connectivity[bifindex, porindex] = -val

    indexglobaloutlet = 3
    # add global inlet and outlet
    for porindex in range(nportions):

        bifone = np.hstack([np.where(connectivity[:, porindex] == 1)[0], np.where(connectivity[:, porindex] == 0.5)[0]])
        # then, this portion has a global inlet
        if bifone.shape[0] == 0:
            if portions[porindex].pathname != inlet_name:
                raise ValueError(f"Invalid inlet recognized in vessel {portions[porindex].pathname}, "
                                 f"while inlet is expected in vessel {inlet_name}")
            bifurcations = np.vstack([bifurcations, portions[porindex].coords[0, :]])
            newconn = np.zeros([1, nportions])
            newconn[0, porindex] = 2
            connectivity = np.vstack([connectivity, newconn])

        bifminusone = np.hstack(
            [np.where(connectivity[:, porindex] == -1)[0], np.where(connectivity[:, porindex] == -0.5)[0]])
        if bifminusone.shape[0] == 0:
            bifurcations = np.vstack([bifurcations, portions[porindex].coords[-1, :]])
            newconn = np.zeros([1, nportions])
            newconn[0, porindex] = indexglobaloutlet
            connectivity = np.vstack([connectivity, newconn])
            indexglobaloutlet += 1

        # TODO: maybe avoid to have stenoses in the inlet / outlet portions

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


def find_stenoses(portions, stenoses_map):
    stenoses_points = []

    for portion in portions:
        split_cur_indices = []
        if portion.pathname in stenoses_map.keys():
            cur_indices = stenoses_map[portion.pathname]

            split_cur_indices.append([cur_indices[0], ])
            for cnt in range(1, len(cur_indices)):
                if cur_indices[cnt] != cur_indices[cnt - 1] + 1:
                    split_cur_indices[-1].append(cur_indices[cnt - 1])
                    split_cur_indices.append([cur_indices[cnt], ])
                if cnt == len(cur_indices) - 1:
                    split_cur_indices[-1].append(cur_indices[cnt])

            cur_stenoses_points = np.zeros((2 * len(split_cur_indices), 3))
            count = 0
            for stenose_index in split_cur_indices:
                id_path_beg = portion.segmented_contours[stenose_index[0]].id_path - 1
                id_path_end = portion.segmented_contours[stenose_index[1]].id_path + 1

                cur_stenoses_points[count, :] = portion.coords[id_path_beg, :]
                cur_stenoses_points[count + 1, :] = portion.coords[id_path_end, :]

                count += 2

            stenoses_points.append(cur_stenoses_points)

    if stenoses_points:
        return_stenoses_points = np.vstack([tmp_stenoses_points for tmp_stenoses_points in stenoses_points])
    else:
        return_stenoses_points = None

    return return_stenoses_points


def find_stenoses_automatically(portions, threshold=0.85, threshold_length=0.5):

    # TODO: future development: automatic detection of stenotic regions based on radia variation?

    stenoses_points = []
    window_len = 9
    assert window_len >= 5

    for portion in portions:
        cur_stenoses_points = []
        ncoords = portion.coords.shape[0]
        stenoticIndicator = np.zeros(ncoords, dtype=bool)
        posindices = np.where(portion.radii > 0)[0]
        M = 1.0

        print(f"Considering portion {portion.pathname}")

        # loop over radii from window_len/2 to posindices[-1]-window_len/2
        for index in range(posindices[0]+int(window_len/2), posindices[-1]-int(window_len/2)):
            # compute reference radius and area
            r0 = np.mean(portion.radii[index-int(window_len/2):index+int(window_len/2)])
            A0 = np.pi * r0**2
            # evaluate current radius and area
            r = portion.radii[index]
            A = np.pi * r**2
            # evaluate metric --> M = 1 - |A-A0| / A0
            M = 1.0 - np.abs(A-A0) / A0
            # if metric < threshold  --> act
            if M <= threshold:
                stenoticIndicator[index] = True

                if stenoticIndicator[index-1] and stenoticIndicator[index-2]:
                    cur_stenoses_points.pop()

                cur_stenoses_points.append(portion.coords[index])

            else:
                stenoticIndicator[index] = False
                if stenoticIndicator[index-1] and not stenoticIndicator[index-2]:
                    cur_stenoses_points.pop()
                    stenoticIndicator[index-1] = False
                elif stenoticIndicator[index-1] and stenoticIndicator[index-2]:
                    # removing stenosis that are too short
                    count = 0
                    idx = index-1
                    arclength = 0.0
                    while stenoticIndicator[idx]:
                        arclength += np.linalg.norm(portion.coords[idx, :] - portion.coords[idx - 1, :])
                        count += 1
                        idx -= 1

                    print(arclength)

                    if arclength < threshold_length:
                        cur_stenoses_points.pop()
                        cur_stenoses_points.pop()

        assert len(cur_stenoses_points) % 2 == 0
        N_cur_stenoses = int(len(cur_stenoses_points) / 2)

        # joining together close stenoses
        for cnt in range(N_cur_stenoses-1):
            dist = np.linalg.norm(cur_stenoses_points[cnt+1] - cur_stenoses_points[cnt+2])
            if dist < threshold_length:
                cur_stenoses_points.pop(cnt+1)
                cur_stenoses_points.pop(cnt+1)
                N_cur_stenoses -= 1

        print(f"Found {N_cur_stenoses} stenoses!\n")
        stenoses_points.extend(cur_stenoses_points)

    if stenoses_points:
        return_stenoses_points = np.vstack([tmp_stenoses_points for tmp_stenoses_points in stenoses_points])
    else:
        return_stenoses_points = None

    return return_stenoses_points


def identify_stenoses(portions, stenoses, tol):

    if stenoses is not None:
        for portion in portions:

            if (np.min([np.linalg.norm(portion.coords[0, :] - stenoses[i, :]) for i in range(0,stenoses.shape[0],2)]) < tol) and \
                    (np.min([np.linalg.norm(portion.coords[-1, :] - stenoses[i, :]) for i in range(1,stenoses.shape[0],2)]) < tol):
                portion.isStenotic = True

    return


def find_neighbours(portions, portion_index, tol):
    neighs = {
        'IN': [],
        'OUT': []
    }

    for (cnt, portion) in enumerate(portions):
        if np.min([np.linalg.norm(portion.coords[0, :] - portions[portion_index].coords[i, :])
                   for i in range(portions[portion_index].coords.shape[0])]) < tol:
            neighs['IN'].append(cnt)
        if np.min([np.linalg.norm(portion.coords[-1, :] - portions[portion_index].coords[i, :])
                   for i in range(portions[portion_index].coords.shape[0])]) < tol:
            neighs['OUT'].append(cnt)

    return neighs
