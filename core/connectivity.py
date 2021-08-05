import numpy as np
from numpy import linalg


# we need to look for the position of the bifurcations and stenoses
def build_slices(portions, problem_data):
    newportions = []
    bifurcations = find_bifurcations(portions, problem_data.tol)
    stenoses = None
    if not problem_data.isHealthy:
        if problem_data.autodetect_stenoses:
            stenoses = find_stenoses_automatically(portions,
                                                   threshold=problem_data.threshold_metric,
                                                   threshold_length=problem_data.min_stenoses_length)
        else:
            stenoses = find_stenoses(portions, problem_data.stenoses)
    breakpoints = np.vstack((stenoses, bifurcations)) if stenoses is not None else bifurcations
    breakpoints = simplify_bifurcations(breakpoints, problem_data.tol)

    for portion in portions:
        curportions = portion.break_at_points(breakpoints, problem_data.tol)
        identify_stenoses(curportions, stenoses, 1e-10)
        for (index, curportion) in enumerate(curportions):
            slicedportions, joints = curportion.limit_length(problem_data.tol, problem_data.maxlength)

            newportions += slicedportions
            breakpoints = np.vstack([breakpoints, joints]) if breakpoints.shape[0] > 0 else joints

    identify_stenoses(newportions, stenoses, 1e-10)
    breakpoints = simplify_bifurcations(breakpoints, problem_data.tol)
    breakpoints, connectivity = build_connectivity(newportions, breakpoints,
                                                   problem_data.tol, problem_data.inlet_name)

    return newportions, breakpoints, connectivity


# code: 1 inlet node, -1 outlet node, 2 global input, 3,4,..., outlet nodes, +/- 0.5 inlet/outlet of stenosis
def build_connectivity(portions, bifurcations, tol, inlet_name):
    nportions = len(portions)
    nbifurcations = len(bifurcations)

    connectivity = np.zeros([nbifurcations, nportions])

    for bifindex in range(nbifurcations):
        for porindex in range(nportions):
            min_index = portions[porindex].find_closest_point(bifurcations[bifindex], tol)
            val = 1 if not portions[porindex].isStenotic else 0.5
            if min_index != -1:
                if min_index < portions[porindex].coords.shape[0] / 2:
                    connectivity[bifindex, porindex] = val
                else:
                    connectivity[bifindex, porindex] = -val

    indexglobaloutlet = 3
    # add global inlet and outlet
    for porindex in range(nportions):

        bifin = np.hstack([np.where(connectivity[:, porindex] == 1)[0],
                           np.where(connectivity[:, porindex] == 0.5)[0]])
        # then, this portion has a global inlet
        if bifin.shape[0] == 0:
            if portions[porindex].pathname != inlet_name:
                raise ValueError(f"Invalid inlet recognized in vessel {portions[porindex].pathname}, "
                                 f"while inlet is expected in vessel {inlet_name}")
            bifurcations = np.vstack([bifurcations, portions[porindex].coords[0, :]])
            newconn = np.zeros([1, nportions])
            newconn[0, porindex] = 2
            connectivity = np.vstack([connectivity, newconn])

        bifout = np.hstack([np.where(connectivity[:, porindex] == -1)[0],
                            np.where(connectivity[:, porindex] == -0.5)[0]])
        # then, this portion has a global outlet
        if bifout.shape[0] == 0:
            bifurcations = np.vstack([bifurcations, portions[porindex].coords[-1, :]])
            newconn = np.zeros([1, nportions])
            newconn[0, porindex] = indexglobaloutlet
            connectivity = np.vstack([connectivity, newconn])
            indexglobaloutlet += 1

    invalid_portions = np.where([np.sum(connectivity[:, index_portion]) -
                                 int(np.sum(connectivity[:, index_portion])) >= 1e-2
                                 for index_portion in range(nportions)])[0]
    print(f"Number of invalid portions: {len(invalid_portions)}")
    for invalid_portion in invalid_portions:
        idxs_pos = np.hstack([np.where(connectivity[:, invalid_portion] == 0.5)[0],
                             np.where(connectivity[:, invalid_portion] >= 1)[0]])
        idxs_neg = np.hstack([np.where(connectivity[:, invalid_portion] == -0.5)[0],
                             np.where(connectivity[:, invalid_portion] == -1)[0]])

        connectivity[idxs_pos, invalid_portion] = 1
        connectivity[idxs_neg, invalid_portion] = -1

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

    stenoses_points = []
    window_len = 11
    assert window_len >= 5

    for portion in portions:
        cur_stenoses_points = []
        ncoords = portion.coords.shape[0]
        # stenoticIndicator = np.zeros(ncoords, dtype=bool)
        posindices = np.where(portion.radii > 0)[0]
        posradii = portion.radii[posindices]
        posarclength = portion.arclength[posindices]
        coeffs = np.polyfit(posarclength[:, 0], posradii[:, 0], 7)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(posarclength, posradii, '--o')
        # plt.plot(posarclength, np.polyval(coeffs, posarclength))
        # plt.title(f"{portion.pathname} - radii")
        # plt.show()

        print(f"Considering portion {portion.pathname}")

        # loop over radii from window_len/2 to posindices[-1]-window_len/2
        M = np.ones(ncoords)
        for index in posindices:
            # compute reference radius and area
            r0 = np.polyval(coeffs, portion.arclength[index])
            A0 = np.pi * r0**2

            # evaluate current radius and area
            r = portion.radii[index]
            A = np.pi * r**2

            # evaluate current metric --> M = 1 - |A-A0| / A0
            M[index] = (1.0 - np.abs(A-A0) / A0) if A < A0 else 1.0

            # # STRATEGY 1
            # # if metric < threshold  --> act
            # if M[index] <= threshold:
            #     stenoticIndicator[index] = True
            #
            #     if index >= 2 and not stenoticIndicator[index-1] and stenoticIndicator[index-2]:
            #         stenoticIndicator[index-1] = True
            #
            # else:
            #     stenoticIndicator[index] = False
            #     if index >= 2 and stenoticIndicator[index-1] and not stenoticIndicator[index-2]:
            #         stenoticIndicator[index-1] = False
            #     elif index >= 2 and stenoticIndicator[index-1] and stenoticIndicator[index-2]:
            #         # removing stenosis that are too short
            #         count = 0
            #         idx = index - 1
            #         arclength = 0.0
            #         while stenoticIndicator[idx]:
            #             arclength += np.linalg.norm(portion.coords[idx, :] - portion.coords[idx - 1, :])
            #             count += 1
            #             idx -= 1
            #
            #         print(arclength)
            #
            #         if arclength >= threshold_length:
            #             start_idx = idx + 1
            #             end_idx = index - 1
            #             cur_stenoses_points.append(portion.coords[start_idx, :])
            #             cur_stenoses_points.append(portion.coords[end_idx, :])

        # plt.figure()
        # plt.plot(posarclength, M[posindices], '--o')
        # plt.plot(posarclength, threshold * np.ones_like(posarclength), 'r-.')
        # plt.title(f"{portion.pathname} - M")
        # plt.show()

        # STRATEGY 2
        # identification of stenoses based on the value of M
        cur_stenoses_indices = []
        window_len2 = 5
        for idx in range(ncoords - window_len2):
            if M[idx] < threshold and M[idx+1] < threshold and \
               np.count_nonzero(M[idx:idx+window_len2] < threshold) >= 0.8 * window_len2 and \
               not any([all([M[i:i+int(window_len2/3)]] >= threshold) for i in range(idx, idx-int(window_len2/3))]):
                cur_stenoses_indices.extend(np.arange(idx, idx+window_len2).tolist())

        if cur_stenoses_indices:
            cur_stenoses_indices = np.unique(cur_stenoses_indices)
            start_idx = cur_stenoses_indices[0]
            end_idx = start_idx
            for it in range(1, len(cur_stenoses_indices)):
                if cur_stenoses_indices[it] != cur_stenoses_indices[it-1] + 1:
                    end_idx = cur_stenoses_indices[it-1]
                elif it == len(cur_stenoses_indices) - 1:
                    end_idx = cur_stenoses_indices[-1]

                if end_idx > start_idx:
                    arclength = np.sum([np.linalg.norm(portion.coords[j, :] - portion.coords[j-1, :])
                                        for j in range(start_idx+1, end_idx)])

                    print(arclength)

                    if arclength >= threshold_length:
                        cur_stenoses_points.append(portion.coords[start_idx, :])
                        cur_stenoses_points.append(portion.coords[end_idx, :])

                    start_idx = cur_stenoses_indices[it]

        # if the number of stenotic points is odd, it means that a stenosis-start has been added in the last loop;
        # being it close to an outlet, such stenosis is discarded for simplification
        if len(cur_stenoses_points) % 2 != 0:
            cur_stenoses_points = cur_stenoses_points[:-1]
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

            if (np.min([np.linalg.norm(portion.coords[0, :] - stenoses[i, :])
                        for i in range(0, stenoses.shape[0], 2)]) < tol) or \
               (np.min([np.linalg.norm(portion.coords[-1, :] - stenoses[i, :])
                        for i in range(1, stenoses.shape[0], 2)]) < tol):
                portion.isStenotic = True

    return


def show_stenoses_details(portions, tol):
    print("\n IDENTIFIED STENOSES SUMMARY \n")
    for (index_portion, portion) in enumerate(portions):
        if portion.isStenotic:
            portion.compute_mean_radius()
            portion.compute_min_radius()
            neighs = find_neighbours(portions, index_portion, tol)
            r0 = 0
            for in_neigh in neighs['IN']:
                portions[in_neigh].compute_mean_radius()
                if portions[in_neigh].mean_radius > r0:
                    r0 = portions[in_neigh].mean_radius
            A0 = np.pi * r0**2

            print(f"Stenosis in vessel {portion.pathname} - Portion {index_portion}")
            print(f"Start coordinates: {portion.coords[0,:]}")
            print(f"End coordinates: {portion.coords[-1,:]}")
            print(f"Average radius: {portion.mean_radius} - Minimum radius {portion.min_radius}")
            print(f"Average area: {np.pi * portion.mean_radius**2} - Minimum area {np.pi * portion.min_radius**2}")
            print(f"Length: {portion.arclength[-1]}")
            print(f"Stenosis severity (radius-based): {(1 - portion.min_radius/r0)*100} %")
            print(f"Stenosis severity (area-based): {(1 - np.pi*portion.min_radius**2/A0)*100} %\n")

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


def find_downstream_outlets_portions(portions, portion_index, connectivity, outlets):

    if np.where(connectivity[:, portion_index] > 2)[0].shape[0] == 1:
        outlets.append(portions[portion_index])
        return

    bifs = np.hstack([np.where(connectivity[:, portion_index] == -0.5)[0],
                      np.where(connectivity[:, portion_index] == -1)[0]])

    idxs = []
    for bif in bifs:
        idxs_bif = np.hstack([np.where(connectivity[bif, :] == 0.5)[0],
                              np.where(connectivity[bif, :] == 1)[0]]).tolist()
        idxs.extend(idxs_bif)

    for idx in idxs:
        find_downstream_outlets_portions(portions, idx, connectivity, outlets)

    return


def find_inlet_portion(portions, connectivity):
    for idx_portion in range(len(portions)):
        if np.where(connectivity[:, idx_portion] == 2)[0].shape[0] == 1:
            return portions[idx_portion]
    return


def find_outlet_portions(portions, connectivity, outlets):
    for idx_portion in range(len(portions)):
        if np.where(connectivity[:, idx_portion] > 2)[0].shape[0] == 1:
            outlets.append(portions[idx_portion])
    return


