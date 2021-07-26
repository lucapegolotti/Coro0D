import numpy as np
from scipy.integrate import simps


class VesselPortion:
    def __init__(self, xs=None, ys=None, zs=None, pathname=None):
        if xs is not None and ys is not None and zs is not None:
            coords = np.hstack([np.c_[xs], np.c_[ys], np.c_[zs]])
            self.set_coords(coords)
        self.pathname = pathname
        self.isStenotic = False

        return

    def set_coords(self, coords):
        self.coords = coords
        self.compute_arclength()
        return

    def compute_arclength(self):
        ncoords = self.coords.shape[0]
        self.arclength = np.zeros([ncoords, 1])

        for icoord in range(1, ncoords):
            diff = self.coords[icoord, :] - self.coords[icoord - 1, :]
            self.arclength[icoord] = self.arclength[icoord - 1] + np.linalg.norm(diff)

        return

    def find_closest_point(self, point, tol):
        ncoords = self.coords.shape[0]
        minindex = -1
        minmagdiff = tol
        for i in range(ncoords):
            magdiff = np.linalg.norm(self.coords[i, :] - point)
            if magdiff < minmagdiff:
                minmagdiff = magdiff
                minindex = i

        return minindex

    # break the VesselPortion at points; the points are identified up to tol
    def break_at_points(self, points, tol):
        indicestobreak = []
        for point in points:
            minindex = self.find_closest_point(point, tol)

            if minindex != -1:
                indicestobreak.append(minindex)

        return self.break_at_indices(indicestobreak, tol)

    def break_at_indices(self, indices, tol):
        ncoords = self.coords.shape[0]
        if len(indices) == 0:
            return [self]
        # we add the last index in order to include the last portion
        indices.append(ncoords)
        indices.sort()
        if np.linalg.norm(self.coords[indices[0], :] - self.coords[0, :]) > tol:
            indices.insert(0, 0)

        portions = []
        previndex = indices[0]
        for index in indices[1:]:
            portions.append(self.split(previndex, index))
            previndex = index

        return portions

    def compute_area_outlet(self):
        # find last non zero radius
        index = np.where(self.radii > 0)[0][-1]
        return np.pi * self.radii[index] ** 2

    def set_total_outlet_resistance(self, resistance):
        self.total_outlet_resistance = resistance
        return

    def set_downstream_outlet_resistance(self, resistance):
        self.downstream_outlet_resistance = resistance
        return

    def set_total_outlet_capacitance(self, capacitance):
        self.total_outlet_capacitance = capacitance
        return

    def add_contours(self, contours):
        """
        Note: this function should only be called when the path_ids still correspond
        to the indices of the coordinates! Namely, the vessel must have been just
        read from file.
        Contours correspond to a sparse number of coords. Here we interpolate the radii
        in between through linear interpolation.

        :param contours:
        :type contours:
        :return:
        :rtype:
        """
        ncoords = self.coords.shape[0]
        ncontours = len(contours)
        self.segmented_contours = contours
        self.contours = [None] * ncoords
        self.radii = np.zeros([ncoords, 1])

        if contours[-1].id_path >= ncoords:
            raise ValueError("Invalid segmentation! The segmentation contours are longer than "
                             "the corresponding centerline paths!")

        for icont in range(ncontours):
            curid = contours[icont].id_path
            self.contours[curid] = contours[icont]
            if icont == 0:
                self.radii[:curid+1] = contours[icont].radius
            else:
                self.radii[curid] = contours[icont].radius
            # interpolate between consecutive radii
            if icont != ncontours - 1:
                curarclength = self.arclength[curid]
                nextid = contours[icont + 1].id_path
                nextarclength = self.arclength[nextid]
                nextradius = contours[icont + 1].radius
                for jcoord in range(curid, nextid):
                    jarclength = self.arclength[jcoord]
                    curradius = self.radii[curid] + (jarclength - curarclength) / (nextarclength - curarclength) * \
                                (nextradius - self.radii[curid])
                    self.radii[jcoord] = curradius

        return

    def limit_length(self, tol, length):
        narclengths = self.arclength.shape[0]
        indicestobreak = []
        joints = np.zeros([0, 3])
        totarclength = self.arclength[-1, 0]

        # find maximum number of equally sized slices which we can divide the
        # portion while still respecting the limit length
        ndivisions = 1
        while ndivisions * length < totarclength:
            ndivisions += 1

        caplength = totarclength / ndivisions

        for iarchlg in range(narclengths):
            if self.arclength[iarchlg] > caplength:
                indicestobreak.append(iarchlg - 1)
                caplength += totarclength / ndivisions
                joints = np.vstack([joints, self.coords[iarchlg - 1, :]])

        slicedportions = self.break_at_indices(indicestobreak, tol)

        # for portion in slicedportions:
        #     if portion.arclength[-1] - portion.arclength[0] < 0.1 * length:
        #         raise ValueError(f"A portion has a length of {portion.arclength[-1] - portion.arclength[0]}, which "
        #                          f"is less than 10% the target length of {length}. Try to set different tolerances!")

        return slicedportions, joints

    def split(self, begin, end):
        newvessel = VesselPortion(pathname=self.pathname)
        newvessel.set_coords(self.coords[begin:end + 1, :])
        newvessel.contours = self.contours[begin:end + 1]
        newvessel.radii = self.radii[begin:end + 1, :]
        newvessel.isStenotic = self.isStenotic
        return newvessel

    def show_me(self):
        print("[VesselPortion] show me")
        if hasattr(self, "coords"):
            print("coords:")
            print(self.coords)
        if hasattr(self, "contours"):
            print("contours:")
            print(self.contours)
        if hasattr(self, "radii"):
            print("radii:")
            print(self.radii)
        if hasattr(self, "arclength"):
            print("arclength:")
            print(self.arclength)

    # parameters of the coronary boundary conditions
    def compute_Ra(self):
        return 0.32 * self.total_outlet_resistance

    def compute_Ramicro(self):
        return 0.52 * self.total_outlet_resistance

    def compute_Rvmicro(self):
        return 0.0

    def compute_Rv(self):
        return 0.16 * self.total_outlet_resistance

    def compute_Ca(self):
        return 0.11 * self.total_outlet_capacitance

    def compute_Cim(self):
        return 0.89 * self.total_outlet_capacitance

    def compute_mean_radius(self):
        if not hasattr(self, "mean_radius"):
            posindices = np.where(self.radii > 0)
            posradii = self.radii[posindices]
            posarclength = self.arclength[posindices]

            if len(posarclength) == 0:
                breakpoint()
                posarclength = np.subtract(posarclength, posarclength[0])
            # we compute the mean radius using the integral over the arclength
            integrated_radius = simps(posradii, posarclength)
            self.mean_radius = integrated_radius / (posarclength[-1] - posarclength[0])

            return

    def compute_min_radius(self):
        if not hasattr(self, "min_radius"):
            posindices = np.where(self.radii > 0)
            self.min_radius = np.min(self.radii[posindices])

            return
