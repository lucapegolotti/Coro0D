import numpy as np
from scipy.integrate import simps
import math


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
        return math.pi * self.radii[index] ** 2

    def set_total_outlet_resistance(self, resistance):
        self.total_outlet_resistance = resistance
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

        return self.break_at_indices(indicestobreak, tol), joints

    def split(self, begin, end):
        newvessel = VesselPortion(pathname=self.pathname)
        newvessel.set_coords(self.coords[begin:end + 1, :])
        newvessel.contours = self.contours[begin:end + 1]
        newvessel.radii = self.radii[begin:end + 1, :]
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

    # resistance, capacitance, and inductance where taken from
    # "Design of a 0D image-based coronary blood flow model" by Uus, Liatsis
    def compute_R(self, viscosity):
        self.compute_mean_radius()
        self.R = float(128 * viscosity * self.arclength[-1] /
                       (np.pi * ((2 * self.mean_radius) ** 4)))
        return self.R

    def compute_C(self, E, thickness_ratio):
        self.compute_mean_radius()
        self.C = float(np.pi * ((2 * self.mean_radius) ** 3) * self.arclength[-1] /
                       (4 * E * thickness_ratio * (2 * self.mean_radius)))
        return self.C

    def compute_L(self, density):
        self.compute_mean_radius()
        self.L = float(4 * density * self.arclength[-1] /
                       (np.pi * (2 * self.mean_radius) ** 2))
        return self.L

    # parameters of the Young-Tsai (1973) 0D stenosis model, taken from
    # "Reduced Order Model for Transstenotic Pressure Drop in Coronary Arteries" by Mirramezani et al. (2019)
    def compute_R_YT(self, viscosity, r0):
        self.compute_min_radius()
        La = 0.83 * self.arclength[-1] + 3.28 * self.min_radius
        Kv = 16 * La / r0 * (r0/self.min_radius)**4
        self.R_YT = float((viscosity * Kv) /
                          (2 * np.pi * r0**3))

        return self.R_YT

    def compute_R2_YT(self, density, r0):
        self.compute_min_radius()
        Kt = 1.52
        self.R2_YT = float((Kt * density) /
                           (2 * np.pi**2 * r0**4) *
                           ((r0/self.min_radius)**2 - 1)**2)

        return self.R2_YT

    def compute_L_YT(self, density, r0):
        Ku = 1.2
        self.L_YT = float((Ku * density * self.arclength[-1]) /
                          (np.pi * r0**2))

        return self.L_YT

    # parameters of the Itu-Sharma (2012) 0D stenosis model, taken from
    # "Reduced Order Model for Transstenotic Pressure Drop in Coronary Arteries" by Mirramezani et al. (2019)
    def compute_R_IS(self, density, viscosity, HR, r0):
        self.compute_min_radius()
        omega = HR * 2 * np.pi / 60.0
        alpha = r0 * np.sqrt(density * omega / viscosity)
        Kv = 1 + 0.053 * self.min_radius / r0 * alpha**2

        posindices = np.where(self.radii > 0)
        posradii = self.radii[posindices]
        posarclength = self.arclength[posindices]
        posarclength = np.subtract(posarclength, posarclength[0])
        # we compute the mean radius using the integral over the arclength
        integral = simps(posradii**(-4), posarclength)
        Rvc = (8 * viscosity / np.pi) * integral

        self.R_IS = Kv * Rvc

        return self.R_IS

    def compute_R2_IS(self, density, r0):
        self.R2_IS = self.compute_R2_YT(density, r0)

        return self.R2_IS

    def compute_L_IS(self, density):
        Ku = 1.2

        posindices = np.where(self.radii > 0)
        posradii = self.radii[posindices]
        posarclength = self.arclength[posindices]
        posarclength = np.subtract(posarclength, posarclength[0])
        # we compute the mean radius using the integral over the arclength
        integral = simps(posradii**(-2), posarclength)
        Lu = (density / np.pi) * integral

        self.L_IS = Ku * Lu

        return self.L_IS

    # parameters of the Garcia (2005) 0D stenosis model, taken from
    # "Reduced Order Model for Transstenotic Pressure Drop in Coronary Arteries" by Mirramezani et al. (2019)
    def compute_R2_G(self, density, r0):
        self.compute_min_radius()
        self.R2_G = float((density/2) *
                          ((r0**2 - self.min_radius**2) / (np.pi * r0**2 * self.min_radius**2))**2)

        return self.R2_G

    def compute_L_G(self, density, r0):
        self.compute_min_radius()
        alpha = 6.28
        beta = 0.5
        self.L_G = float((alpha * density) /
                         (np.sqrt(np.pi) * r0) *
                         ((r0/self.min_radius)**2 - 1)**beta)

        return self.L_G

    # parameters of the Resistance/Windkessel2 model extended to stenoses (no reference)
    def compute_R2(self, density, r0):
        self.compute_mean_radius()
        Kt = 1.5
        self.R2_WKs = float((Kt * density) /
                            (2 * np.pi**2 * r0**4) *
                            ((r0/self.mean_radius)**2 - 1)**2)

        return self.R2_WKs

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
            posarclength = np.subtract(posarclength, posarclength[0])
            # we compute the mean radius using the integral over the arclength
            integrated_radius = simps(posradii, posarclength)
            # area = simps(np.ones(posradii.shape),posarclength)
            self.mean_radius = integrated_radius / (posarclength[-1] - posarclength[0])  # area

            return

    def compute_min_radius(self):
        if not hasattr(self, "min_radius"):
            posindices = np.where(self.radii > 0)
            self.min_radius = np.min(self.radii[posindices])

            return
