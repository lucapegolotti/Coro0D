import numpy as np
import contour as Contour
from problem_data import ProblemData
from scipy.integrate import simps
import math

class VesselPortion:
    def __init__(self, xs = None, ys = None, zs = None):
        if xs != None and ys != None and zs != None:
            coords = np.hstack([np.c_[xs],np.c_[ys],np.c_[zs]])
            self.set_coords(coords)

    def set_coords(self, coords):
        self.coords = coords
        self.compute_arclength()

    def compute_arclength(self):
        ncoords = self.coords.shape[0]
        self.arclength = np.zeros([ncoords,1])

        for icoord in range(1, ncoords):
            diff = self.coords[icoord,:] - self.coords[icoord-1,:]
            self.arclength[icoord] = self.arclength[icoord-1] + np.linalg.norm(diff)

    def find_closest_point(self, point, tol):
        ncoords = self.coords.shape[0]
        minindex = -1
        minmagdiff = tol
        for i in range(0, ncoords):
            magdiff = np.linalg.norm(self.coords[i,:] - point)
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
        if np.linalg.norm(self.coords[indices[0],:] - self.coords[0,:]) > tol:
            indices.insert(0,0)

        portions = []
        previndex = indices[0]
        for index in indices[1:]:
            portions.append(self.split(previndex,index))
            previndex = index
        return portions

    # Note: this function should only be called when the path_ids still correspond
    # to the indices of the coordinates! Namely, the vessel must have been just
    # read from file.
    # Contours correspond to a sparse number of coords. Here we interpolate the radii
    # in between through linear interpolation.
    def add_contours(self, contours):
        ncoords = self.coords.shape[0]
        ncontours = len(contours)
        self.contours = [None] * ncoords
        self.radii = np.zeros([ncoords,1])
        for icont in range(0, ncontours):
            curid = contours[icont].id_path
            self.contours[curid] = contours[icont]
            self.radii[curid] = contours[icont].radius
            # interpolate between consecutve radii
            if icont != ncontours-1:
                curarclength = self.arclength[curid]
                nextid = contours[icont+1].id_path
                nextarclength = self.arclength[nextid]
                nextradius = contours[icont+1].radius
                for jcoord in range(curid, nextid):
                    jarclength = self.arclength[jcoord]
                    curradius = self.radii[curid] + (jarclength - curarclength) / (nextarclength - curarclength) * (nextradius - self.radii[curid])
                    self.radii[jcoord] = curradius

    def limit_length(self, tol, length):
        narclengths = self.arclength.shape[0]
        caplength = length
        indicestobreak = []
        joints = np.zeros([0,3])
        totarclength = self.arclength[-1,0]

        # find maximum number of equally sized slices which we can divide the
        # portion while still respecting the limit length
        ndivisions = 1
        while ndivisions * length < totarclength:
            ndivisions += 1

        caplength = totarclength / ndivisions

        for iarchlg in range(0, narclengths):
            if self.arclength[iarchlg] > caplength:
                indicestobreak.append(iarchlg - 1)
                caplength += totarclength / ndivisions
                joints = np.vstack([joints,self.coords[iarchlg - 1,:]])
        return self.break_at_indices(indicestobreak, tol), joints

    def split(self, begin, end):
        newvessel = VesselPortion()
        newvessel.set_coords(self.coords[begin:end+1,:])
        newvessel.contours = self.contours[begin:end+1]
        newvessel.radii = self.radii[begin:end+1,:]
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
        self.R = float(128 * viscosity * self.arclength[-1] / \
                 (math.pi * ((2 * self.mean_radius)**4)))
        return self.R

    def compute_C(self, E, thickness_ratio):
        self.compute_mean_radius()
        self.C = float(math.pi * ((2 * self.mean_radius)**3) * self.arclength[-1] / \
                 (4 * E * thickness_ratio * (2 * self.mean_radius)))
        return self.C

    def compute_L(self, density):
        self.compute_mean_radius()
        self.L = float(4 * density * self.arclength[-1] / \
                 (math.pi * (2 * self.mean_radius)**2))
        return self.L

    def compute_mean_radius(self):
        if not hasattr(self,"mean_radius"):
            posindices = np.where(self.radii > 0)
            posradii = self.radii[posindices]
            posarclength = self.arclength[posindices]
            posarclength = np.subtract(posarclength, posarclength[0])
            # we compute the mean radius using the integral over the arclength
            integrated_radius = simps(posradii, posarclength)
            area = simps(np.ones(posradii.shape),posarclength)
            self.mean_radius = integrated_radius / area
