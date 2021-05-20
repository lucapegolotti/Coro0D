from inletbc import InletBC
from outletbc import OutletBC
import numpy as np
import math

class BCManager:
    def __init__(self, portions, connectivity, inletbc_type, outletbc_type):
        self.portions = portions
        self.connectivity = connectivity
        self.inletbc_type = inletbc_type
        self.outletbc_type = outletbc_type
        self.create_bcs()

    # we set the row where the boundary conditions start in matrices and vectors
    def set_starting_row_bcs(self, row):
        self.starting_row = row

    def create_bcs(self):
        # get index of inlet block
        self.inletindex = int(np.where(self.connectivity == 2)[1])
        print(self.inletindex)
        self.inletbc = InletBC(self.portions[self.inletindex], \
                               self.inletindex, self.inletbc_type)

        # find max outlet flag
        maxoutletflag = int(np.max(self.connectivity))
        self.outletindices = []
        self.outletbcs = []
        for flag in range(3, maxoutletflag + 1):
            self.outletindices.append(int(np.where(self.connectivity == flag)[1]))
            self.outletbcs.append(OutletBC(self.portions[self.outletindices[-1]],
                                  self.outletindices[-1], self.outletbc_type))
        self.noutlets = len(self.outletbcs)

    # rowbcs is the first index of the boundary conditions
    def add_bcs_dot(self, matrix_dot):
        self.inletbc.apply_bc_matrix_dot(matrix_dot, self.starting_row)

        curcol = len(self.portions) * 3
        currow = self.starting_row + 1
        for ibc in range(0, len(self.outletbcs)):
            currow += self.outletbcs[ibc].apply_bc_matrix_dot(matrix_dot,
                                                              currow,
                                                              curcol)
            curcol += self.outletbcs[ibc].nvariables

    # rowbcs is the first index of the boundary conditions
    def add_bcs(self, matrix_dot):
        self.inletbc.apply_bc_matrix(matrix_dot, self.starting_row)

        curcol = len(self.portions) * 3
        currow = self.starting_row + 1
        for ibc in range(0, len(self.outletbcs)):
            currow += self.outletbcs[ibc].apply_bc_matrix(matrix_dot,
                                                          currow,
                                                          curcol)

            curcol += self.outletbcs[ibc].nvariables

    def apply_bc_vector(self, vector, time):
        vector[self.starting_row] = math.sin(time)
