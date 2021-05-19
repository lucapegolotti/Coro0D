from inletbc import InletBC
from outletbc import OutletBC
import numpy as np

class BCManager:
    def __init__(self, portions, connectivity, inletbc_type, outletbc_type):
        self.portions = portions
        self.connectivity = connectivity
        self.inletbc_type = inletbc_type
        self.outletbc_type = outletbc_type
        self.create_bcs()

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
    def add_bcs_dot(self, matrix_dot, rowbcs):
        self.inletbc.apply_bc_matrix_dot(matrix_dot, rowbcs)

        curcol = len(self.portions) * 3
        currow = rowbcs + 1
        for ibc in range(0, len(self.outletbcs)):
            currow += self.outletbcs[ibc].apply_bc_matrix_dot(matrix_dot,
                                                              currow,
                                                              curcol)
            curcol += self.outletbcs[ibc].nvariables

    # rowbcs is the first index of the boundary conditions
    def add_bcs(self, matrix_dot, rowbcs):
        self.inletbc.apply_bc_matrix(matrix_dot, rowbcs)

        curcol = len(self.portions) * 3
        currow = rowbcs + 1
        for ibc in range(0, len(self.outletbcs)):
            currow += self.outletbcs[ibc].apply_bc_matrix(matrix_dot,
                                                          currow,
                                                          curcol)

            curcol += self.outletbcs[ibc].nvariables
