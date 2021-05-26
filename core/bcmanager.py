from inletbc import InletBC
from outletbc import OutletBC
from distal_pressure_generator import DistalPressureGenerator
import numpy as np

class BCManager:
    def __init__(self, portions, connectivity, inletbc_type, outletbc_type, folder,
                 problem_data, coronary, distal_pressure_coeff = 1, distal_pressure_shift = 0.0):
        self.portions = portions
        self.connectivity = connectivity
        self.inletbc_type = inletbc_type
        self.outletbc_type = outletbc_type
        self.folder = folder
        self.problem_data = problem_data
        self.coronary = coronary
        self.distal_pressure_coeff = distal_pressure_coeff
        self.distal_pressure_shift = distal_pressure_shift
        self.create_bcs()

    # we set the row where the boundary conditions start in matrices and vectors
    def set_starting_row_bcs(self, row):
        self.starting_row = row

    def create_bcs(self):
        # get index of inlet block
        self.inletindex = int(np.where(self.connectivity == 2)[1])
        self.inletbc = InletBC(self.portions[self.inletindex],
                               self.inletindex, self.inletbc_type,
                               self.folder,
                               self.problem_data)

        self.distal_pressure_generator = DistalPressureGenerator(self.inletbc.times,
                                                                 self.inletbc.indices_minpressures,
                                                                 self.folder,
                                                                 self.problem_data,
                                                                 self.coronary,
                                                                 self.distal_pressure_coeff,
                                                                 self.distal_pressure_shift)

        # find max outlet flag
        maxoutletflag = int(np.max(self.connectivity))
        self.outletindices = []
        self.outletbcs = []
        for flag in range(3, maxoutletflag + 1):
            self.outletindices.append(int(np.where(self.connectivity == flag)[1]))
            self.outletbcs.append(OutletBC(self.portions[self.outletindices[-1]],
                                  self.outletindices[-1], self.outletbc_type,
                                  self.distal_pressure_generator))
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
        self.inletbc.apply_bc_vector(vector, time, self.starting_row)

        currow = self.starting_row + 1
        for ibc in range(0, len(self.outletbcs)):
            currow += self.outletbcs[ibc].apply_bc_vector(vector, time, currow)
