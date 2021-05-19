from ode_system import ODESystem
from numpy import linalg
import numpy as np
import matplotlib.pylab as plt
from problem_data import ProblemData


class BDF1:
    def __init__(self, ode_system, connectivity, problem_data):
        self.ode_system = ode_system
        self.connectivity = connectivity
        self.use_inlet_pressure = problem_data.use_inlet_pressure
        self.deltat = problem_data.deltat
        self.t0 = problem_data.t0
        self.setup_system()

    def run(self):
        t = self.t0


    def setup_system(self):
        self.matrix_dot = self.ode_system.get_system_matrix_dot()
        self.matrix = self.ode_system.get_system_matrix()

        self.bdfmatrix = (self.matrix_dot - self.deltat * self.matrix)

        # apply bcs to matrix
        # find first row with all zeros
        self.inletbcrow = np.where(~self.matrix.any(axis=1))[0][0]

        # find index inlet block
        indexinletblock = np.where(self.connectivity == 2)
        if self.use_inlet_pressure:
            # in this case we fix the inlet pressure
            self.bdfmatrix[self.inletbcrow,indexinletblock[1] * 3 + 0] = 1
        else:
            # in this case we fix the inlet flowrate
            self.bdfmatrix[self.inletbcrow,indexinletblock[1] * 3 + 2] = 1

        # find max outlet flag
        maxoutletflag = int(np.max(self.connectivity))

        self.outletbcrows = []
        for flag in range(3, maxoutletflag + 1):
            currow = self.inletbcrow + flag - 2
            indexoutletblock = np.where(self.connectivity == flag)
            self.bdfmatrix[currow,indexoutletblock[1] * 3 + 1] = 1
            self.outletbcrows.append(currow)

        # print(np.linalg.cond(self.bdfmatrix))
        # print(np.where(~self.bdfmatrix.any(axis=1)))
        plt.spy(self.bdfmatrix)
        # plt.show()
