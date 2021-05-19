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
        self.T = problem_data.T
        self.setup_system()

    def run(self):
        t = self.t0
        syssize = bdfmatrix.shape[0]
        u = np.zeros([syssize,1])
        sols = [u]
        while t < self.T:
            print('Solving t = ' + str(t))




    def setup_system(self):
        self.matrix_dot = self.ode_system.smatrix_dot
        self.matrix = self.ode_system.smatrix

        self.bdfmatrix = (self.matrix_dot - self.deltat * self.matrix)

        print(np.linalg.cond(self.bdfmatrix))
        print(np.where(~self.bdfmatrix.any(axis=1)))
        plt.spy(self.bdfmatrix)
        # plt.show()
