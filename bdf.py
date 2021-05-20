from ode_system import ODESystem
from numpy import linalg
import numpy as np
import matplotlib.pylab as plt
from problem_data import ProblemData

class BDF1:
    def __init__(self, ode_system, connectivity, problem_data, bc_manager):
        self.ode_system = ode_system
        self.connectivity = connectivity
        self.use_inlet_pressure = problem_data.use_inlet_pressure
        self.deltat = problem_data.deltat
        self.t0 = problem_data.t0
        self.T = problem_data.T
        self.t0ramp = problem_data.t0ramp
        self.bc_manager = bc_manager
        self.setup_system()

    def run(self):
        t = self.t0ramp
        times = [t]
        syssize = self.bdfmatrix.shape[0]
        sols = [np.zeros([syssize,1])]
        bcvec = np.zeros([syssize,1])
        while t < self.T:
            t = t + self.deltat

            print('Solving t = ' + str(t))

            # assemble rhs
            rhs = self.matrix_dot.dot(sols[-1])
            self.bc_manager.apply_bc_vector(bcvec, t)

            rhs += self.deltat * bcvec

            u = np.linalg.solve(self.bdfmatrix, rhs)
            sols.append(u)
            times.append(t)

        return np.array(sols).squeeze().T, np.array(times)


    def setup_system(self):
        self.matrix_dot = self.ode_system.smatrix_dot
        self.matrix = self.ode_system.smatrix

        self.bdfmatrix = (self.matrix_dot - self.deltat * self.matrix)

        print(np.linalg.cond(self.bdfmatrix))
        print(np.where(~self.bdfmatrix.any(axis=1)))
        plt.spy(self.bdfmatrix)
        # plt.show()
