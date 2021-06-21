from numpy import linalg
import numpy as np
from abc import abstractmethod


class BDF:
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

        return

    @abstractmethod
    def order(self):
        pass

    @abstractmethod
    def beta(self):
        pass

    @abstractmethod
    def prev_solutions_contribution(self, solutions):
        pass

    def run(self):
        t = self.t0ramp
        times = [t]
        syssize = self.bdfmatrix.shape[0]
        prev_solutions = [np.zeros([syssize, 1]) for _ in range(self.order())]
        sols = [np.zeros([syssize, 1])]
        bcvec = np.zeros([syssize, 1])
        while t < self.T:
            t += self.deltat

            print('Solving t = ' + "{:.2f}".format(t) + " s")

            # assemble rhs
            rhs = self.matrix_dot.dot(self.prev_solutions_contribution(prev_solutions))
            self.bc_manager.apply_bc_vector(bcvec, t)
            rhs += self.beta() * self.deltat * bcvec

            u = np.linalg.solve(self.bdfmatrix, rhs)

            sols.append(u)
            prev_solutions.append(u)
            prev_solutions.pop(0)
            times.append(t)

        return np.array(sols).squeeze().T, np.array(times)

    def setup_system(self):
        self.matrix_dot = self.ode_system.smatrix_dot
        self.matrix = self.ode_system.smatrix

        self.bdfmatrix = (self.matrix_dot - self.deltat * self.beta() * self.matrix)

        # print(np.linalg.cond(self.bdfmatrix))
        # print(np.where(~self.bdfmatrix.any(axis=1)))
        # plt.spy(self.bdfmatrix)
        # plt.show()

        return


class BDF1(BDF):
    def __init__(self, ode_system, connectivity, problem_data, bc_manager):
        super().__init__(ode_system, connectivity, problem_data, bc_manager)
        return

    def order(self):
        return 1

    def beta(self):
        return 1

    def prev_solutions_contribution(self, solutions):
        return solutions[-1]


class BDF2(BDF):
    def __init__(self, ode_system, connectivity, problem_data, bc_manager):
        super().__init__(ode_system, connectivity, problem_data, bc_manager)
        return

    def order(self):
        return 2

    def beta(self):
        return 2 / 3

    def prev_solutions_contribution(self, solutions):
        return solutions[-1] * 4 / 3 - solutions[-2] * 1 / 3
