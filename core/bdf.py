import numpy as np
from abc import abstractmethod

from solver import SystemSolver


class BDF:
    def __init__(self, ode_system, connectivity, problem_data, solver_data, bc_manager):
        self.ode_system = ode_system
        self.connectivity = connectivity
        self.use_inlet_pressure = problem_data.use_inlet_pressure
        self.deltat = problem_data.deltat
        self.t0 = problem_data.t0
        self.T = problem_data.T
        self.t0ramp = problem_data.t0ramp
        self.bc_manager = bc_manager
        self.setup_system()
        self.solver = SystemSolver(tol=solver_data.tol,
                                   min_err=solver_data.min_err,
                                   max_iter=solver_data.max_iter)
        self.solver_strategy = solver_data.strategy

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

    @abstractmethod
    def extrapolated_solution(self, solutions):
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

            rhs = self.matrix_dot.dot(self.prev_solutions_contribution(prev_solutions))
            self.bc_manager.apply_bc_vector(bcvec, t)
            rhs += self.beta() * self.deltat * bcvec

            if self.ode_system.is_linear:
                u = self.solver.solve_linear(self.bdfmatrix, rhs)

            else:
                if self.solver_strategy == "implicit":

                    # initial_guess = self.solver.solve_linear(self.bdfmatrix, rhs)
                    initial_guess = self.extrapolated_solution(prev_solutions)
                    initial_guess[self.bc_manager.inletindex*3] = self.bc_manager.inletbc.inlet_function(t)

                    def fun(sol):
                        retVec = self.bdfmatrix.dot(sol)
                        retVec -= self.beta() * self.deltat * self.ode_system.evaluate_nonlinear(sol)
                        retVec -= rhs
                        self.bc_manager.apply_inlet0bc_vector(retVec, t)
                        return retVec

                    def jac(sol):
                        retMat = self.bdfmatrix
                        # retMat -= self.beta() * self.deltat * self.ode_system.evaluate_jacobian_nonlinear(sol)
                        return retMat

                    u = self.solver.solve_nonlinear(fun, jac, initial_guess)

                elif self.solver_strategy == "semi-implicit":
                    guess = self.extrapolated_solution(prev_solutions)
                    # guess = self.solver.solve_linear(self.bdfmatrix, rhs)

                    nl_rhs = rhs + self.beta() * self.deltat * self.ode_system.evaluate_nonlinear(guess)

                    u = self.solver.solve_linear(self.bdfmatrix, nl_rhs)

                else:
                    raise ValueError(f"Unrecognized solver strategy {self.solver_strategy}!")

            if np.isnan(np.linalg.norm(u)):
                raise ValueError(f"The solution at time {t} features NaNs!")

            sols.append(u)
            prev_solutions.append(u)
            prev_solutions.pop(0)
            times.append(t)

        return np.array(sols).squeeze().T, np.array(times)

    def setup_system(self):
        self.matrix_dot = self.ode_system.smatrix_dot
        self.matrix = self.ode_system.smatrix

        self.bdfmatrix = self.matrix_dot - self.deltat * self.beta() * self.matrix

        return


class BDF1(BDF):
    def __init__(self, ode_system, connectivity, problem_data, solver_data, bc_manager):
        super().__init__(ode_system, connectivity, problem_data, solver_data, bc_manager)
        return

    def order(self):
        return 1

    def beta(self):
        return 1

    def prev_solutions_contribution(self, solutions):
        return solutions[-1]

    def extrapolated_solution(self, solutions):
        return solutions[-1]


class BDF2(BDF):
    def __init__(self, ode_system, connectivity, problem_data, solver_data, bc_manager):
        super().__init__(ode_system, connectivity, problem_data, solver_data, bc_manager)
        return

    def order(self):
        return 2

    def beta(self):
        return 2 / 3

    def prev_solutions_contribution(self, solutions):
        return solutions[-1] * 4 / 3 - solutions[-2] * 1 / 3

    def extrapolated_solution(self, solutions):
        return solutions[-1] * 2.0 - solutions[-2] * 1.0


class BDF3(BDF):
    def __init__(self, ode_system, connectivity, problem_data, solver_data, bc_manager):
        super().__init__(ode_system, connectivity, problem_data, solver_data, bc_manager)
        return

    def order(self):
        return 3

    def beta(self):
        return 6 / 11

    def prev_solutions_contribution(self, solutions):
        return solutions[-1] * 18 / 11 - solutions[-2] * 9 / 11 + solutions[-3] * 2 / 11

    def extrapolated_solution(self, solutions):
        return solutions[-1] * 3.0 - solutions[-2] * 3.0 + solutions[-3] * 1.0

