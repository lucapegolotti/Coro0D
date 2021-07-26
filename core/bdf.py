import numpy as np
from abc import abstractmethod

from solver import SystemSolver


class BDF:
    def __init__(self, ode_system, connectivity, problem_data, solver_data, bc_manager):
        self.name = "BDF"
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

    def run(self, times=None, old_solutions=None):

        t = self.t0ramp if times is None else times[-1]
        syssize = self.lhsmatrix.shape[0]

        times = [t] if times is None else times.tolist()
        prev_solutions = [np.zeros([syssize, 1]) for _ in range(self.order())] if old_solutions is None \
                         else [np.array([elem]).T for elem in old_solutions.T.tolist()]

        sols = [np.zeros([syssize, 1])] if old_solutions is None else prev_solutions[:]
        bcvec = np.zeros([syssize, 1])

        while t < self.T:
            t += self.deltat

            print(f"{self.name}: solving t = {t:.5f} s")

            rhs = self.matrix_dot.dot(self.prev_solutions_contribution(prev_solutions))
            rhs += self.beta() * self.deltat * self.ode_system.evaluate_constant_term()
            self.bc_manager.apply_bc_vector(bcvec, t)
            rhs += self.beta() * self.deltat * bcvec

            if self.ode_system.is_linear:
                u = self.solver.solve_linear(self.lhsmatrix, rhs)

            else:
                if self.solver_strategy == "implicit":

                    # initial_guess = self.solver.solve_linear(self.lhsmatrix, rhs)
                    initial_guess = self.extrapolated_solution(prev_solutions)
                    initial_guess[self.bc_manager.inletindex * 3] = self.bc_manager.inletbc.inlet_function(t)

                    def fun(sol):
                        retVec = self.lhsmatrix.dot(sol)
                        retVec -= self.beta() * self.deltat * self.ode_system.evaluate_nonlinear(sol)
                        retVec -= rhs
                        self.bc_manager.apply_inlet0bc_vector(retVec, t)
                        return retVec

                    def jac(sol):
                        retMat = self.lhsmatrix
                        # retMat += self.beta() * self.deltat * self.ode_system.evaluate_jacobian_nonlinear(sol)
                        return retMat

                    u = self.solver.solve_nonlinear(fun, jac, initial_guess)

                elif self.solver_strategy == "semi-implicit":
                    guess = self.extrapolated_solution(prev_solutions)
                    # guess = self.solver.solve_linear(self.lhsmatrix, rhs)

                    nl_rhs = rhs + self.beta() * self.deltat * self.ode_system.evaluate_nonlinear(guess)

                    u = self.solver.solve_linear(self.lhsmatrix, nl_rhs)

                else:
                    raise ValueError(f"Unrecognized solver strategy {self.solver_strategy}!")

            if np.isnan(np.linalg.norm(u)):
                raise ValueError(f"The solution at time {t} features NaNs!")

            sols.append(u.copy())
            prev_solutions.append(u)
            prev_solutions.pop(0)
            times.append(t)

        return np.array(sols).squeeze().T, np.array(times)

    def setup_system(self):
        self.matrix_dot = self.ode_system.smatrix_dot
        self.matrix = self.ode_system.smatrix

        self.lhsmatrix = self.matrix_dot - self.deltat * self.beta() * self.matrix

        return


class BDF1(BDF):
    def __init__(self, ode_system, connectivity, problem_data, solver_data, bc_manager):
        super().__init__(ode_system, connectivity, problem_data, solver_data, bc_manager)
        self.name = "BDF1"
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
        self.name = "BDF2"

        self.CN = CN(ode_system, connectivity, problem_data, solver_data, bc_manager)
        self.CN.set_T(self.t0ramp + self.deltat)

        return

    def order(self):
        return 2

    def beta(self):
        return 2 / 3

    def prev_solutions_contribution(self, solutions):
        return solutions[-1] * 4 / 3 - solutions[-2] * 1 / 3

    def extrapolated_solution(self, solutions):
        return solutions[-1] * 2.0 - solutions[-2] * 1.0

    # def run(self, times=None, old_solutions=None):
    #     solutions, times = self.CN.run()
    #     solutions, times = super().run(times, solutions)
    #
    #     return solutions, times


class CN:
    def __init__(self, ode_system, connectivity, problem_data, solver_data, bc_manager):
        self.name = "Crank-Nicholson"
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

        return

    def order(self):
        return 2

    def setup_system(self):
        self.matrix_dot = self.ode_system.smatrix_dot
        self.matrix = self.ode_system.smatrix

        self.lhsmatrix = self.matrix_dot - (self.deltat / 2.0) * self.matrix
        self.rhsmatrix = self.matrix_dot + (self.deltat / 2.0) * self.matrix

        return

    def set_T(self, T):
        self.T = T
        return

    def run(self):

        t = self.t0ramp
        times = [t]
        syssize = self.lhsmatrix.shape[0]
        prev_solution = np.zeros([syssize, 1])
        sols = [np.zeros([syssize, 1])]
        bcvec = np.zeros([syssize, 1])

        while t < self.T:
            t += self.deltat

            print(f"{self.name}: solving t = {t:.5f} s")

            rhs = self.rhsmatrix.dot(prev_solution)
            rhs += self.deltat * self.ode_system.evaluate_constant_term()
            self.bc_manager.apply_bc_vector(bcvec, t)
            rhs += self.deltat * bcvec

            if self.ode_system.is_linear:
                u = self.solver.solve_linear(self.lhsmatrix, rhs)

            else:
                # initial_guess = self.solver.solve_linear(self.lhsmatrix, rhs)
                initial_guess = prev_solution[:]
                initial_guess[self.bc_manager.inletindex * 3] = self.bc_manager.inletbc.inlet_function(t)

                def fun(sol):
                    retVec = self.lhsmatrix.dot(sol)
                    retVec -= (self.deltat / 2.0) * (self.ode_system.evaluate_nonlinear(sol) +
                                                     self.ode_system.evaluate_nonlinear(prev_solution))
                    retVec -= rhs
                    self.bc_manager.apply_inlet0bc_vector(retVec, t)
                    return retVec

                def jac(sol):
                    retMat = self.lhsmatrix
                    # retMat -= (self.deltat / 2.0) * self.ode_system.evaluate_jacobian_nonlinear(sol)
                    return retMat

                u = self.solver.solve_nonlinear(fun, jac, initial_guess)

            if np.isnan(np.linalg.norm(u)):
                raise ValueError(f"The solution at time {t} features NaNs!")

            sols.append(u.copy())
            prev_solution = u
            times.append(t)

        return np.array(sols).squeeze().T, np.array(times)
