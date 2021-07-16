import numpy as np


class SystemSolver:
    def __init__(self, tol=1e-6, min_err=1e-14, max_iter=100):

        self.tol = tol
        self.min_err = min_err
        self.max_iter = max_iter

        return

    def solve_linear(self, matrix, vector):
        return np.linalg.solve(matrix, vector)

    def solve_nonlinear(self, fun, jac, initial_guess):
        sol = initial_guess
        cnt = 0

        curFun = fun(sol)
        curJac = jac(sol)
        err = np.linalg.norm(curFun)
        err0 = err

        while (err / err0 > self.tol and err > self.min_err and not np.isnan(err)) and cnt < self.max_iter:
            incr = -np.linalg.solve(curJac, curFun)
            sol += incr

            print(f"Newton's method. Iteration: {cnt+1}  -  Relative Error: {err/err0}  -  Absolute Error: {err}")

            curFun = fun(sol)
            curJac = jac(sol)
            err = np.linalg.norm(curFun)
            cnt += 1

        if (cnt == self.max_iter and (err / err0 > self.tol or err > self.min_err or np.isnan(err))) or np.isnan(err):
            raise ValueError(f"Newton's method has failed after {cnt} iterations!")
        else:
            print(f"Newton's method converged after {cnt} iterations! -  Relative Error: {err/err0}  -  Absolute Error: {err}")

        return sol

