from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import simps


# The constitutive equation is written in this form:
#
# A * [P_in, P_out, Q_in, Q_out]^T = B * [P_in, P_out, Q_in, Q_out]^T + K

class Model(ABC):

    def __init__(self, portion, problem_data):
        self.vessel_portion = portion
        self.problem_data = problem_data
        self.R = 0.0
        self.C = 0.0
        self.L = 0.0

        self.compute_R()
        self.compute_C()
        self.compute_L()

        return

    @abstractmethod
    def compute_R(self):
        pass

    @abstractmethod
    def compute_C(self):
        pass

    @abstractmethod
    def compute_L(self):
        pass

    # this is the method that computes A
    def get_matrix_dot(self):
        return np.array([[0.0, 0.0, self.L, 0.0],
                         [0.0, self.C, 0.0, 0.0]])

    # this is the method that computes B
    def get_matrix(self):
        return np.array([[1.0, -1.0, -self.R, 0.0],
                         [0.0, 0.0, 1.0, -1.0]])

    # this is the method that computes K
    def get_constant(self):
        return np.array([[0.0, 0.0]]).T


class R_model(Model):
    def __init__(self, portion, problem_data):
        super().__init__(portion, problem_data)
        return

    def compute_R(self):
        posindices = np.where(self.vessel_portion.radii > 0)
        posradii = self.vessel_portion.radii[posindices]
        posarclength = self.vessel_portion.arclength[posindices]
        posarclength = np.subtract(posarclength, posarclength[0])

        integral = simps(posradii ** (-4), posarclength)
        self.R = 8 * self.problem_data.viscosity * integral / np.pi

        return

    def compute_C(self):
        self.C = 0.0
        return

    def compute_L(self):
        self.L = 0.0
        return


class RC_model(R_model):
    def __init__(self, portion, problem_data):
        super().__init__(portion, problem_data)
        return

    def compute_C(self):
        self.vessel_portion.compute_mean_radius()

        posindices = np.where(self.vessel_portion.radii > 0)
        posradii = self.vessel_portion.radii[posindices]
        posarclength = self.vessel_portion.arclength[posindices]
        posarclength = np.subtract(posarclength, posarclength[0])
        integral = simps(posradii ** 3, posarclength)

        self.C = float(np.pi * integral /
                       (self.problem_data.E * self.problem_data.thickness_ratio * self.vessel_portion.mean_radius))
        return self.C


class RL_model(R_model):
    def __init__(self, portion, problem_data):
        super().__init__(portion, problem_data)
        return

    def compute_L(self):
        posindices = np.where(self.vessel_portion.radii > 0)
        posradii = self.vessel_portion.radii[posindices]
        posarclength = self.vessel_portion.arclength[posindices]
        posarclength = np.subtract(posarclength, posarclength[0])

        integral = simps(posradii ** (-2), posarclength)
        self.L = (self.problem_data.density / np.pi) * integral

        return


class RLC_model(RC_model, RL_model):
    def __init__(self, portions, problem_data):
        RC_model.__init__(self, portions, problem_data)
        RL_model.__init__(self, portions, problem_data)
        return

    def compute_R(self):
        return RC_model.compute_R(self)

    def compute_C(self):
        return RC_model.compute_C(self)

    def compute_L(self):
        return RL_model.compute_L(self)


class RCL_model(RC_model, RL_model):
    def __init__(self, portions, problem_data):
        RC_model.__init__(self, portions, problem_data)
        RL_model.__init__(self, portions, problem_data)
        return

    def compute_R(self):
        return RC_model.compute_R(self)

    def compute_C(self):
        return RC_model.compute_C(self)

    def compute_L(self):
        return RL_model.compute_L(self)

    def get_matrix_dot(self):
        return np.array([[0.0, 0.0, 0.0, self.L],
                         [self.C, 0.0, -self.R * self.C, 0.0]])


class Windkessel2(RC_model):
    def __init__(self, portions, problem_data):
        super().__init__(portions, problem_data)
        return

    def get_matrix_dot(self):
        return np.array([[-self.C * self.R, self.C * self.R, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0]])

# The constitutive equation is written in this form:
#
# A * [P_in, P_out, Q_in, Q_out]^T = B * [P_in, P_out, Q_in, Q_out]^T + C * f([P_in, P_out, Q_in, Q_out])^T + K


class ModelStenosis(Model):

    def __init__(self, portion, problem_data, r0):
        self.R2 = 0.0
        self.r0 = r0
        self.S = 1.0

        super().__init__(portion, problem_data)

        self.compute_S()
        self.compute_R2()

        return

    def compute_S(self):
        self.vessel_portion.compute_min_radius()

        A0 = np.pi * self.r0**2
        A = np.pi * self.vessel_portion.min_radius**2

        self.S = 1 - A/A0

        return

    @abstractmethod
    def compute_R2(self):
        pass

    @abstractmethod
    def nonlinear_function(self, sol):
        pass

    @abstractmethod
    def nonlinear_function_der(self, sol):
        pass

    # this is the method that computes C
    def get_matrix_nonlinear(self):
        return np.array([[0.0, 0.0, -self.R2, 0.0],
                         [0.0, 0.0, 0.0, 0.0]])

    def evaluate_nonlinear(self, sol, index):
        mat = self.get_matrix_nonlinear()
        local_sol = sol[4 * index:4 * (index + 1), 0]
        nl_sol = self.nonlinear_function(local_sol)
        retVec = np.dot(mat, nl_sol)
        retVec = np.reshape(retVec, (2, 1))
        return retVec

    def evaluate_jacobian_nonlinear(self, sol, index):
        mat = self.get_matrix_nonlinear()
        local_sol = sol[4 * index:4 * (index + 1), 0]
        nl_sol_der = self.nonlinear_function_der(local_sol)
        retMat = np.array([[mat[0, i] * nl_sol_der[i] for i in range(4)],
                           [mat[1, i] * nl_sol_der[i] for i in range(4)]])
        return retMat


class YoungTsai(ModelStenosis):
    def __init__(self, portion, problem_data, r0):
        ModelStenosis.__init__(self, portion, problem_data, r0)
        return

    def compute_R(self):
        self.vessel_portion.compute_min_radius()
        La = 0.83 * self.vessel_portion.arclength[-1] + 3.28 * self.vessel_portion.min_radius
        Kv = 16 * La / self.r0 * (self.r0 / self.vessel_portion.min_radius) ** 4
        self.R = float((self.problem_data.viscosity * Kv) /
                       (2 * np.pi * self.r0 ** 3))

        return

    def compute_C(self):
        self.C = 0.0
        return

    def compute_L(self):
        Ku = 1.2
        self.L = float((Ku * self.problem_data.density * self.vessel_portion.arclength[-1]) /
                       (np.pi * self.r0 ** 2))

        return

    def compute_R2(self):
        self.vessel_portion.compute_min_radius()
        Kt = 1.52
        self.R2 = float((Kt * self.problem_data.density) /
                        (2 * np.pi ** 2 * self.r0 ** 4) *
                        ((self.r0 / self.vessel_portion.min_radius) ** 2 - 1) ** 2)

        return

    def nonlinear_function(self, sol):
        return sol * np.abs(sol)

    def nonlinear_function_der(self, sol):
        return 2.0 * sol * np.sign(sol)


class Garcia(ModelStenosis):
    def __init__(self, portion, problem_data, r0):
        ModelStenosis.__init__(self, portion, problem_data, r0)
        return

    def compute_R(self):
        self.R = 0.0
        return

    def compute_C(self):
        self.C = 0.0
        return

    def compute_L(self):
        self.vessel_portion.compute_min_radius()
        alpha = 6.28
        beta = 0.5
        self.L = float((alpha * self.problem_data.density) /
                       (np.sqrt(np.pi) * self.r0) *
                       ((self.r0 / self.vessel_portion.min_radius) ** 2 - 1) ** beta)

        return

    def compute_R2(self):
        self.vessel_portion.compute_min_radius()
        self.R2 = float((self.problem_data.density / 2) *
                        ((self.r0 ** 2 - self.vessel_portion.min_radius ** 2) /
                         (np.pi * self.r0 ** 2 * self.vessel_portion.min_radius ** 2)) ** 2)

        return

    def nonlinear_function(self, sol):
        return np.square(sol)

    def nonlinear_function_der(self, sol):
        return 2.0 * sol


class ItuSharma(ModelStenosis):
    # CAVEAT: the flow rate from a steady simulation (if provided) is used to compute the continuous term
    def __init__(self, portion, problem_data, r0, HR, sol_steady=None):
        self.HR = HR
        self.sol_steady = sol_steady
        ModelStenosis.__init__(self, portion, problem_data, r0)
        return

    def compute_R(self):
        self.vessel_portion.compute_min_radius()
        omega = self.HR * 2 * np.pi / 60.0
        alpha = self.r0 * np.sqrt(self.problem_data.density * omega / self.problem_data.viscosity)
        Kv = 1 + 0.053 * (self.vessel_portion.min_radius / self.r0)**2 * alpha ** 2

        posindices = np.where(self.vessel_portion.radii > 0)
        posradii = self.vessel_portion.radii[posindices]
        posarclength = self.vessel_portion.arclength[posindices]
        posarclength = np.subtract(posarclength, posarclength[0])
        # we compute the mean radius using the integral over the arclength
        integral = simps(posradii ** (-4), posarclength)
        Rvc = (8 * self.problem_data.viscosity / np.pi) * integral

        self.R = Kv * Rvc

        return

    def compute_C(self):
        self.C = 0.0
        return

    def compute_L(self):
        Ku = 1.2

        posindices = np.where(self.vessel_portion.radii > 0)
        posradii = self.vessel_portion.radii[posindices]
        posarclength = self.vessel_portion.arclength[posindices]
        posarclength = np.subtract(posarclength, posarclength[0])
        # we compute the mean radius using the integral over the arclength
        integral = simps(posradii ** (-2), posarclength)
        Lu = (self.problem_data.density / np.pi) * integral

        self.L = Ku * Lu

        return

    def compute_R2(self):
        self.vessel_portion.compute_min_radius()
        Kt = 1.52
        # Kt = 27.3 * 1e-16 * np.exp(34 * self.S) + 0.26 * np.exp(1.51 * self.S)
        self.R2 = float((Kt * self.problem_data.density) /
                        (2 * np.pi ** 2 * self.r0 ** 4) *
                        ((self.r0 / self.vessel_portion.min_radius) ** 2 - 1) ** 2)

        return

    def get_constant(self):
        if self.sol_steady is None:
            return 0.0

        omega = self.HR * 2 * np.pi / 60.0
        alpha = self.r0 * np.sqrt(self.problem_data.density * omega / self.problem_data.viscosity)
        Kc = 0.0018 * alpha**2

        posindices = np.where(self.vessel_portion.radii > 0)
        posradii = self.vessel_portion.radii[posindices]
        posarclength = self.vessel_portion.arclength[posindices]
        posarclength = np.subtract(posarclength, posarclength[0])

        integral = simps(posradii ** (-4), posarclength)
        Rvc = (8 * self.problem_data.viscosity / np.pi) * integral

        # Rtot = self.MAP / self.CO
        # Rpart = self.vessel_portion.downstream_outlet_resistance
        # Rcoeff = Rtot / Rpart
        # K = Kc * Rvc * Rcoeff * self.CO

        K = Kc * Rvc * self.sol_steady[2]  # the steady inflow is used as mean flow!
        retVec = np.array([[-K, 0.0]]).T

        return retVec

    def nonlinear_function(self, sol):
        return sol * np.abs(sol)

    def nonlinear_function_der(self, sol):
        return 2.0 * sol * np.sign(sol)


class ResistanceStenosis(ModelStenosis, R_model):
    def __init__(self, portion, problem_data, r0):
        ModelStenosis.__init__(self, portion, problem_data, r0)
        R_model.__init__(self, portion, problem_data)
        return

    def compute_L(self):
        self.L = 0.0
        return

    def compute_R2(self):
        self.vessel_portion.compute_min_radius()
        Kt = 1.52
        self.R2 = float((Kt * self.problem_data.density) /
                        (2 * np.pi ** 2 * self.r0 ** 4) *
                        ((self.r0 / self.vessel_portion.min_radius) ** 2 - 1) ** 2)

        return self.R2

    def nonlinear_function(self, sol):
        return np.square(sol)

    def nonlinear_function_der(self, sol):
        return 2.0 * sol
