from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import simps


# The constitutive equation is written in this form:
#
# [a, b, c]\dot{[P_in, P_out, Q]^T} = [d, e, f][P_in, P_out, Q]^T

class Model(ABC):

    def __init__(self, portion, problem_data):
        self.vessel_portion = portion
        self.problem_data = problem_data
        self.R = 0.0
        self.C = 0.0

        self.compute_R()
        self.compute_C()

        return

    @abstractmethod
    def compute_R(self):
        pass

    @abstractmethod
    def compute_C(self):
        pass

    # this is the method that computes [a, b, c]
    def get_vector_dot(self):
        return np.array([self.R * self.C, -self.R * self.C, 0.0])

    # this is the method that computes [d, e, f]
    def get_vector(self):
        return np.array([-1.0, 1.0, self.R])


class Resistance(Model):
    def __init__(self, portion, problem_data):
        super().__init__(portion, problem_data)
        return

    def compute_R(self):
        self.vessel_portion.compute_mean_radius()
        self.R = float(128 * self.problem_data.viscosity * self.vessel_portion.arclength[-1] /
                       (np.pi * ((2 * self.vessel_portion.mean_radius) ** 4)))
        return

    def compute_C(self):
        self.C = 0.0
        return


class Windkessel2(Resistance):
    def __init__(self, portion, problem_data):
        super().__init__(portion, problem_data)
        return

    def compute_C(self):
        self.vessel_portion.compute_mean_radius()
        self.C = float(np.pi * ((2 * self.vessel_portion.mean_radius) ** 3) * self.vessel_portion.arclength[-1] /
                       (4 * self.problem_data.E * self.problem_data.thickness_ratio *
                        (2 * self.vessel_portion.mean_radius)))
        return self.C


# The constitutive equation is written in this form:
#
# [a, b, c]\dot{[P_in, P_out, Q]^T} = [d, e, f][P_in, P_out, Q]^T + [g, h, i][P_in^2, P_out^2, Q^2]^T


class ModelStenosis(Model):

    def __init__(self, portion, problem_data, r0):
        self.L = 0.0
        self.R2 = 0.0
        self.r0 = r0

        super().__init__(portion, problem_data)

        self.compute_L()
        self.compute_R2()

        return

    @abstractmethod
    def compute_L(self):
        pass

    @abstractmethod
    def compute_R2(self):
        pass

    # this is the method that computes [a, b, c]
    def get_vector_dot(self):
        return np.array([self.R * self.C, -self.R * self.C, -self.L])

    # this is the method that computes [d, e, f]
    def get_vector(self):
        return np.array([-1.0, 1.0, self.R])

    # this is the method that computes [g, h, i]
    def get_vector_nonlinear(self):
        return np.array([0.0, 0.0, self.R2])

    @abstractmethod
    def evaluate_nonlinear(self, sol, index):
        pass

    @abstractmethod
    def evaluate_jacobian_nonlinear(self, sol, index):
        pass


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

    def evaluate_nonlinear(self, sol, index):
        vec = self.get_vector_nonlinear()
        local_sol = sol[3 * index:3 * (index + 1), 0]
        nl_sol = local_sol * np.abs(local_sol)
        return np.inner(vec, nl_sol)

    def evaluate_jacobian_nonlinear(self, sol, index):
        vec = self.get_vector_nonlinear()
        local_sol = sol[3 * index:3 * (index + 1), 0]
        nl_sol = 2.0 * local_sol * np.sign(local_sol)
        retVec = np.array([vec[i] * nl_sol[i] for i in range(3)])
        return retVec


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

    def evaluate_nonlinear(self, sol, index):
        vec = self.get_vector_nonlinear()
        local_sol = sol[3 * index:3 * (index + 1), 0]
        nl_sol = np.square(local_sol)
        return np.inner(vec, nl_sol)

    def evaluate_jacobian_nonlinear(self, sol, index):
        vec = self.get_vector_nonlinear()
        local_sol = sol[3 * index:3 * (index + 1), 0]
        nl_sol = 2.0 * local_sol
        retVec = np.array([vec[i] * nl_sol[i] for i in range(3)])
        return retVec


class ItuSharma(ModelStenosis):
    # CAVEAT: here I do not take into account the term with the average flow rate over the heartbeat!!
    def __init__(self, portion, problem_data, r0, HR):
        self.HR = HR
        ModelStenosis.__init__(self, portion, problem_data, r0)
        return

    def compute_R(self):
        self.vessel_portion.compute_min_radius()
        omega = self.HR * 2 * np.pi / 60.0
        alpha = self.r0 * np.sqrt(self.problem_data.density * omega / self.problem_data.viscosity)
        Kv = 1 + 0.053 * self.vessel_portion.min_radius / self.r0 * alpha ** 2

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
        self.R2 = float((Kt * self.problem_data.density) /
                        (2 * np.pi ** 2 * self.r0 ** 4) *
                        ((self.r0 / self.vessel_portion.min_radius) ** 2 - 1) ** 2)

        return

    def evaluate_nonlinear(self, sol, index):
        vec = self.get_vector_nonlinear()
        local_sol = sol[3 * index:3 * (index + 1), 0]
        nl_sol = local_sol * np.abs(local_sol)
        return np.inner(vec, nl_sol)

    def evaluate_jacobian_nonlinear(self, sol, index):
        vec = self.get_vector_nonlinear()
        local_sol = sol[3 * index:3 * (index + 1), 0]
        nl_sol = 2.0 * local_sol * np.sign(local_sol)
        retVec = np.array([vec[i] * nl_sol[i] for i in range(3)])
        return retVec


class ResistanceStenosis(ModelStenosis, Resistance):
    def __init__(self, portion, problem_data, r0):
        ModelStenosis.__init__(self, portion, problem_data, r0)
        Resistance.__init__(self, portion, problem_data)
        return

    def compute_L(self):
        self.L = 0.0
        return

    def compute_R2(self):
        self.vessel_portion.compute_mean_radius()
        Kt = 1.5
        self.R2 = float((Kt * self.problem_data.density) /
                        (2 * np.pi ** 2 * self.r0 ** 4) *
                        ((self.r0 / self.vessel_portion.mean_radius) ** 2 - 1) ** 2)

        return self.R2

    def evaluate_nonlinear(self, sol, index):
        vec = self.get_vector_nonlinear()
        local_sol = sol[3 * index:3 * (index + 1), 0]
        nl_sol = np.square(local_sol)
        return np.inner(vec, nl_sol)

    def evaluate_jacobian_nonlinear(self, sol, index):
        vec = self.get_vector_nonlinear()
        local_sol = sol[3 * index:3 * (index + 1), 0]
        nl_sol = 2.0 * local_sol
        retVec = np.array([vec[i] * nl_sol[i] for i in range(3)])
        return retVec


class Windkessel2Stenosis(ResistanceStenosis, Windkessel2):
    def __init__(self, portion, problem_data, r0):
        ResistanceStenosis.__init__(self, portion, problem_data, r0)
        Windkessel2.__init__(self, portion, problem_data)
        return

    def compute_C(self):
        Windkessel2.compute_C(self)
        return

    def get_vector_dot(self):
        return ResistanceStenosis.get_vector_dot(self)

    def get_vector(self):
        return ResistanceStenosis.get_vector(self)

    def evaluate_nonlinear(self, sol, index):
        vec = self.get_vector_nonlinear()
        local_sol = sol[3 * index:3 * (index + 1), 0]
        nl_sol = np.square(local_sol)
        return np.inner(vec, nl_sol)

    def evaluate_jacobian_nonlinear(self, sol, index):
        vec = self.get_vector_nonlinear()
        local_sol = sol[3 * index:3 * (index + 1), 0]
        nl_sol = 2.0 * local_sol
        retVec = np.array([vec[i] * nl_sol[i] for i in range(3)])
        return retVec
