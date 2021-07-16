from abc import ABC, abstractmethod
import numpy as np


# The constitutive equation is written in this form:
#
# [a, b, c]\dot{[P_in, P_out, Q]^T} = [d, e, f][P_in, P_out, Q]^T

class Model(ABC):
    # this is the method that computes [a, b, c]
    @abstractmethod
    def get_vector_dot(self):
        pass

    # this is the method that computes [d, e, f]
    @abstractmethod
    def get_vector(self):
        pass


class Windkessel2(Model):
    def __init__(self, R, C):
        self.R = R
        self.C = C
        return

    def get_vector_dot(self):
        return np.array([self.R*self.C, -self.R*self.C, 0.0])

    def get_vector(self):
        return np.array([-1.0, 1.0, self.R])


class Resistance(Model):
    def __init__(self, R):
        self.R = R
        return

    def get_vector_dot(self):
        return np.array([0.0, 0.0, 0.0])

    def get_vector(self):
        return np.array([-1.0, 1.0, self.R])


# The constitutive equation is written in this form:
#
# [a, b, c]\dot{[P_in, P_out, Q]^T} = [d, e, f][P_in, P_out, Q]^T + [g, h, i][P_in^2, P_out^2, Q^2]^T


class ModelStenosis(Model):

    # this is the method that computes [g, h, i]
    @abstractmethod
    def get_vector_nonlinear(self):
        pass

    @abstractmethod
    def evaluate_nonlinear(self, sol, index):
        pass

    @abstractmethod
    def evaluate_jacobian_nonlinear(self, sol, index):
        pass


class YoungTsai(ModelStenosis):
    def __init__(self, R, R2, L):
        self.R = R
        self.R2 = R2
        self.L = L
        return

    def get_vector_dot(self):
        return np.array([0.0, 0.0, self.L])

    def get_vector(self):
        return np.array([-1.0, 1.0, self.R])

    def get_vector_nonlinear(self):
        return np.array([0.0, 0.0, self.R2])

    def evaluate_nonlinear(self, sol, index):
        vec = self.get_vector_nonlinear()
        local_sol = sol[3*index:3*(index+1), 0]
        nl_sol = local_sol * np.abs(local_sol)
        return np.inner(vec, nl_sol)

    def evaluate_jacobian_nonlinear(self, sol, index):
        vec = self.get_vector_nonlinear()
        local_sol = sol[3*index:3*(index+1), 0]
        nl_sol = 2.0 * local_sol * np.sign(local_sol)
        retVec = np.array([vec[i] * nl_sol[i] for i in range(3)])
        return retVec


class Garcia(ModelStenosis):
    def __init__(self, R2, L):
        self.R2 = R2
        self.L = L
        return

    def get_vector_dot(self):
        return np.array([0.0, 0.0, self.L])

    def get_vector(self):
        return np.array([-1.0, 1.0, 0.0])

    def get_vector_nonlinear(self):
        return np.array([0.0, 0.0, self.R2])

    def evaluate_nonlinear(self, sol, index):
        vec = self.get_vector_nonlinear()
        local_sol = sol[3*index:3*(index+1), 0]
        nl_sol = np.square(local_sol)
        return np.inner(vec, nl_sol)

    def evaluate_jacobian_nonlinear(self, sol, index):
        vec = self.get_vector_nonlinear()
        local_sol = sol[3*index:3*(index+1), 0]
        nl_sol = 2.0 * local_sol
        retVec = np.array([vec[i] * nl_sol[i] for i in range(3)])
        return retVec


class ItuSharma(ModelStenosis):
    # CAVEAT: here I do not take into account the term with the average flow rate over the heartbeat!!
    def __init__(self, R, R2, L):
        self.R = R
        self.R2 = R2
        self.L = L
        return

    def get_vector_dot(self):
        return np.array([0.0, 0.0, self.L])

    def get_vector(self):
        return np.array([-1.0, 1.0, self.R])

    def get_vector_nonlinear(self):
        return np.array([0.0, 0.0, self.R2])

    def evaluate_nonlinear(self, sol, index):
        vec = self.get_vector_nonlinear()
        local_sol = sol[3*index:3*(index+1), 0]
        nl_sol = local_sol * np.abs(local_sol)
        return np.inner(vec, nl_sol)

    def evaluate_jacobian_nonlinear(self, sol, index):
        vec = self.get_vector_nonlinear()
        local_sol = sol[3*index:3*(index+1), 0]
        nl_sol = 2.0 * local_sol * np.sign(local_sol)
        retVec = np.array([vec[i] * nl_sol[i] for i in range(3)])
        return retVec


class ResistanceStenosis(ModelStenosis, Resistance):
    def __init__(self, R, R2):
        Resistance.__init__(self, R)
        self.R2 = R2
        return

    def get_vector_nonlinear(self):
        return np.array([0.0, 0.0, self.R2])

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


class Windkessel2Stenosis(ModelStenosis, Windkessel2):
    def __init__(self, R, C, R2):
        Windkessel2.__init__(self, R, C)
        self.R2 = R2
        return

    def get_vector_nonlinear(self):
        return np.array([0.0, 0.0, self.R2])

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

