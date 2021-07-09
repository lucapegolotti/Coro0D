from abc import ABC, abstractmethod
import numpy as np


# The constitutive equation is written in this form:
#
# [a, b, c]\dot{[P_in, P_out, Q]^T} = [d, e, f][P_in, P_out, Q]^T + [g, h, i][P_in^2, P_out^2, Q^2]^T

class Model(ABC):
    # this is the method that computes [a, b, c]
    @abstractmethod
    def get_vector_dot(self):
        pass

    # this is the method that computes [d, e, f]
    @abstractmethod
    def get_vector(self):
        pass

    # this is the method that computes [g, h, i]
    @abstractmethod
    def get_vector_square(self):
        pass


class Windkessel2(Model):
    def __init__(self, R, C):
        self.R = R
        self.C = C
        return

    def get_vector_dot(self):
        return np.array([self.C, -self.C, 0])

    def get_vector(self):
        return np.array([-1 / self.R, 1 / self.R, 1])

    def get_vector_square(self):
        return np.array([0, 0, 0])


class Resistance(Model):
    def __init__(self, R):
        self.R = R
        return

    def get_vector_dot(self):
        return np.array([0, 0, 0])

    def get_vector(self):
        return np.array([-1 / self.R, 1 / self.R, 1])

    def get_vector_square(self):
        return np.array([0, 0, 0])


class Stenosis(Model):
    def __init__(self, R, R2):
        self.R = R
        self.R2 = R2
        return

    def get_vector_dot(self):
        return np.array([0, 0, 0])

    def get_vector(self):
        return np.array([-1 / self.R, 1 / self.R, 1])

    def get_vector_square(self):
        return np.array([-1 / self.R2, 1 / self.R2, 1])
