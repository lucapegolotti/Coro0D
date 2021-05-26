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

    def get_vector_dot(self):
        return np.array([self.C, -self.C, 0])

    def get_vector(self):
        return np.array([-1/self.R, 1/self.R, 1])

class Resistance(Model):
    def __init__(self, R):
        self.R = R

    def get_vector_dot(self):
        return np.array([0, 0, 0])

    def get_vector(self):
        return np.array([-1/self.R, 1/self.R, 1])
