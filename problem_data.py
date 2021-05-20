class ProblemData:
    def __init__(self):
        # tolerance to determine if two points are the same
        self.tol = 0.5
        # maxlength of the singe vessel portion
        self.maxlength = 4 * self.tol
        # density of blood
        self.density = 1.06
        # viscosity of blood
        self.viscosity = 0.04
        # elastic modulus
        self.E = 2 * 10**5
        # vessel thickness ration w.r.t. diameter
        self.thickness_ratio = 0.08
        # use pressure at inlet
        self.use_inlet_pressure = True
        # timestep size
        self.deltat = 0.005
        # initial time
        self.t0 = 0.0
        # final time
        self.T = 5.0
        # ramp rime
        self.t0ramp = -2.0
