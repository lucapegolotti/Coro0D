from vessel_portion import VesselPortion

class OutletBC:
    def __init__(self, portion, index, bc_type, distal_pressure_generator):
        self.portion = portion
        self.portionindex = index
        self.bc_type = bc_type
        self.nvariables = 5
        self.distal_pressure_generator = distal_pressure_generator


    # the variables are ordered as follows: p0, pd, Q, K1,K2, where K1 and K2
    # are the pressure drops across the two capacitors
    def apply_bc_matrix_dot(self, matrix_dot, row, col):
        # here we only need to add the eqs for K1 and K2
        matrix_dot[row + 0, col + 3] = 1
        matrix_dot[row + 1, col + 4] = 1

        return 6

    def apply_bc_matrix(self, matrix, row, col):
        if self.bc_type == "coronary":
            Ra = self.portion.compute_Ra()
            Ramicro = self.portion.compute_Ramicro()
            Rvmicro = self.portion.compute_Rvmicro()
            Rv = self.portion.compute_Rv()
            Ca = self.portion.compute_Ca()
            Cim = self.portion.compute_Cim()

            # Ca \dot{K1} = Q - (K1 - K2 - Pd) / Ramicro
            matrix[row + 0, col + 1] = 1 / (Ramicro * Ca)
            matrix[row + 0, col + 2] = 1 / Ca
            matrix[row + 0, col + 3] = -1 / (Ramicro * Ca)
            matrix[row + 0, col + 4] = 1 / (Ramicro * Ca)

            # Cim \dot{K2} = (K1 - K2 - Pd) / (Ramicro) - (K2 + Pd) / (Rvmicro + Rv)
            matrix[row + 1, col + 1] = -1 / (Cim) * (1 / Ramicro + 1/(Rvmicro + Rv))
            matrix[row + 1, col + 3] = 1 / (Ramicro * Cim)
            matrix[row + 1, col + 4] = -1 / (Cim) * (1 / Ramicro + 1/(Rvmicro + Rv))
        elif self.bc_type == "resistance":
            Ra = self.portion.compute_Ra() + self.portion.compute_Rv() + \
                 self.portion.compute_Ramicro() + self.portion.compute_Rvmicro()
        elif self.bc_type == "zero":
            Ra = 0
        else:
            raise NotImplementedError(self.bc_type + " bc not implemented")

        print(Ra)
        # p0 - Ra * Q - K1 = 0
        matrix[row + 2, col + 0] = 1
        matrix[row + 2, col + 1] = 0
        matrix[row + 2, col + 2] = -Ra
        matrix[row + 2, col + 3] = -1

        # pd must be assigned (-1 because we are going to put +pd in the vector with the data)
        matrix[row + 3, col + 1] = -1

        # continuity of pressure with neighboring portion
        matrix[row + 4, col + 0] = 1
        matrix[row + 4, self.portionindex * 3 + 1] = -1 # outlet pressure of portion

        # continuity of flowrate with neighboring portion
        matrix[row + 5, col + 2] = 1
        matrix[row + 5, self.portionindex * 3 + 2] = -1

        return 6

    def apply_bc_vector(self, vector, time, row):
        vector[row + 3] = self.distal_pressure_generator.distal_pressure(time)
        return 6
