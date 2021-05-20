from vessel_portion import VesselPortion

class InletBC:
    def __init__(self, portion, index, bc_type):
        self.portion = portion
        self.index = index
        self.bc_type = bc_type

    def apply_bc_matrix_dot(self, matrix_dot, row):
        return

    def apply_bc_matrix(self, matrix, row):
        if self.bc_type == "pressure":
            matrix[row,self.index * 3 + 0] = -1
        elif self.bc_type == "flowrate":
            matrix[row,self.index * 3 + 2] = -1
        else:
            raise NotImplementedError(self.bc_type + " bc not implemented")
