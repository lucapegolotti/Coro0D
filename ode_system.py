import numpy as np
from physical_block import PhysicalBlock

class ODESystem:
    def __init__(self, blocks, connectivity):
        self.blocks = blocks
        self.connectivity = connectivity

    def get_system_matrix_dot(self):
        nblocks = len(self.blocks)
        smatrix_dot = np.zeros([nblocks * 3, nblocks * 3])
        for i in range(0, nblocks):
            vecdot = self.blocks[i].model.get_vector_dot()
            smatrix_dot[i,3*i + 0] = vecdot[0]
            smatrix_dot[i,3*i + 1] = vecdot[1]
            smatrix_dot[i,3*i + 2] = vecdot[2]

        return smatrix_dot

    def get_system_matrix(self):
        nblocks = len(self.blocks)
        smatrix = np.zeros([nblocks * 3, nblocks * 3])
        for i in range(0, nblocks):
            vecdot = self.blocks[i].model.get_vector()
            smatrix[i,3*i + 0] = vecdot[0]
            smatrix[i,3*i + 1] = vecdot[1]
            smatrix[i,3*i + 2] = vecdot[2]

        constraintrow = nblocks
        # add constraints
        for connectivity in self.connectivity:
            # conservation of flow
            isboundary = True
            for iflag in range(0, nblocks):
                # incoming flow
                if connectivity[iflag] == 1:
                    isboundary = False
                    smatrix[constraintrow,3*iflag + 2] = 1
                if connectivity[iflag] == -1:
                    isboundary = False
                    smatrix[constraintrow,3*iflag + 2] = -1
            if not isboundary:
                constraintrow += 1

            # equality of pressure: we look for all blocks with +-1 and we
            # add one constraint for every couple
            indices = np.where(np.abs(connectivity) == 1)[0]
            nindices = indices.shape[0]
            for i in range(1, nindices):
                if connectivity[indices[0]] == 1:
                    # + 0 corresponds to inlet
                    smatrix[constraintrow, 3*indices[0] + 0] = 1
                else:
                    # + 1 corresponds to outlet
                    smatrix[constraintrow, 3*indices[0] + 1] = 1
                if connectivity[indices[i]] == 1:
                    # + 0 corresponds to inlet
                    smatrix[constraintrow, 3*indices[i] + 0] = -1
                else:
                    # + 1 corresponds to outlet
                    smatrix[constraintrow, 3*indices[i] + 1] = -1
                constraintrow += 1
        # the boundary conditions (remaining rows) are applied in BDF
        return smatrix
