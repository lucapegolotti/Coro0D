import numpy as np
from physical_block import PhysicalBlock

class ODESystem:
    def __init__(self, blocks, connectivity, bc_manager):
        nblocks = len(blocks)
        self.blocks = blocks
        self.connectivity = connectivity
        self.bc_manager = bc_manager
        self.nvariables = 3 * nblocks + \
                          bc_manager.noutlets * bc_manager.outletbcs[0].nvariables

        self.smatrix_dot = self.assemble_system_matrix_dot()
        self.smatrix = self.assemble_system_matrix()
        bc_manager.add_bcs_dot(self.smatrix_dot, self.rowbcs)
        bc_manager.add_bcs(self.smatrix, self.rowbcs)

    def assemble_system_matrix_dot(self):
        nblocks = len(self.blocks)
        smatrix_dot = np.zeros([self.nvariables, self.nvariables])
        for i in range(0, nblocks):
            vecdot = self.blocks[i].model.get_vector_dot()
            smatrix_dot[i,3*i + 0] = vecdot[0]
            smatrix_dot[i,3*i + 1] = vecdot[1]
            smatrix_dot[i,3*i + 2] = vecdot[2]

        return smatrix_dot

    def assemble_system_matrix(self):
        nblocks = len(self.blocks)
        smatrix = np.zeros([self.nvariables, self.nvariables])
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

        self.rowbcs = constraintrow

        # # apply bcs to matrix
        # # find first row with all zeros
        # self.inletbcrow = constraintrow
        #
        # # find index inlet block
        # indexinletblock = np.where(self.connectivity == 2)
        # if self.use_inlet_pressure:
        #     # in this case we fix the inlet pressure
        #     self.bdfmatrix[self.inletbcrow,indexinletblock[1] * 3 + 0] = 1
        # else:
        #     # in this case we fix the inlet flowrate
        #     self.bdfmatrix[self.inletbcrow,indexinletblock[1] * 3 + 2] = 1
        #
        # # find max outlet flag
        # maxoutletflag = int(np.max(self.connectivity))
        #
        # self.outletbcrows = []
        # for flag in range(3, maxoutletflag + 1):
        #     currow = self.inletbcrow + flag - 2
        #     indexoutletblock = np.where(self.connectivity == flag)
        #     self.bdfmatrix[currow,indexoutletblock[1] * 3 + 1] = 1
        #     self.outletbcrows.append(currow)
        #


        return smatrix
