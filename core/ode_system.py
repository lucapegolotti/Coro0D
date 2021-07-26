import numpy as np


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
        bc_manager.set_starting_row_bcs(self.rowbcs)
        bc_manager.add_bcs_dot(self.smatrix_dot)
        bc_manager.add_bcs(self.smatrix)

        self.is_linear = all([not block.isStenotic for block in blocks])

        return

    def solve_steady(self):

        print("\nSolving the steady problem...")

        syssize = self.smatrix.shape[0]
        rhs = np.zeros((syssize, 1))
        self.bc_manager.apply_bc_vector(rhs, 0.0, steady=True)

        sol = np.linalg.solve(-self.smatrix, rhs)

        return sol

    def assemble_system_matrix_dot(self):
        nblocks = len(self.blocks)
        smatrix_dot = np.zeros([self.nvariables, self.nvariables])
        for i in range(nblocks):
            vecdot = self.blocks[i].model.get_vector_dot()
            smatrix_dot[i, 3 * i + 0] = vecdot[0]
            smatrix_dot[i, 3 * i + 1] = vecdot[1]
            smatrix_dot[i, 3 * i + 2] = vecdot[2]

        return smatrix_dot

    def assemble_system_matrix(self):
        nblocks = len(self.blocks)
        smatrix = np.zeros([self.nvariables, self.nvariables])
        for i in range(nblocks):
            vecdot = self.blocks[i].model.get_vector()
            smatrix[i, 3 * i + 0] = vecdot[0]
            smatrix[i, 3 * i + 1] = vecdot[1]
            smatrix[i, 3 * i + 2] = vecdot[2]

        self.add_constraints(smatrix)

        return smatrix

    def add_constraints(self, matrix):
        nblocks = len(self.blocks)
        constraintrow = nblocks
        # add constraints
        for connectivity in self.connectivity:
            # conservation of flow
            isboundary = True
            for iflag in range(nblocks):
                # incoming flow
                if connectivity[iflag] in {0.5, 1}:
                    isboundary = False
                    matrix[constraintrow, 3 * iflag + 2] = 1
                if connectivity[iflag] in {-1, -0.5}:
                    isboundary = False
                    matrix[constraintrow, 3 * iflag + 2] = -1
            if not isboundary:
                constraintrow += 1

            # equality of pressure: we look for all blocks with +-1/0.5 and we
            # add one constraint for every couple
            indices = np.hstack([np.where(np.abs(connectivity) == 0.5)[0],
                                 np.where(np.abs(connectivity) == 1)[0]])
            nindices = indices.shape[0]
            for i in range(1, nindices):
                if connectivity[indices[0]] in {0.5, 1}:
                    # + 0 corresponds to inlet
                    matrix[constraintrow, 3 * indices[0] + 0] = 1
                else:
                    # + 1 corresponds to outlet
                    matrix[constraintrow, 3 * indices[0] + 1] = 1
                if connectivity[indices[i]] in {0.5, 1}:
                    # + 0 corresponds to inlet
                    matrix[constraintrow, 3 * indices[i] + 0] = -1
                else:
                    # + 1 corresponds to outlet
                    matrix[constraintrow, 3 * indices[i] + 1] = -1
                constraintrow += 1

        self.rowbcs = constraintrow

        return

    def evaluate_constant_term(self):
        nblocks = len(self.blocks)
        Kvec = np.zeros([self.nvariables, 1])
        for i in range(nblocks):
            Kvec[i] = self.blocks[i].model.get_constant()

        return Kvec

    def evaluate_nonlinear(self, sol):
        nblocks = len(self.blocks)
        nlvec = np.zeros([self.nvariables, 1])
        for i in range(nblocks):
            nlvec[i] = self.blocks[i].model.evaluate_nonlinear(sol, i) \
                       if self.blocks[i].isStenotic else 0

        return nlvec

    def evaluate_jacobian_nonlinear(self, sol):
        nblocks = len(self.blocks)
        jacmatrix = np.zeros([self.nvariables, self.nvariables])
        for i in range(nblocks):
            jacmatrix[i, 3*i:3*(i+1)] = self.blocks[i].model.evaluate_jacobian_nonlinear(sol, i) \
                                        if self.blocks[i].isStenotic else np.zeros(3)

        return jacmatrix



