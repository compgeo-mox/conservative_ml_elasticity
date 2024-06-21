import abc
import time

import numpy as np
import scipy.sparse as sps

import pygeon as pg
import porepy as pp


class Solver:
    def __init__(self, mdg, data, keyword, if_spt):
        self.mdg = mdg
        self.sd = mdg.subdomains()[0]
        self.keyword = keyword

        # define the discretization objects useful for our case
        self.discr_s = pg.VecBDM1(self.keyword)
        self.discr_u = pg.VecPwConstants(self.keyword)
        self.discr_r = (pg.PwConstants if self.sd.dim == 2 else pg.VecPwConstants)(
            self.keyword
        )

        # build the matrices
        self.build_matrices(data, if_spt)

    def build_matrices(self, data, if_spt):
        # build the mass matrix for the stress
        self.Ms = self.discr_s.assemble_mass_matrix(self.sd, data)

        # buld the lumped mass matrix for the preconditioning
        self.Ls = self.discr_s.assemble_lumped_matrix(self.sd, data)

        # build the mass matrix for the displacement
        self.Mu = self.discr_u.assemble_mass_matrix(self.sd)

        # build the mass matrix for the rotation
        self.Mr = self.discr_r.assemble_mass_matrix(self.sd)

        # build the divergence operator acting on the stress
        div = self.Mu @ self.discr_s.assemble_diff_matrix(self.sd)

        # build the asymmetric operator acting on the stress
        asym = self.Mr @ self.discr_s.assemble_asym_matrix(self.sd)

        # build the constriant operator
        self.B = sps.vstack((-div, -asym))

        # assemble the vector BDM1 mass matrix useful for the ROM
        mass = self.discr_s.scalar_discr.assemble_mass_matrix(self.sd, data)
        self.D = sps.block_diag([mass] * self.sd.dim, format="csc")

        # build the degrees of freedom
        self.dofs = np.array(
            [
                self.discr_s.ndof(self.sd),
                self.discr_u.ndof(self.sd),
                self.discr_r.ndof(self.sd),
            ]
        )

        # consider essential bc
        self.ess_dof, self.ess_val = self.ess_bc()

        # consider the natual bc and the vector source term
        self.g_val, self.nat_dof = self.get_g()

        # define the restriction operator
        to_keep = np.logical_not(self.ess_dof)
        self.R_0 = pg.numerics.linear_system.create_restriction(to_keep)

        if if_spt:
            sptr = pg.SpanningTreeElasticity(self.mdg, self.nat_dof)

            self.sptr = sptr
            self.sptr_solve = sptr.solve
            self.sptr_solve_transpose = sptr.solve_transpose

        else:
            # consider the standard BBT approach
            B_red = self.B @ self.R_0.T @ self.R_0
            BBT = sps.linalg.splu(B_red @ B_red.T)

            self.sptr_solve = lambda x: B_red.T @ BBT.solve(x)
            self.sptr_solve_transpose = lambda x: BBT.solve(B_red @ x)

        # build the saddle point matrix
        self.spp = sps.bmat([[self.Ms, -self.B.T], [self.B, None]], format="csc")

    def SI(self, x):
        # solve the spanning tree problem
        return self.sptr_solve(x)

    def SI_T(self, x):
        # solve the transpose of the spanning tree problem
        return self.sptr_solve_transpose(x)

    def S0(self, x):
        # project to the kernel
        return x - self.SI(self.B @ x)

    def S0_T(self, x):
        # transpose of the projection to the kernel
        return x - self.B.T @ self.SI_T(x)

    def compute_sf(self):
        # compute the particular solution
        f = self.get_f() - self.B @ self.ess_val
        return self.sptr_solve(f) + self.ess_val

    def compute_s0(self, sf):
        # compute the homogeneous solution
        S_I = self.sptr.assemble_SI()
        S_0 = sps.eye_array(S_I.shape[0]) - S_I @ self.B

        A = S_0.T @ self.Ms @ S_0 + S_I @ S_I.T
        b = self.S0_T(self.g_val - self.Ms @ sf)

        ls = pg.LinearSystem(A, b)
        ls.flag_ess_bc(self.ess_dof, np.zeros_like(sf))

        return self.S0(ls.solve())

    def compute_s0_cg(self, sf, tol=1e-10, verbose = True):
        # compute the homogeneous solution by solving the iterative problem

        # help function to count the number of iterations
        iters = 0

        def nonlocal_iterate(_):
            nonlocal iters
            iters += 1

        # define implicitly the operator associated to the reduced system
        A_op = lambda x: self.S0_T(self.Ms @ self.S0(x))  # + self.SI(self.SI_T(x))
        A_op_red = lambda x: self.R_0 @ A_op(self.R_0.T @ x)

        # define implicitly the preconditioner
        L_inv = sps.linalg.splu(self.R_0 @ self.Ls @ self.R_0.T)
        P_op_red = lambda x: L_inv.solve(x)

        # define the right-hand side of the reduced system
        b = self.R_0 @ self.S0_T(self.g_val - self.Ms @ sf)

        A = sps.linalg.LinearOperator([b.size] * 2, matvec=A_op_red)
        P = sps.linalg.LinearOperator([b.size] * 2, matvec=P_op_red)

        # solve the reduced system with CG
        # start = time.time()

<<<<<<< HEAD
        if(verbose):
            print("Time to solve the reduced system", time.time() - start)

        if exit_code != 0:
            raise ValueError("CG did not converge")
        else:
            if(verbose):
                print("Number of iterations", iters)
=======
        s, exit_code = sps.linalg.cg(A, b, M=P, rtol=tol, callback=nonlocal_iterate)

        # print("Time to solve the reduced system", time.time() - start)

        if exit_code != 0:
            raise ValueError("CG did not converge")
        # else:
        #     print("Number of iterations / ndof: {} / {}".format(iters, len(b)))
>>>>>>> f53a9e08069078f72f5f38e5f171d8b1414fc88e

        return self.S0(self.R_0.T @ s)

    def check_s0(self, s0):
        # check if the homogeneous solution respects the constraints
        if np.allclose(self.B @ s0, 0):
            print("s0 is in the kernel of B")
            print(np.abs(self.B @ s0).max())
        else:
            raise ValueError("s0 is not in the kernel of B")

    def compute_all(self, s0, sf):
        # post process the stress
        s = s0 + sf

        # post process the displacemnet and the rotation
        x = self.sptr_solve_transpose(self.Ms @ s - self.g_val)
        u, r = x[: self.dofs[1]], x[self.dofs[1] :]

        return s, u, r

    def compute_direct(self):
        # compute both right hand sides
        f = self.get_f()
        rhs = np.hstack((self.g_val, f))

        # solve the saddle point problem
        ls = pg.LinearSystem(self.spp, rhs)
        if np.any(self.ess_dof):
            ls.flag_ess_bc(np.where(self.ess_dof)[0], self.ess_val)
        x = ls.solve()

        # split and return the solution
        idx = np.cumsum(self.dofs[:-1])
        return np.split(x, idx)

    @abc.abstractmethod
    def get_f(self):
        pass

    @abc.abstractmethod
    def get_g(self):
        pass

    @abc.abstractmethod
    def ess_bc(self):
        pass

    def compute_error(self, xn, x, M):
        # compute the L2 error
        delta = xn - x
        norm_x = np.sqrt(x @ M @ x)
        return np.sqrt(delta @ M @ delta) / (norm_x if not np.isclose(norm_x, 0) else 1)

    def export(self, u, r, file_name, folder):
        sd = self.sd
        # post process variables
        proj_u = self.discr_u.eval_at_cell_centers(sd)
        cell_u = (proj_u @ u).reshape((sd.dim, -1), order="C")

        proj_r = self.discr_r.eval_at_cell_centers(sd)
        if sd.dim == 2:
            cell_u = np.vstack((cell_u, np.zeros(cell_u.shape[1])))
            cell_r = proj_r @ r
        else:
            cell_r = (proj_r @ r).reshape((sd.dim, -1), order="C")

        save = pp.Exporter(sd, file_name, folder_name=folder)
        save.write_vtu([("cell_u", cell_u), ("cell_r", cell_r)])
