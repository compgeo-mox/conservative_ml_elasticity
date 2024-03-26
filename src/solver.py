import abc

import numpy as np
import scipy.sparse as sps

import pygeon as pg
import porepy as pp


class Solver:
    def __init__(self, sd, data, keyword):
        self.sd = sd
        self.keyword = keyword

        # define the discretization objects useful for our case
        self.discr_s = pg.VecBDM1(self.keyword)
        self.discr_u = pg.VecPwConstants(self.keyword)
        self.discr_r = (pg.PwConstants if sd.dim == 2 else pg.VecPwConstants)(
            self.keyword
        )

        self.build_matrices(data)

    def build_matrices(self, data):
        # build the mass matrix for the stress
        self.Ms = self.discr_s.assemble_mass_matrix(self.sd, data)

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

        # build the saddle point matrix
        self.spp = sps.bmat([[self.Ms, -self.B.T], [self.B, None]], format="csc")

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
        # ess_dof, ess_val = ess_dof[: self.dofs[0]], ess_val[: self.dofs[0]]

        # define the restriction operator
        to_keep = np.logical_not(self.ess_dof[: self.dofs[0]])
        self.R_0 = pg.numerics.linear_system.create_restriction(to_keep)

        if self.sd.dim == 2:
            # build the spanning tree solve if it is available
            sptr = pg.SpanningTreeElasticity(self.sd, 0)
            self.sptr_solve = sptr.solve
            self.sptr_solve_transpose = sptr.solve_transpose
        elif self.sd.dim == 3:
            # in 3d consider the standard BBT approach
            B_red = self.B @ self.R_0.T @ self.R_0
            BBT = B_red @ B_red.T
            self.sptr_solve = lambda x: B_red.T @ sps.linalg.spsolve(BBT, x)
            self.sptr_solve_transpose = lambda x: sps.linalg.spsolve(BBT, B_red @ x)
        else:
            raise ValueError("not implemented")

    def SI(self, x):
        # solve the spanning tree problem
        return self.sptr_solve(x)

    def SI_T(self, x):
        # solve the transpose of the spanning tree problem
        return self.sptr_solve_transpose(x)

    def S0(self, x):
        # solve the homogeneous problem
        return x - self.SI(self.B @ x)

    def S0_T(self, x):
        # solve the transpose of the homogeneous problem
        return x - self.B.T @ self.SI_T(x)

    def compute_sf(self):
        # compute the particular solution
        f = self.get_f() - self.B @ self.ess_val[: self.dofs[0]]
        return self.sptr_solve(f) + self.ess_val[: self.dofs[0]]

    def compute_s0(self):
        # compute the homogeneous solution by solving the direct problem
        s, _, _ = self.compute_direct()
        return self.S0(s)

    def compute_s0_cg(self, sf, rtol=1e-10):
        # compute the homogeneous solution by solving the iterative problem
        print("da inserire le bc essenziali")

        # help function to count the number of iterations
        iters = 0

        def nonlocal_iterate(_):
            nonlocal iters
            iters += 1

        # define implicitly the operator associated to the reduced system
        A_op = lambda x: self.S0_T(self.Ms @ self.S0(x))
        A_op_red = lambda x: self.R_0 @ A_op(self.R_0.T @ x)

        # define the right-hand side of the reduced system
        b = self.R_0 @ self.S0_T(self.get_g() - self.Ms @ sf)

        A = sps.linalg.LinearOperator([b.size] * 2, matvec=A_op_red)

        # solve the reduced system with CG
        s, exit_code = sps.linalg.cg(A, b, callback=nonlocal_iterate)  # rtol=rtol

        if exit_code != 0:
            raise ValueError("CG did not converge")
        else:
            print("Number of iterations", iters)

        return self.S0(self.R_0.T @ s)

    def check_s0(self, s0):
        # check if the homogeneous solution respects the constraints
        if np.allclose(self.B @ s0, 0):
            print("s0 is in the kernel of B")
        else:
            raise ValueError("s0 is not in the kernel of B")

    def compute_all(self, s0, sf):
        # compute the source term
        g = self.get_g()

        # post process the stress
        s = s0 + sf

        # post process the displacemnet and the rotation
        x = self.sptr_solve_transpose(self.Ms @ s - g)
        u, r = x[: self.dofs[1]], x[self.dofs[1] :]

        return s, u, r

    def compute_direct(self):
        # compute both right hand sides
        f = self.get_f()
        g = self.get_g()
        rhs = np.hstack((g, f))

        # solve the saddle point problem
        ls = pg.LinearSystem(self.spp, rhs)
        if np.any(self.ess_dof):
            ls.flag_ess_bc(self.ess_dof, self.ess_val)
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

    def compute_error(self, xn, x, M):
        # compute the L2 error
        delta = xn - x
        norm_x = np.sqrt(x @ M @ x)
        return np.sqrt(delta @ M @ delta) / (norm_x if not np.isclose(norm_x, 0) else 1)

    def export(self, u, r, file_name, folder):
        dim = self.sd.dim
        # post process variables
        proj_u = self.discr_u.eval_at_cell_centers(self.sd)
        cell_u = (proj_u @ u).reshape((dim, -1), order="C")

        proj_r = self.discr_r.eval_at_cell_centers(self.sd)
        if dim == 2:
            cell_u = np.vstack((cell_u, np.zeros(cell_u.shape[1])))
            cell_r = proj_r @ r
        else:
            cell_r = (proj_r @ r).reshape((dim, -1), order="C")

        save = pp.Exporter(self.sd, file_name, folder_name=folder)
        save.write_vtu([("cell_u", cell_u), ("cell_r", cell_r)])
