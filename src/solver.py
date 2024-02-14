import abc

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class Solver:
    def __init__(self, mdg, data, keyword):
        self.mdg = mdg
        self.keyword = keyword

        self.sptr = pg.SpanningTreeElasticity(mdg)

        self.discr_s = pg.VecBDM1(self.keyword)
        self.discr_u = pg.VecPwConstants(self.keyword)
        self.discr_r = pg.PwConstants(self.keyword)

        self.build_matrices(data)

    def build_matrices(self, data):
        sd = self.mdg.subdomains(dim=self.mdg.dim_max())[0]

        self.Ms = self.discr_s.assemble_mass_matrix(sd, data)
        self.Ms_lumped = self.discr_s.assemble_lumped_matrix(sd, data)

        self.Mu = self.discr_u.assemble_mass_matrix(sd)
        self.Mr = self.discr_r.assemble_mass_matrix(sd)

        div = self.Mu @ self.discr_s.assemble_diff_matrix(sd)
        asym = self.Mr @ self.discr_s.assemble_asym_matrix(sd)

        self.B = sps.vstack((-div, -asym))

        self.spp = sps.bmat([[self.Ms, -self.B.T], [self.B, None]], format="csc")

        self.dofs = np.array(
            [self.discr_s.ndof(sd), self.discr_u.ndof(sd), self.discr_r.ndof(sd)]
        )

    def SI(self, x):
        return self.sptr.solve(x)

    def SI_T(self, x):
        return self.sptr.solve_transpose(x)

    def S0(self, x):
        return x - self.SI(self.B @ x)

    def S0_T(self, x):
        return x - self.B.T @ self.SI_T(x)

    def compute_sf(self):
        f = self.get_f()
        return self.sptr.solve(f)

    def compute_s0(self):
        s, _, _ = self.compute_direct()
        return self.S0(s)

    def compute_s0_cg(self, sf, rtol=1e-10):

        iters = 0

        def nonlocal_iterate(arr):
            nonlocal iters
            iters += 1

        def matvec(x):
            return self.S0_T(self.Ms @ self.S0(x))

        b = self.S0_T(self.get_g() - self.Ms @ sf)
        A = sps.linalg.LinearOperator([b.size] * 2, matvec=matvec)

        s, exit_code = sps.linalg.cg(A, b, rtol=rtol, callback=nonlocal_iterate)
        if exit_code != 0:
            raise ValueError("CG did not converge")
        print(iters)

        return self.S0(s)

    def check_s0(self, s0):
        if np.allclose(self.B @ s0, 0):
            print("s0 is in the kernel of B")
        else:
            raise ValueError("s0 is not in the kernel of B")

    def compute_all(self, s0, sf):
        g = self.get_g()

        s = s0 + sf
        x = self.sptr.solve_transpose(self.Ms @ s - g)
        u, r = x[: self.dofs[1]], x[self.dofs[1] :]

        return s, u, r

    def compute_direct(self):
        f = self.get_f()
        g = self.get_g()

        rhs = np.hstack((g, f))
        x = sps.linalg.spsolve(self.spp, rhs)

        idx = np.cumsum(self.dofs[:-1])
        return np.split(x, idx)

    @abc.abstractmethod
    def get_f(self):
        pass

    @abc.abstractmethod
    def get_g(self):
        pass

    def compute_error(self, xn, x, M):
        delta = xn - x
        norm_x = np.sqrt(x @ M @ x)
        return np.sqrt(delta @ M @ delta) / (norm_x if norm_x else 1)
