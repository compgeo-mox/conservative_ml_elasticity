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
        self.Mu = self.discr_u.assemble_mass_matrix(sd)
        self.Mr = self.discr_r.assemble_mass_matrix(sd)

        div = self.Mu @ self.discr_s.assemble_diff_matrix(sd)
        asym = self.Mr @ self.discr_s.assemble_asym_matrix(sd)

        self.B = sps.vstack((-div, -asym))

        self.spp = sps.bmat([[self.Ms, -self.B.T], [self.B, None]], format="csc")

        self.dofs = np.array(
            [self.discr_s.ndof(sd), self.discr_u.ndof(sd), self.discr_r.ndof(sd)]
        )

    def S_I(self, f):
        return self.sptr.solve(f)

    def S_0(self, s):
        return s - self.S_I(self.B @ s)

    def compute_s0(self):
        s, _, _ = self.compute_direct()
        return self.S_0(s)

    def compute_sf(self):
        f = self.get_f()
        return self.sptr.solve(f)

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
