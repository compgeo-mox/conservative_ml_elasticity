import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class Solver:
    def __init__(self, mdg, data, keyword):
        self.mdg = mdg
        self.keyword = keyword

        self.sptr = pg.SpanningTreeElasticity(mdg)

        self.vec_bdm1 = pg.VecBDM1(self.keyword)
        self.vec_p0 = pg.VecPwConstants(self.keyword)
        self.p0 = pg.PwConstants(self.keyword)

        self.build_matrices(data)

    def build_matrices(self, data):
        sd = mdg.subdomains(dim=self.mdg.dim_max())[0]

        self.Ms = self.vec_bdm1.assemble_mass_matrix(sd, data)
        self.Mu = self.vec_p0.assemble_mass_matrix(sd)
        self.Mr = self.p0.assemble_mass_matrix(sd)

        div = self.Mu @ self.vec_bdm1.assemble_diff_matrix(sd)
        asym = self.Mr @ self.vec_bdm1.assemble_asym_matrix(sd)

        self.B = sps.vstack((-div, -asym))

        self.spp = sps.bmat([[self.Ms, -self.B.T], [self.B, None]], format="csc")

        self.dofs = np.array(
            [self.vec_bdm1.ndof(sd), self.vec_p0.ndof(sd), self.p0.ndof(sd)]
        )

    def S_I(self, f):
        return self.sptr.solve(f)

    def S_0(self, r):
        return r - self.S_I(self.B @ r)

    def compute_s0(self):
        s, _, _ = self.compute_direct()
        return s - self.S_I(self.B @ s)

    def compute_sf(self):
        f = self.get_f()
        return self.sptr.solve(f)

    def compute_all(self, s0, sf):
        s = s0 + sf

        g = self.get_g()
        x = self.sptr.solve_transpose(self.Ms @ s - g)
        u, r = x[: self.dofs[1]], x[self.dofs[1] :]

        return s, u, r

    def get_f(self):
        return np.zeros(self.dofs[1] + self.dofs[2])

    def get_g(self):
        sd = mdg.subdomains(dim=self.mdg.dim_max())[0]

        u_boundary = lambda x: np.array([-0.5 - x[1], -0.5 + x[0], 0])

        b_faces = sd.tags["domain_boundary_faces"]
        return self.vec_bdm1.assemble_nat_bc(sd, u_boundary, b_faces)

    def compute_direct(self):
        f = self.get_f()
        g = self.get_g()

        rhs = np.hstack((g, f))
        x = sps.linalg.spsolve(self.spp, rhs)

        idx = np.cumsum(self.dofs[:-1])
        return np.split(x, idx)


if __name__ == "__main__":
    step_size = 0.1
    keyword = "elasticity"

    mdg = pg.unit_grid(2, step_size)
    mdg.compute_geometry()

    data = {pp.PARAMETERS: {keyword: {"mu": 0.5, "lambda": 0.5}}}
    solver = Solver(mdg, data, keyword)

    s0 = solver.compute_s0()
    sf = solver.compute_sf()

    s, u, r = solver.compute_all(s0, sf)

    s_dir, u_dir, r_dir = solver.compute_direct()

    print(
        np.linalg.norm(s - s_dir), np.linalg.norm(u - u_dir), np.linalg.norm(r - r_dir)
    )

    pass
