import numpy as np

import scipy.sparse as sps

import porepy as pp
import pygeon as pg

import sys

sys.path.insert(0, "src/")
from solver import Solver


class LocalSolver(Solver):
    def get_f(self):
        return np.ones(self.dofs[1] + self.dofs[2])

    def get_g(self):
        sd = self.mdg.subdomains(dim=self.mdg.dim_max())[0]

        u_boundary = lambda x: np.array([-0.5 - x[1], -0.5 + x[0], 0])

        b_faces = sd.tags["domain_boundary_faces"]
        return self.discr_s.assemble_nat_bc(sd, u_boundary, b_faces)


if __name__ == "__main__":
    step_size = 0.1
    keyword = "elasticity"

    dim = 2
    mdg = pg.unit_grid(dim, step_size)
    mdg.compute_geometry()

    data = {pp.PARAMETERS: {keyword: {"mu": 0.5, "lambda": 0.5}}}
    solver = LocalSolver(mdg, data, keyword)

    # step 1
    sf = solver.compute_sf()

    # step 2
    s0 = solver.compute_s0_cg(sf, rtol=1e-5)
    solver.check_s0(s0)

    # step 3
    s, u, r = solver.compute_all(s0, sf)

    # check with a direct computation
    s_dir, u_dir, r_dir = solver.compute_direct()

    err_s = solver.compute_error(s, s_dir, solver.Ms)
    err_u = solver.compute_error(u, u_dir, solver.Mu)
    err_r = solver.compute_error(r, r_dir, solver.Mr)

    print(err_s, err_u, err_r)
