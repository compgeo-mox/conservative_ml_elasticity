import numpy as np

import porepy as pp
import pygeon as pg

import os
import sys

src_path = os.path.join(__file__, "../../../src")
sys.path.insert(0, os.path.realpath(src_path))

from solver import Solver


class LocalSolver(Solver):
    def __init__(self, sd, data, keyword, sptr, body_force, force):
        self.body_force = body_force
        self.force = force
        super().__init__(sd, data, keyword, sptr)

    def get_f(self):
        fun = lambda _: np.array([0, self.body_force, 0])
        mass = self.discr_u.assemble_mass_matrix(self.sd)
        bd = self.discr_u.interpolate(self.sd, fun)

        f = np.zeros(self.dofs[1] + self.dofs[2])
        f[: self.dofs[1]] = mass @ bd
        return f

    @staticmethod
    def get_nat_bc(sd):
        bottom = np.isclose(sd.face_centers[1, :], 0)
        return bottom

    def get_g(self):
        nat_bc = self.get_nat_bc(self.sd)

        # define the boundary condition
        u_boundary = lambda _: np.array([0, 0, 0])

        return self.discr_s.assemble_nat_bc(self.sd, u_boundary, nat_bc), nat_bc

    def ess_bc(self):
        sd = self.sd

        # select the faces for the essential boundary conditions
        top = np.isclose(sd.face_centers[1, :], 1)
        left = np.isclose(sd.face_centers[0, :], 0)
        right = np.isclose(sd.face_centers[0, :], 1)

        b_faces = np.logical_or.reduce((top, left, right))
        ess_dof = np.tile(b_faces, sd.dim**2)

        # function for the essential boundary conditions
        val = np.array([[0, 0, 0], [0, self.force, 0]])
        fct = lambda pt: val if np.isclose(pt[1], 1) else 0 * val

        # interpolate the essential boundary conditions
        ess_val = -self.discr_s.interpolate(sd, fct)

        return ess_dof, ess_val


if __name__ == "__main__":
    folder = os.path.dirname(os.path.abspath(__file__))
    mesh_size = 0.1
    keyword = "elasticity"

    dim = 2
    mdg = pg.unit_grid(dim, mesh_size)
    mdg.compute_geometry()

    data = {pp.PARAMETERS: {keyword: {"mu": 0.5, "lambda": 1}}}
    body_force = -1e-2
    force = 1e-3

    nat_bc = LocalSolver.get_nat_bc(mdg.subdomains()[0])
    sptr = pg.SpanningTreeElasticity(mdg, nat_bc)

    solver = LocalSolver(mdg, data, keyword, sptr, body_force, force)

    # check with a direct computation
    s, u, r = solver.compute_direct()
    sf = solver.compute_sf()

    # compute s0 directly from s and sf
    s0 = s - sf

    s0_v2 = solver.S0(s)
    print(np.linalg.norm(s0 - s0_v2))
    solver.check_s0(s0)

    solver.export(u, r, "sol", folder)
