import numpy as np

import porepy as pp
import pygeon as pg

import os
import sys

src_path = os.path.join(__file__, "../../../src")
sys.path.insert(0, os.path.realpath(src_path))

from solver import Solver


class LocalSolver(Solver):
    def __init__(self, sd, data, keyword, if_spt, body_force, force):
        self.body_force = body_force
        self.force = force
        super().__init__(sd, data, keyword, if_spt)

    def get_f(self):
        fun = lambda _: np.array([0, self.body_force, 0])
        mass = self.discr_u.assemble_mass_matrix(self.sd)
        bd = self.discr_u.interpolate(self.sd, fun)

        f = np.zeros(self.dofs[1] + self.dofs[2])
        f[: self.dofs[1]] = mass @ bd
        return f

    def get_g(self):
        sd = self.sd

        top = np.isclose(sd.face_centers[1, :], 1)
        bottom = np.isclose(sd.face_centers[1, :], 0)
        left = np.isclose(sd.face_centers[0, :], 0)
        right = np.isclose(sd.face_centers[0, :], 1)

        b_faces = np.logical_or.reduce((bottom, right, left))

        # define the boundary condition
        u_boundary = lambda _: np.array([0, 0, 0])

        return self.discr_s.assemble_nat_bc(sd, u_boundary, b_faces), b_faces

    def ess_bc(self):
        sd = self.sd

        # select the faces for the essential boundary conditions
        # top = np.isclose(sd.face_centers[1, :], 1)
        top = np.zeros(sd.num_faces, dtype=bool)
        ess_dof = np.tile(top, sd.dim**2)

        # function for the essential boundary conditions
        val = np.array([[0, 0, 0], [0, self.force, 0]])
        # fct = lambda pt: val if np.isclose(pt[1], 1) else 0 * val
        fct = lambda pt: 0 * val

        # interpolate the essential boundary conditions
        ess_val = -self.discr_s.interpolate(sd, fct)

        return ess_dof, ess_val


if __name__ == "__main__":
    folder = os.path.dirname(os.path.abspath(__file__))
    mesh_size = 0.05
    keyword = "elasticity"

    dim = 2
    mdg = pg.unit_grid(dim, mesh_size)
    mdg.compute_geometry()

    data = {pp.PARAMETERS: {keyword: {"mu": 0.5, "lambda": 1}}}
    body_force = -1e-2
    force = 1e-3
    if_spt = True
    solver = LocalSolver(mdg, data, keyword, if_spt, body_force, force)

    # check with a direct computation
    s, u, r = solver.compute_direct()
    sf = solver.compute_sf()

    # compute s0 directly from s and sf
    s0 = s - sf

    s0_v2 = solver.S0(s)
    print(np.linalg.norm(s0 - s0_v2))

    solver.export(u, r, "sol", folder)
