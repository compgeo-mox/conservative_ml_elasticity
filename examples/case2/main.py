import numpy as np

import porepy as pp
import pygeon as pg

import os
import sys

sys.path.insert(0, "src/")
from solver import Solver


class LocalSolver(Solver):
    def __init__(self, sd, data, keyword, if_spt, body_force):
        self.body_force = body_force
        super().__init__(sd, data, keyword, if_spt)

    def get_f(self):

        # e' strana la body_force

        fun = lambda _: np.array([0, self.body_force, 0])
        mass = self.discr_u.assemble_mass_matrix(self.sd)
        bd = self.discr_u.interpolate(self.sd, fun)

        f = np.zeros(self.dofs[1] + self.dofs[2])
        f[: self.dofs[1]] = mass @ bd
        return f

    def get_g(self):
        x_min = self.sd.face_centers[0, :].min()
        left = np.isclose(self.sd.face_centers[0, :], x_min)

        # define the boundary condition
        u_boundary = lambda _: np.array([0, 0, 0])

        return self.discr_s.assemble_nat_bc(self.sd, u_boundary, left), left

    def ess_bc(self):
        sd = self.sd

        x_max = sd.face_centers[0, :].max()
        y_max, y_min = sd.face_centers[1, :].max(), sd.face_centers[1, :].min()
        z_max, z_min = sd.face_centers[2, :].max(), sd.face_centers[2, :].min()

        # select the faces for the essential boundary conditions
        top = np.isclose(sd.face_centers[2, :], z_max)
        bottom = np.isclose(sd.face_centers[2, :], z_min)
        right = np.isclose(sd.face_centers[0, :], x_max)
        front = np.isclose(sd.face_centers[1, :], y_max)
        back = np.isclose(sd.face_centers[1, :], y_min)

        ess_dof = np.tile(
            np.logical_or.reduce((top, bottom, right, front, back)), sd.dim**2
        )
        ess_val = np.zeros_like(ess_dof, dtype=float)

        return ess_dof, ess_val


if __name__ == "__main__":
    folder = os.path.dirname(os.path.abspath(__file__))
    mesh_size = 0.2
    keyword = "elasticity"
    tol = 1e-5
    tol_array = np.power(10.0, np.arange(-5, -10, -1))

    bbox = {"xmin": 0, "xmax": 2, "ymin": 0, "ymax": 0.5, "zmin": 0, "zmax": 0.5}
    domain = pp.Domain(bbox)
    mdg = pg.grid_from_domain(domain, mesh_size)
    mdg.compute_geometry()

    data = {pp.PARAMETERS: {keyword: {"mu": 0.5, "lambda": 0.5}}}
    body_force = 1e-2
    if_spt = True
    solver = LocalSolver(mdg, data, keyword, if_spt, body_force)

    # check with a direct computation
    s, u, r = solver.compute_direct()
    sf = solver.compute_sf()

    # compute s0 directly from s and sf
    s0 = s - sf

    s0_v2 = solver.S0(s)
    print(np.linalg.norm(s0 - s0_v2))
    solver.check_s0(s0)

    # export the results
    solver.export(u, r, "sol", folder)
