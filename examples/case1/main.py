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
        top = np.isclose(sd.face_centers[1, :], 1)
        ess_dof = np.tile(top, sd.dim**2)

        # function for the essential boundary conditions
        val = np.array([[0, 0, 0], [0, self.force, 0]])
        fct = lambda pt: val if np.isclose(pt[1], 1) else 0 * val

        # interpolate the essential boundary conditions
        ess_val = -self.discr_s.interpolate(sd, fct)

        return ess_dof, ess_val


if __name__ == "__main__":
    # NOTE: difficulty to converge for RBM
    folder = "examples/case1/"
    mesh_size = 0.05
    keyword = "elasticity"
    tol_array = np.power(10.0, np.arange(-7, -13, -1))

    dim = 2
    mdg = pg.unit_grid(dim, mesh_size)
    mdg.compute_geometry()

    data = {pp.PARAMETERS: {keyword: {"mu": 0.5, "lambda": 0.5}}}
    body_force = -1e-2
    force = 1e-3
    if_spt = True
    solver = LocalSolver(mdg, data, keyword, if_spt, body_force, force)

    # check with a direct computation
    s_dir, u_dir, r_dir = solver.compute_direct()

    # step 1
    sf = solver.compute_sf()

    for tol in tol_array:
        # step 2
        # s0 = solver.compute_s0(sf)
        s0 = solver.compute_s0_cg(sf, tol=tol)
        # solver.check_s0(s0)

        # # step 3
        s, u, r = solver.compute_all(s0, sf)

        # compute the errors
        err_s = solver.compute_error(s, s_dir, solver.Ms)
        err_u = solver.compute_error(u, u_dir, solver.Mu)
        err_r = solver.compute_error(r, r_dir, solver.Mr)

        print("{:.2E}, {:.2E}, {:.2E}".format(err_s, err_u, err_r))

    # export the results
    solver.export(u, r, "tsp", folder)
    solver.export(u_dir, r_dir, "dir", folder)
