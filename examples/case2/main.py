import numpy as np

import porepy as pp
import pygeon as pg

import sys

sys.path.insert(0, "src/")
from solver import Solver


class LocalSolver(Solver):
    def get_f(self):

        fun = lambda _: np.array([0, 0, -1e-2])
        mass = self.discr_u.assemble_mass_matrix(self.sd)
        bd = self.discr_u.interpolate(self.sd, fun)

        f = np.zeros(self.dofs[1] + self.dofs[2])
        f[: self.dofs[1]] = mass @ bd
        return f

    def get_g(self):
        x_min = sd.face_centers[0, :].min()
        b_faces = np.isclose(self.sd.face_centers[0, :], x_min)

        # define the boundary condition
        u_boundary = lambda _: np.array([0, 0, 0])

        return self.discr_s.assemble_nat_bc(self.sd, u_boundary, b_faces)

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

        b_faces = np.tile(
            np.logical_or.reduce((top, bottom, right, front, back)), sd.dim**2
        )

        ess_dof = np.zeros(self.dofs.sum(), dtype=bool)
        ess_dof[: self.dofs[0]] = b_faces

        ess_val = np.zeros(ess_dof.size)

        return ess_dof, ess_val


if __name__ == "__main__":
    # NOTE: difficulty to converge for RBM
    folder = "examples/case2/"
    step_size = 0.25
    keyword = "elasticity"
    tol = 1e-12

    bbox = {"xmin": 0, "xmax": 2, "ymin": 0, "ymax": 0.5, "zmin": 0, "zmax": 0.5}
    domain = pp.Domain(bbox)
    sd = pg.grid_from_domain(domain, step_size, as_mdg=False)
    sd.compute_geometry()

    data = {pp.PARAMETERS: {keyword: {"mu": 0.5, "lambda": 0.5}}}
    solver = LocalSolver(sd, data, keyword)

    # step 1
    sf = solver.compute_sf()

    # step 2
    s0 = solver.compute_s0_cg(sf, rtol=tol)
    solver.check_s0(s0)

    # step 3
    s, u, r = solver.compute_all(s0, sf)

    # check with a direct computation
    s_dir, u_dir, r_dir = solver.compute_direct()

    # compute the errors
    err_s = solver.compute_error(s, s_dir, solver.Ms)
    err_u = solver.compute_error(u, u_dir, solver.Mu)
    err_r = solver.compute_error(r, r_dir, solver.Mr)

    print(err_s, err_u, err_r)

    # export the results
    solver.export(u, r, "tsp", folder)
    solver.export(u_dir, r_dir, "dir", folder)
