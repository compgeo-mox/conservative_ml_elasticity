import numpy as np

import porepy as pp
import pygeon as pg

import sys

sys.path.insert(0, "src/")
from solver import Solver


class LocalSolver(Solver):
    def get_f(self):
        return np.zeros(self.dofs[1] + self.dofs[2])

    def get_g(self):
        b_faces = np.isclose(self.sd.face_centers[1, :], 0)

        # define the boundary condition
        u_boundary = lambda _: np.array([0, 0, 0])

        return self.discr_s.assemble_nat_bc(self.sd, u_boundary, b_faces)

    def ess_bc(self):
        sd = self.sd
        # select the faces for the essential boundary conditions
        top = np.isclose(sd.face_centers[1, :], 1)
        left = np.isclose(sd.face_centers[0, :], 0)
        right = np.isclose(sd.face_centers[0, :], 1)
        b_faces = np.tile(np.logical_or.reduce((left, right, top)), sd.dim**2)

        # only on the top boundary for the y-component we actually impose something not null
        top_y = np.hstack((np.zeros(2 * top.size, dtype=bool), top, top))

        # compute the actual of the imposed force, which is 1 downward
        val = -np.array(np.sum(sd.cell_faces, axis=1)).flatten() * sd.face_areas

        # insert the values in the correct position
        b_val = np.zeros_like(b_faces, dtype=float)
        b_val[top_y] = np.hstack([np.zeros(2 * top.size)] + 2 * [val])[top_y]

        ess_dof = np.zeros(self.dofs.sum(), dtype=bool)
        ess_dof[: self.dofs[0]] = b_faces

        ess_val = np.zeros(ess_dof.size)
        ess_val[: self.dofs[0]] = b_val

        return ess_dof, ess_val


if __name__ == "__main__":
    # NOTE: difficulty to converge for RBM
    folder = "examples/case1/"
    step_size = 0.1
    keyword = "elasticity"
    tol = 1e-12

    dim = 2
    sd = pg.unit_grid(dim, step_size, as_mdg=False)
    sd.compute_geometry()

    data = {pp.PARAMETERS: {keyword: {"mu": 0.5, "lambda": 0.5}}}
    solver = LocalSolver(sd, data, keyword)

    # # step 1
    # sf = solver.compute_sf()

    # # step 2
    # s0 = solver.compute_s0_cg(sf, rtol=tol)
    # solver.check_s0(s0)

    # # step 3
    # s, u, r = solver.compute_all(s0, sf)

    # check with a direct computation
    s_dir, u_dir, r_dir = solver.compute_direct()

    # # compute the errors
    # err_s = solver.compute_error(s, s_dir, solver.Ms)
    # err_u = solver.compute_error(u, u_dir, solver.Mu)
    # err_r = solver.compute_error(r, r_dir, solver.Mr)

    # print(err_s, err_u, err_r)

    # export the results
    # solver.export(u, r, "tsp", folder)
    solver.export(u_dir, r_dir, "dir", folder)
