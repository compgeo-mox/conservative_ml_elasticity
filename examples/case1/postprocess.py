import numpy as np

import porepy as pp
import pygeon as pg

import sys

sys.path.insert(0, "src/")
from solver import Solver


class LocalSolver(Solver):
    def __init__(self, sd, data, keyword, spanning_tree, body_force, force):
        self.body_force = body_force
        self.force = force
        super().__init__(sd, data, keyword, spanning_tree)

    def get_f(self):
        fun = lambda _: np.array([0, self.body_force, 0])
        mass = self.discr_u.assemble_mass_matrix(self.sd)
        bd = self.discr_u.interpolate(self.sd, fun)

        f = np.zeros(self.dofs[1] + self.dofs[2])
        f[: self.dofs[1]] = mass @ bd
        return f

    def get_g(self):
        b_faces = np.isclose(self.sd.face_centers[1, :], 0)

        # define the boundary condition
        u_boundary = lambda _: np.array([0, 0, 0])

        return self.discr_s.assemble_nat_bc(self.sd, u_boundary, b_faces), b_faces

    def ess_bc(self):
        sd = self.sd

        # select the faces for the essential boundary conditions
        top = np.isclose(sd.face_centers[1, :], 1)
        left = np.isclose(sd.face_centers[0, :], 0)
        right = np.isclose(sd.face_centers[0, :], 1)
        ess_dof = np.tile(np.logical_or.reduce((left, right, top)), sd.dim**2)

        # function for the essential boundary conditions
        val = np.array([[0, 0, 0], [0, self.force, 0]])
        fct = lambda pt: val if np.isclose(pt[1], 1) else 0 * val

        # interpolate the essential boundary conditions
        ess_val = -self.discr_s.interpolate(sd, fct)

        return ess_dof, ess_val


if __name__ == "__main__":
    # NOTE: difficulty to converge for RBM
    folder = "examples/case1/"
    step_size = 0.05
    keyword = "elasticity"
    tol = 1e-8

    dim = 2
    sd = pg.unit_grid(dim, step_size, as_mdg=False)
    sd.compute_geometry()
    
    ntrain = 150
    j = ntrain + int(input("Sample to visualize: "))
    outputs = np.load("outputs-case1.npz") 
    data = {pp.PARAMETERS: {keyword: {"mu": outputs['params'][j, 0], "lambda": outputs['params'][j, 1]}}}
    body_force = outputs['params'][j, 3]
    force = outputs['params'][j, 2]
    solver = LocalSolver(sd, data, keyword, False, body_force, force)

    # step 1
    sf = solver.compute_sf()

    for which in [0,1,2]:

        # step 2
        if(which == 0):
            s0 = solver.compute_s0_cg(sf, tol=tol)
            s0gt = s0 + 0.0
        elif(which == 1):
            s0 = outputs['s0PODNN'][j] - sf
            print(np.linalg.norm(s0gt-s0)/np.linalg.norm(s0gt+sf))
        else:
            s0 = outputs['PODNN'][j] - sf
            print(np.linalg.norm(s0gt-s0)/np.linalg.norm(s0gt+sf))
        # # step 3
        s, u, r = solver.compute_all(s0, sf)

        # export the results
        strings = ["gt", "cnsv", "bbx"]
        filename = "case1j%d_%s" % (j-ntrain, strings[which])
        solver.export(u, r, filename, folder)
