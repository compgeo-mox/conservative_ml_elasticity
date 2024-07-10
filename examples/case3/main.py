import numpy as np
import scipy.optimize as spo

import porepy as pp
import pygeon as pg

import os
import sys

from true_solution_symbolic import symbolic_u

src_path = os.path.join(__file__, "../../../src")
sys.path.insert(0, os.path.realpath(src_path))

from solver import Solver


class LocalSolver(Solver):

    def get_f(self):
        fun = f_ex
        mass = self.discr_u.assemble_mass_matrix(self.sd)
        bd = self.discr_u.interpolate(self.sd, fun)

        f = np.zeros(self.dofs[1] + self.dofs[2])
        f[: self.dofs[1]] = mass @ bd
        return f

    def get_g(self):
        sd = self.sd

        bdry = sd.tags["domain_boundary_faces"]
        return self.discr_s.assemble_nat_bc(sd, u_ex, bdry), bdry

    def set_rhs(self):
        self.rhs = np.hstack((self.g_val, self.get_f()))

    def ess_bc(self):
        ess_dof = np.zeros(np.sum(self.dofs), dtype=bool)
        ess_val = np.zeros_like(ess_dof)
        return ess_dof, ess_val


def mu(rho, beta=0.75e4):
    return beta * (1 + (1 + np.square(rho)) ** (-1 / 2))


def labda(rho, beta=0.75e4):
    return beta * (1 - 2 * mu(rho))


def find_rho(dev_s):
    """
    Given the norm of the deviatoric stress,
    find the norm of the deviatoric strain.
    """
    # Set up nonlinear problem
    func = lambda x: 2 * mu(x) * x - dev_s
    # Initial guess asserts that mu(rho) is close to mu(0)
    x0 = dev_s / (2 * mu(0))
    # Use scipy to find root
    result = spo.root_scalar(func, x0=x0)
    # assert result.converged

    return result.root


def compute_deviatoric_stress(s, Pi):
    # Extract the stress per cell
    s_cc = Pi @ s
    s_cc = np.reshape(s_cc, (-1, solver.sd.num_cells), order="C")
    s_cc = s_cc[[0, 1, 3, 4]]  # components xz (2) and yz (5) are zero

    trace = s_cc[0] + s_cc[-1]
    dev_s = s_cc - np.outer(np.array([1, 0, 0, 1]), trace) / 2

    return np.linalg.norm(dev_s, axis=0)


if __name__ == "__main__":
    folder = os.path.dirname(os.path.abspath(__file__))
    mesh_size = 0.08
    keyword = "elasticity"

    dim = 2
    mdg = pg.unit_grid(dim, mesh_size)
    mdg.compute_geometry()

    u_ex, f_ex = symbolic_u()

    data = {pp.PARAMETERS: {keyword: {"mu": 0.5, "lambda": 1}}}
    if_spt = True
    solver = LocalSolver(mdg, data, keyword, if_spt)

    Pi = solver.discr_s.eval_at_cell_centers(solver.sd)

    max_iters = 100
    tol = 1e-10

    solver.set_rhs()

    for _ in range(max_iters):
        s, u, r = solver.compute_direct()

        dev_s = compute_deviatoric_stress(s, Pi)
        rho = np.array([find_rho(s) for s in dev_s])

        data[pp.PARAMETERS][keyword]["mu"] = mu(rho)
        data[pp.PARAMETERS][keyword]["lambda"] = labda(rho)

        solver.Ms = solver.discr_s.assemble_mass_matrix(solver.sd, data)
        solver.build_spp()

        # Calculate how well the current solution solves the updated system
        residual = solver.spp @ np.hstack((s, u, r)) - solver.rhs
        res = np.linalg.norm(residual) / np.linalg.norm(solver.rhs)
        print("Residual: {:.2e}".format(res))

        if res < tol:
            break

    sf = solver.compute_sf()

    # compute s0 directly from s and sf
    s0 = s - sf

    s0_v2 = solver.S0(s)
    print(np.linalg.norm(s0 - s0_v2))
    # solver.check_s0(s0)

    solver.export(u, r, "sol", folder)
