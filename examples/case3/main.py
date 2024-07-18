import numpy as np
import scipy.optimize as spo

import porepy as pp
import pygeon as pg

import os
import sys

src_path = os.path.join(__file__, "../../../src")
sys.path.insert(0, os.path.realpath(src_path))

from solver import Solver


def mu(rho, beta=1.5):
    return 1 + (1 + np.square(rho)) ** ((beta - 2) / 2)


def labda(rho, alpha=1):
    return alpha * (1 - mu(rho) / 2)


def g_u(x, gamma=1):
    return gamma / 10 * np.array([x[0] * (1 - x[0]), x[1] * (1 - x[1])])


def force(x, delta=1):
    return delta * np.array(
        [(4 * x[1] - 1) * (4 * x[1] - 3), (4 * x[0] - 1) * (4 * x[0] - 3)]
    )


class LocalSolver(Solver):

    def __init__(self, sd, data, keyword, sptr, g_u_gamma, force_delta):
        self.g_u_gamma = g_u_gamma
        self.force_delta = force_delta

        super().__init__(sd, data, keyword, sptr)

    def get_f(self):
        mass = self.discr_u.assemble_mass_matrix(self.sd)
        bd = self.discr_u.interpolate(self.sd, self.force_delta)

        f = np.zeros(self.dofs[1] + self.dofs[2])
        f[: self.dofs[1]] = mass @ bd
        return f

    @staticmethod
    def get_nat_bc(sd):
        bdry = sd.tags["domain_boundary_faces"]
        return bdry

    def get_g(self):
        # Displacement bcs along the entire bdry.
        nat_bc = self.get_nat_bc(self.sd)

        # return np.zeros(self.discr_s.ndof(sd)), bdry
        return self.discr_s.assemble_nat_bc(self.sd, self.g_u_gamma, nat_bc), nat_bc

    def save_rhs(self):
        self.rhs = np.hstack((self.g_val, self.get_f()))

    def ess_bc(self):
        # No essential bcs in this case
        ess_dof = np.zeros(np.sum(self.dofs), dtype=bool)
        ess_val = np.zeros_like(ess_dof)
        return ess_dof, ess_val


def find_rho(dev_s, beta):
    """
    Given the norm of the deviatoric stress,
    find the norm of the deviatoric strain.
    """
    # Set up nonlinear problem
    func = lambda x: 2 * mu(x, beta) * x - dev_s
    # Initial guess asserts that mu(rho) is close to mu(0)
    x0 = dev_s / (2 * mu(0, beta))
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
    mesh_size = 0.05
    keyword = "elasticity"

    dim = 2
    mdg = pg.unit_grid(dim, mesh_size)
    mdg.compute_geometry()

    nat_bc = LocalSolver.get_nat_bc(mdg.subdomains()[0])
    sptr = pg.SpanningTreeElasticity(mdg, nat_bc)

    # incorporate the parameters
    alpha = 1
    beta = 1.5
    gamma = 0
    delta = 1

    # Create lambda functions for the parametrization
    labda_alpha = lambda x: labda(x, alpha)
    mu_beta = lambda x: mu(x, beta)
    g_u_gamma = lambda x: g_u(x, gamma)
    force_delta = lambda x: force(x, delta)

    data = {pp.PARAMETERS: {keyword: {"mu": 0.5, "lambda": 1}}}
    solver = LocalSolver(mdg, data, keyword, sptr, g_u_gamma, force_delta)

    Pi = solver.discr_s.eval_at_cell_centers(solver.sd)

    max_iters = 30
    tol = 1e-8

    solver.save_rhs()

    for _ in range(max_iters):
        # Solve the system
        s, u, r = solver.compute_direct()

        # Post-process the deviatoric stress and strain
        dev_s = compute_deviatoric_stress(s, Pi)
        rho = np.array([find_rho(s, beta) for s in dev_s])

        # Update the Lamé parameters
        data[pp.PARAMETERS][keyword]["mu"] = mu_beta(rho)
        data[pp.PARAMETERS][keyword]["lambda"] = labda_alpha(rho)

        # Reassemble the mass matrix for the stress
        solver.Ms = solver.discr_s.assemble_mass_matrix(solver.sd, data)
        solver.build_spp()

        # Calculate how well the current solution solves the updated system
        residual = solver.spp @ np.hstack((s, u, r)) - solver.rhs
        res_norm = np.linalg.norm(residual) / np.linalg.norm(solver.rhs)
        print(f"Residual: {res_norm:.2e}")

        if res_norm < tol:
            break
    else:
        print(f"Did not converge in {max_iters} iterations.")

    # compute s0 directly from s and sf
    sf = solver.compute_sf()
    s0 = s - sf
    solver.check_s0(s0)

    solver.export(u, r, "sol", folder)
