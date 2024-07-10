import numpy as np
import scipy.optimize as spo


def mu(rho, beta_0=0.25, beta_1=0.25, beta=1.5):
    return beta_0 + beta_1 * (1 + np.square(rho)) ** (beta / 2 - 1)


def labda(rho, kappa=0.25):
    return kappa - 0.5 * mu(rho)


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


if __name__ == "__main__":
    # Take a random distribution of deviatoric stresses in the cells
    n_cells = 100
    dev_s = 100 * np.random.rand(n_cells)

    # Compute the different rhos
    rho = np.array([find_rho(s) for s in dev_s])

    # Sanity check
    assert np.allclose(dev_s, 2 * mu(rho) * rho)

    # Retrieve Lamé parameters
    Lame_mu = mu(rho)
    Lame_la = labda(rho)
