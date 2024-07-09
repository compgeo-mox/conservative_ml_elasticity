import numpy as np
import scipy.optimize as spo


def mu(rho, beta_0=0.25, beta_1=0.25, beta=1.5):
    return beta_0 + beta_1 * (1 + np.square(rho)) ** (beta / 2 - 1)


def labda(rho, kappa=0.25):
    return kappa - 0.5 * mu(rho)


if __name__ == "__main__":
    """ Given a deviatoric stress, find the Lamé parameters """

    # Given norm of the deviatoric stress 
    dev_s = np.random.rand()

    # Initial guess asserts that mu(rho) is close to mu(0)
    x0 = dev_s / (2 * mu(0))

    # Set up nonlinear problem
    nonlinear_func = lambda x: 2 * mu(x) * x - dev_s

    # Use scipy to find root
    iterator = spo.root_scalar(nonlinear_func, x0=x0)
    rho = iterator.root

    assert iterator.converged
    assert np.allclose(dev_s, 2 * mu(rho) * rho)

    print("Converged in {} iterations".format(iterator.iterations))
    print("mu: {}".format(mu(rho)))
    print("lambda: {}".format(labda(rho)))
