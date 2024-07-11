import numpy as np
import sympy as sp
from sympy.physics.vector import ReferenceFrame


def symbolic_u_and_f():
    """
    Compute a forcing term according to a given displacement
    field with constant Lamé parameters
    """

    dim = 2
    R = ReferenceFrame("R")
    x, y, _ = R.varlist

    # define the displacement
    u_x = sp.sin(x) * sp.cos(y) * sp.exp(x * y)
    u_y = sp.cos(x) * sp.sin(y) * sp.exp(-x * y)
    u = sp.Matrix([u_x, u_y])

    # Compute the strain
    epsilon = sp.Matrix([sp.diff(u, x_i).T for x_i in (x, y)])
    epsilon += epsilon.T
    epsilon /= 2

    dev = epsilon - 0.5 * epsilon.trace() * sp.Identity(2)
    rho = sp.Matrix(dev).norm()

    beta = 0.25
    mu = beta  # * (1 + (1 + rho**2) ** (-0.5))
    labda = beta * (1 - 2 * mu)

    sigma = 2 * mu * epsilon + labda * epsilon.trace() * sp.Identity(2)
    sigma = sp.Matrix(sigma)

    f_u = -matrix_divergence(sigma, R)

    u_ex = lambdify_vec(u, R)
    f_u_ex = lambdify_vec(f_u, R)

    return u_ex, f_u_ex


def lambdify_vec(func, R):
    x, y, _ = R.varlist
    lamb = sp.lambdify([x, y], func)
    return lambda pt: lamb(*pt[:2]).ravel()


def matrix_divergence(sigma, R):
    x, y, z = R.varlist
    return sp.diff(sigma[:, 0], x) + sp.diff(sigma[:, 1], y)
