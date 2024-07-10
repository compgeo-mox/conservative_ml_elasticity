import sympy as sp
from sympy.physics.vector import ReferenceFrame


def symbolic_u():
    dim = 2
    R = ReferenceFrame("R")
    x, y, _ = R.varlist

    # define the displacement
    u_x = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    u_y = u_x
    u = sp.Matrix([u_x, u_y])

    epsilon = sp.Matrix([sp.diff(u, x_i).T for x_i in (x, y)])
    epsilon += epsilon.T
    epsilon /= 2

    dev = epsilon - 0.5 * epsilon.trace() * sp.Identity(2)

    rho = sp.Matrix(dev).norm()

    beta = 0.75e4
    mu = 2 * beta  # beta * (1 + (1 + rho**2) ** (-0.5))
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
