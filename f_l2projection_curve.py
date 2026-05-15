
import numpy as np


def _curved_G(xi, eta):
    """Jacobian-style denominator (matches NS_Inputfile.exact_solution_curve)."""
    xm1 = xi - 1.0
    return (
        -1.0
        + 4.0 * xi**2
        - 11.0 * xi**3
        + 10.0 * xi**4
        - 3.0 * xi**5
        + eta**3 * xm1**2 * (1.0 - 2.0 * xi + 3.0 * xi**2)
        - 3.0 * eta**2 * xm1**2 * (1.0 - 2.0 * xi - xi**2 + xi**3)
        + eta * (1.0 - 10.0 * xi - 2.0 * xi**2 + 30.0 * xi**3 - 27.0 * xi**4 + 6.0 * xi**5)
    )


def _curved_H(xi, eta):
    """H = -G; Mathematica prints this as the ( ... )^(-2) factor."""
    return -_curved_G(xi, eta)


def _Gja(xi, eta):
    """First line of DF^{-T} style factor: 1 - 4 xi^2 + ... + eta(...)."""
    xm1 = xi - 1.0
    return (
        1.0
        - 4.0 * xi**2
        + 11.0 * xi**3
        - 10.0 * xi**4
        + 3.0 * xi**5
        - eta**3 * xm1**2 * (1.0 - 2.0 * xi + 3.0 * xi**2)
        + 3.0 * eta**2 * xm1**2 * (1.0 - 2.0 * xi - xi**2 + xi**3)
        + eta * (-1.0 + 10.0 * xi + 2.0 * xi**2 - 30.0 * xi**3 + 27.0 * xi**4 - 6.0 * xi**5)
    )


def _p_phys_num(xi, eta):
    """156*e - 8*(...) + exp(xi)*eta*(...) — pressure numerator from exact_solution_l2_curve."""
    ex = np.exp(xi)
    return (
        156.0 * np.e
        - 8.0 * (53.0 - 57.0 * eta + 57.0 * eta**2)
        + ex
        * eta
        * (
            -2.0 * eta**2 * xi * (2.0 - 5.0 * xi + 2.0 * xi**2 + xi**3)
            + eta**3 * xi * (2.0 - 5.0 * xi + 2.0 * xi**2 + xi**3)
            - 12.0 * (38.0 - 38.0 * xi + 19.0 * xi**2 - 6.0 * xi**3 + xi**4)
            + eta * (456.0 - 454.0 * xi + 223.0 * xi**2 - 70.0 * xi**3 + 13.0 * xi**4)
        )
    )


def _A_exact(xi, eta):
    """Q(eta) * exp(xi) bracket from unit-square p_exact (same structure as forcing_function)."""
    ex = np.exp(xi)
    return 228.0 + 2.0 * (-114.0 - eta + eta**2) * xi + (114.0 + 5.0 * eta - 5.0 * eta**2) * xi**2 + 2.0 * (-18.0 - eta + eta**2) * xi**3 + (6.0 - eta + eta**2) * xi**4


def forcing_function_l2projection_curve(xi, eta,nu=1):
    """
    Return (f1, f2) = Force_L2Projection_CurveDomain from the notebook.

    Equivalent to: Rational[-1,81] * H^(-2) * G^(-1) * N  per component,
    with H = -G, G = _curved_G.
    """
    xm1 = xi - 1.0
    G = _curved_G(xi, eta)
    H = _curved_H(xi, eta)
    Gja = _Gja(xi, eta)

    ex = np.exp(xi)
    E = np.e

    # --- shared pieces ---
    poly_xi_eta = (
        -1.0
        + xi**2
        - 3.0 * eta**2 * (1.0 - 4.0 * xi + 3.0 * xi**2)
        + eta**3 * (1.0 - 4.0 * xi + 3.0 * xi**2)
        + eta * (1.0 - 10.0 * xi + 9.0 * xi**2)
    )

    bracket_A = -228.0 + ex * _A_exact(xi, eta)

    dpx_physical = (
        1.0
        - 10.0 * xi
        - 2.0 * xi**2
        + 30.0 * xi**3
        - 27.0 * xi**4
        + 6.0 * xi**5
        + 3.0 * eta**2 * xm1**2 * (1.0 - 2.0 * xi + 3.0 * xi**2)
        - 6.0 * eta * xm1**2 * (1.0 - 2.0 * xi - xi**2 + xi**3)
    )

    pnum = _p_phys_num(xi, eta)

    inner_vel = (
        ex
        * (eta - 1.0)
        * eta
        * (
            12.0 * xm1**2 * xi**2
            - eta * (2.0 - 8.0 * xi + xi**2 + 6.0 * xi**3 + xi**4)
            + eta**2 * (2.0 - 8.0 * xi + xi**2 + 6.0 * xi**3 + xi**4)
        )
    )

    geom_ps = (
        xi * (8.0 - 33.0 * xi + 40.0 * xi**2 - 15.0 * xi**3)
        + 4.0 * eta**3 * (-1.0 + 4.0 * xi - 6.0 * xi**2 + 3.0 * xi**3)
        - 3.0 * eta**2 * (-4.0 + 8.0 * xi + 3.0 * xi**2 - 12.0 * xi**3 + 5.0 * xi**4)
        + 2.0 * eta * (-5.0 - 2.0 * xi + 45.0 * xi**2 - 54.0 * xi**3 + 15.0 * xi**4)
    )

    # --- f1 numerator (notebook inner expression before -1/81 H^-2 G^-1) ---
    u1_num = (
        ex
        * (eta - 1.0)
        * eta
        * xm1**2
        * xi
        * (
            -2.0 * (-2.0 + xi) * xi**2
            + eta**2 * (-2.0 + xi - 4.0 * xi**2 + xi**3)
            + eta * (2.0 - xi - 8.0 * xi**2 + 3.0 * xi**3)
        )
    )

    term1 = -27.0 * u1_num * Gja**2

    term2 = (
        -3.0
        * poly_xi_eta
        * (-2.0 * (-1.0 + 2.0 * eta) * bracket_A * Gja)
        - dpx_physical * pnum
    )

    term3 = (
        -3.0
        * xi
        * (1.0 - 6.0 * eta * xm1**2 + 3.0 * eta**2 * xm1**2 - 5.0 * xi + 3.0 * xi**2)
        * (inner_vel * Gja + geom_ps * pnum)
    )

    N1 = term1 + term2 + term3
    f1 = -N1 / (81.0 * H**2 * G)

    # --- f2 numerator ---
    u2_inner = (
        2.0 * xm1**2 * (1.0 + xi)
        + eta**4 * xm1**2 * (-2.0 - 3.0 * xi + 3.0 * xi**2)
        - eta**3 * xm1**2 * (-4.0 - 15.0 * xi + 9.0 * xi**2)
        + eta * (-4.0 + 13.0 * xi - 14.0 * xi**2 + 10.0 * xi**3 - 3.0 * xi**4)
        + eta**2 * (-4.0 - 19.0 * xi + 56.0 * xi**2 - 44.0 * xi**3 + 9.0 * xi**4)
    )

    v2_num = ex * (eta - 1.0) * eta * xm1 * xi**2 * u2_inner

    t1_2 = 27.0 * v2_num * Gja**2

    t2_2 = (
        -3.0
        * (2.0 + 2.0 * eta - xi)
        * xi
        * (-2.0 * (-1.0 + 2.0 * eta) * bracket_A * Gja)
        - dpx_physical * pnum
    )

    t3_2 = (3.0 - 3.0 * xi**2) * (inner_vel * Gja + geom_ps * pnum)

    N2 = t1_2 + t2_2 + t3_2
    f2 = -N2 / (81.0 * H**2 * G)

    return np.array([f1, f2])


def force_l2projection_curve_components(xi, eta):
    """Same as force_l2projection_curve but returns tuple (f1, f2)."""
    f = forcing_function_l2projection_curve(xi, eta)
    return float(f[0]), float(f[1])
