#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:59:26 2026

@author: raminpahnabi
"""
import sys
import os
import numpy as np
from pathlib import Path

os.environ["SWEEPS_API_PATH"] = "/Users/raminpahnabi/Documents/BYU/sweeps/build/src/api"

from sweeps_path import ensure_sweeps_api_on_path

PROJECT_ROOT = Path(__file__).resolve().parent
sweepspath = ensure_sweeps_api_on_path()
sys.path.append(str(PROJECT_ROOT / 'HWs'))
sys.path.append(str(PROJECT_ROOT / 'Required'))

import splines as spline
import Gaussian_Quadrature_2D_Solution as gq_nD

KINEMATIC_VISCOSITY = 0.1


# ############################## Option No.4 Stokes ################################
def forcing_function(x, y, nu, sigma = 0):
    """
    Force vector for the manufactured Stokes/Brinkman solution.

    Parameters
    ----------
    x, y : float
        Spatial coordinates.
    nu : float or callable
        Kinematic viscosity. If callable, use nu(x, y).
    sigma : float or callable
        Darcy coefficient. If callable, use sigma(x, y).

    Returns
    -------
    (f1, f2) : tuple of floats
    """

    nu_val = nu(x, y) if callable(nu) else nu
    sigma_val = sigma(x, y) if callable(sigma) else sigma

    f1 = np.exp(x) * (
        (y - 1) * y * (
            2 * (y - 1) * y
            - 8 * x * (y - 1) * y
            + 6 * x**3 * (-4 - y + y**2)
            + x**2 * (12 - y + y**2)
            + x**4 * (12 - y + y**2)
        )
        - 2 * (
            2 * y * (1 - 3 * y + 2 * y**2)
            - 8 * x * y * (1 - 3 * y + 2 * y**2)
            + 6 * x**3 * (2 - 3 * y - 3 * y**2 + 2 * y**3)
            + x**2 * (-6 + 13 * y - 3 * y**2 + 2 * y**3)
            + x**4 * (-6 + 13 * y - 3 * y**2 + 2 * y**3)
        ) * nu_val
        + 2 * (x - 1)**2 * x**2 * (y - 1) * y * (2 * y - 1) * sigma_val
    )

    f2 = (
        np.exp(x) * x * (2 - 5 * x + 2 * x**2 + x**3) * (y - 1) * y * (2 * y - 1)
        + (2 * y - 1) * (
            -456
            + np.exp(x) * (
                456
                + x**2 * (228 + 5 * y - 5 * y**2)
                + 2 * x * (-228 - y + y**2)
                + 2 * x**3 * (-36 - y + y**2)
                + x**4 * (12 - y + y**2)
            )
        )
        + np.exp(x) * (
            -6 * (y - 1)**2 * y**2
            + x * (4 - 24 * y + 18 * y**2 + 12 * y**3 - 6 * y**4)
            + x**4 * (2 - 12 * y + 13 * y**2 - 2 * y**3 + y**4)
            + 2 * x**3 * (2 - 12 * y + 17 * y**2 - 10 * y**3 + 5 * y**4)
            + x**2 * (-10 + 60 * y - 41 * y**2 - 38 * y**3 + 19 * y**4)
        ) * nu_val
        - np.exp(x) * (x - 1) * x * (-2 + x * (3 + x)) * (y - 1)**2 * y**2 * sigma_val
    )

    return f1, f2


######################## Option No.4_ NavierStokes #####################
#AI
def forcing_function_ns(x, y, nu, sigma=0):
    ex = np.exp(x)
    e2x = np.exp(2*x)

    f1 = ex * (
            4*ex*(x-1)**3 * x**3 * (-2 + 3*x + x**2) * (1 - 2*y)**2 * (y-1)**2 * y**2
            - 2*ex*(x-1)**3 * x**3 * (-2 + 3*x + x**2) * (y-1)**2 * y**2 * (1 - 6*y + 6*y**2)
            - 2*nu*( -1 + 2*y ) * (
                2*(y-1)*y
                - 8*x*(y-1)*y
                + 6*x**3*(-2 - y + y**2)
                + x**2*(6 - y + y**2)
                + x**4*(6 - y + y**2)
            )
            + (y-1)*y * (
                2*(y-1)*y
                - 8*x*(y-1)*y
                + 6*x**3*(-4 - y + y**2)
                + x**2*(12 - y + y**2)
                + x**4*(12 - y + y**2)
            )
        )

    f2 = (ex * x * (2 - 5*x + 2*x**2 + x**3) * (y-1)*y * (-1 + 2*y)
        + 2*e2x*(x-1)**2 * x**2 * (-2 + 3*x + x**2)**2 * (y-1)**3 * y**3 * (-1 + 2*y)
        - 2*e2x*(x-1)**2 * x**2 * (2 - 8*x + x**2 + 6*x**3 + x**4) * (y-1)**3 * y**3 * (-1 + 2*y)
        - ex*nu * (
            -2*(x-1)*x*(y-1)**2 * y**2
            - 2*(3 + 2*x)*(x-1 + x**2)*(y-1)**2 * y**2
            - x*(3 + x)*(-2 + 3*x + x**2)*(y-1)**2 * y**2
            - 2*(x-1)*x*(-2 + 3*x + x**2)*(1 - 6*y + 6*y**2)
        )
        + (-1 + 2*y) * (
            -456
            + ex * (
                456
                + x**2*(228 + 5*y - 5*y**2)
                + 2*x*(-228 - y + y**2)
                + 2*x**3*(-36 - y + y**2)
                + x**4*(12 - y + y**2)
            )
        )
    )

    return f1, f2

def exact_solution(x, y):
    u1 = 2 * np.exp(x) * (-1 + x)*(-1 + x)* x**2 * (y**2 - y) * (-1 + 2 * y)
    u2 = -np.exp(x) * (-1 + x) * x * (-2 + x * (3 + x)) * (-1 + y)**2 * y**2
    return u1, u2

def exact_solution_l2(x, y):
    p = -424+156*np.exp(1)+(-y+y**2)*(-456+np.exp(x)*(456+x**2*(228-5*(-y+y**2))+2*x*(-228+(-y+y**2))+2*x**3*(-36+(-y+y**2))+x**4*(12+(-y+y**2))))
    return p


def boundary_value_function(x, y):
    return exact_solution(x, y)

max_knot_xi = 0.1
max_knot_eta = 5
min_knot = 0
degree1 = 2
degree2 = 2
nelem1 = 2
nelem2 = 2
degs = [degree1,degree2]
kv1_init = list(np.linspace(0,max_knot_xi,nelem1+1))
kv2_init = list(np.linspace(0,max_knot_eta,nelem2+1))
kv1 = spline.KnotVector([0]*degree1 + kv1_init + [max_knot_xi]*(degree1), 1e-9)
kv2 = spline.KnotVector([0]*degree2 + kv2_init + [max_knot_eta]*(degree2), 1e-9)

unitkv1_init = list(np.linspace(0,1,nelem1+1))
unitkv2_init = list(np.linspace(0,1,nelem2+1))
unitkv1 = spline.KnotVector([0]*degree1 + unitkv1_init + [1]*(degree1), 1e-9)
unitkv2 = spline.KnotVector([0]*degree2 + unitkv2_init + [1]*(degree2), 1e-9)


######## Square domain
cpts = spline.grevillePoints( unitkv1, unitkv2, degree1, degree2 )#/ max_knot # make the domain a unit square   

######## Curve domain
# cpts = np.array([
#     [0.5, 0.5, 1.0,   0.25, 0.25, 1.0,   0.0, 0.0, 1.0 ], 
#     [0.0, 0.5, 0.5,   0.0, 0.75, 0.75,   0.0, 1.0, 1.0 ]
# ])

n_quad   = max(degree1+1, degree2+2)+1
interval = [0, 1]
quad     = gq_nD.GaussQuadrature2D(n_quad, n_quad, interval, interval)
quad_1D  = gq_nD.GaussQuadrature1D(n_quad, start_pt=interval[0], end_pt=interval[1])
gamma    =  20 * max(degree1, degree2)**3
ifID = True
####



######## Geometry switch  
USE_CURVED_GEOMETRY = False

# def map_xi_eta_to_xy(xi, eta):
#     x = -3*eta + 3*xi**2 + 3*xi**2*eta - xi**3

#     y = ( 3*xi - 3*xi*eta + 9*xi*eta**2 - 3*xi*eta**3
#           + 15*xi**2*eta - 18*xi**2*eta**2 + 6*xi**2*eta**3
#           - xi**3 - 9*xi**3*eta + 9*xi**3*eta**2 - 3*xi**3*eta**3 )
#     return x, y

# def make_cpts(kv1_, kv2_, deg1_, deg2_, min_knot, max_knot_xi, max_knot_eta):                                                                                                        
#     cpts_sq = spline.grevillePoints(kv1_, kv2_, deg1_, deg2_)                    
#     if not USE_CURVED_GEOMETRY:                                                    
#         return cpts_sq 
                                                          
#     cpts_ = cpts_sq.copy()
#     for i in range(cpts_.shape[1]):
#         xi_cp  = cpts_sq[0, i]
#         eta_cp = cpts_sq[1, i]
    
#         # if your knot range is [min_knot,max_knot]=[0,1], this is just xi_cp, eta_cp
#         xi_n  = (xi_cp  - min_knot) / (max_knot_xi - min_knot)
#         eta_n = (eta_cp - min_knot) / (max_knot_eta - min_knot)
    
#         X, Y = map_xi_eta_to_xy(xi_n, eta_n)
#         cpts_[0, i] = X
#         cpts_[1, i] = Y
        
#     return cpts_                                                      
                                                               

# cpts = make_cpts(unitkv1_init, unitkv2_init, degree1, degree2, min_knot, max_knot_xi, max_knot_eta)

# d_next = relaxation * d_picard + (1.0 - relaxation) * d_prev 