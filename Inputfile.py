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

import f_stokes_curve as fs

KINEMATIC_VISCOSITY = 0.1

########################### Stokes #########################
def forcing_function_s_1(x, y, nu, sigma = 0):

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


################### NavierStokes with AI ###############
def forcing_function_ns_1(x, y, nu, sigma=0):
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

############# CURVE DOMAIN Section With AI ################

forcing_function_stokes_curve = fs.forcing_function_stokes_curve

def exact_solution_curve(xi, eta):

    ex = np.exp(xi)

    # Common geometric term (same as before)
    G = (
        -1 + 4*xi**2 - 11*xi**3 + 10*xi**4 - 3*xi**5
        + eta**3 * (xi - 1)**2 * (1 - 2*xi + 3*xi**2)
        - 3*eta**2 * (xi - 1)**2 * (1 - 2*xi - xi**2 + xi**3)
        + eta * (1 - 10*xi - 2*xi**2 + 30*xi**3 - 27*xi**4 + 6*xi**5)
    )


    u1_num = (
        ex * (eta - 1) * eta * (xi - 1)**2 * xi * (
            -2 * (-2 + xi) * xi**2
            + eta**2 * (-2 + xi - 4*xi**2 + xi**3)
            + eta * (2 - xi - 8*xi**2 + 3*xi**3)
        )
    )
    
    u1 = u1_num / (3 * G)


    u2_num = (
        ex * (eta - 1) * eta * (xi - 1) * xi**2 * (
            2 * (xi - 1)**2 * (1 + xi)
            + eta**4 * (xi - 1)**2 * (-2 - 3*xi + 3*xi**2)
            - eta**3 * (xi - 1)**2 * (-4 - 15*xi + 9*xi**2)
            + eta * (-4 + 13*xi - 14*xi**2 + 10*xi**3 - 3*xi**4)
            + eta**2 * (-4 - 19*xi + 56*xi**2 - 44*xi**3 + 9*xi**4)
        )
    )

    u2 = - u2_num / (3 * G)

    return u1, u2

def exact_solution_l2_curve(xi, eta):

    ex = np.exp(xi)

    # Common geometric denominator term (same as before)
    G = (
        -1 + 4*xi**2 - 11*xi**3 + 10*xi**4 - 3*xi**5
        + eta**3 * (xi - 1)**2 * (1 - 2*xi + 3*xi**2)
        - 3*eta**2 * (xi - 1)**2 * (1 - 2*xi - xi**2 + xi**3)
        + eta * (1 - 10*xi - 2*xi**2 + 30*xi**3 - 27*xi**4 + 6*xi**5)
    )

    numerator = (
        156*np.e
        - 8*(53 - 57*eta + 57*eta**2)
        + ex * eta * (
            -2*eta**2 * xi * (2 - 5*xi + 2*xi**2 + xi**3)
            + eta**3 * xi * (2 - 5*xi + 2*xi**2 + xi**3)
            - 12*(38 - 38*xi + 19*xi**2 - 6*xi**3 + xi**4)
            + eta*(456 - 454*xi + 223*xi**2 - 70*xi**3 + 13*xi**4)
        )
    )

    ps = - numerator / (9 * G)

    return ps

################### From John Evans ################
def exact_solution_1(x, y):
    u1 = 2 * np.exp(x) * (-1 + x)*(-1 + x)* x**2 * (y**2 - y) * (-1 + 2 * y)
    u2 = -np.exp(x) * (-1 + x) * x * (-2 + x * (3 + x)) * (-1 + y)**2 * y**2
    return u1, u2

def exact_solution_l2_1(x, y):
    p = -424+156*np.exp(1)+(-y+y**2)*(-456+np.exp(x)*(456+x**2*(228-5*(-y+y**2))+2*x*(-228+(-y+y**2))+2*x**3*(-36+(-y+y**2))+x**4*(12+(-y+y**2))))
    return p

def forcing_function_l2projection_1(x, y, nu=1, sigma=0):
    """
    # L2-projection mixed problem: u + grad(p) = f, div(u) = 0.
    # f = u_exact(x,y) + grad(p_exact(x,y)).
    """
    # Step 1: exact velocity contribution
    u1, u2 = exact_solution_1(x, y)

    # Step 2: grad(p_exact) computed analytically from exact_solution_l2.
    # p = -424 + 156*exp(1) + Q*(-456 + exp(x)*A),  Q = y^2 - y
    # A = 456 + x^2*(228+5y-5y^2) + 2x*(-228-y+y^2) + 2x^3*(-36-y+y^2) + x^4*(12-y+y^2)
    ex = np.exp(x)
    Q  = y**2 - y                 

    A = (456
          + x**2 * (228 + 5*y - 5*y**2)
          + 2*x  * (-228 - y + y**2)
          + 2*x**3 * (-36  - y + y**2)
          + x**4  * ( 12  - y + y**2))

    dA_dx = (2*x   * (228 + 5*y - 5*y**2)
              + 2   * (-228 - y + y**2)
              + 6*x**2 * (-36  - y + y**2)
              + 4*x**3 * ( 12  - y + y**2))

    dA_dy = (x**2 * (5 - 10*y)
              + 2*x  * (-1 + 2*y)
              + 2*x**3 * (-1 + 2*y)
              + x**4  * (-1 + 2*y))

    # dp/dx = Q * exp(x) * (A + dA/dx)   
    dp_dx = Q * ex * (A + dA_dx)

    # dp/dy = dQ/dy*(-456 + exp(x)*A) + Q*exp(x)*dA/dy,  dQ/dy = 2y-1
    dp_dy = (2*y - 1) * (-456 + ex * A) + Q * ex * dA_dy

    # f = u_exact + grad(p_exact)
    f1 = u1 + dp_dx
    f2 = u2 + dp_dy
    return f1, f2

def boundary_value_function_1(x, y):
    return exact_solution_1(x, y)


####################### Easiest one #####################
def exact_solution_0(x, y):
    u1 = y
    u2 = x
    return u1, u2

def exact_solution_l2_0(x, y):
    p = 0
    return p

def forcing_function_l2projection_0(x, y, nu=1, sigma=0):
    u1, u2 = exact_solution_0(x, y)
    
    f1 = u1 
    f2 = u2 
    return f1, f2

def boundary_value_function_0(x, y):
    return exact_solution_0(x, y)

################# CAVITY, Option 2. ######################
def forcing_function_cavity_2(x, y, nu=1, sigma=0):
    return 0.0, 0.0

def exact_solution_cavity_2(x, y): 
    # r     = np.sqrt(x**2 + y**2)
    # theta = np.arctan2(y, x)
    
    # # Exact solution using the provided expressions in polar coordinates
    # lambda_value = 0.54448373678246
    # phi = (np.sin((1 + lambda_value) * theta) * np.cos(lambda_value * theta) / (1 + lambda_value) -
    #        np.cos((1 + lambda_value) * theta) / (1 - lambda_value))
    
    # # Calculating the components of the velocity field
    # u_r = r**lambda_value * ((1 + lambda_value) * np.sin(theta) * phi + np.cos(theta) * phi)
    # u_theta = r**lambda_value * (-(1 + lambda_value) * np.cos(theta) * phi + np.sin(theta) * phi)
    
    # return u_r, u_theta
    return 0.0, 0.0

def exact_solution_l2_cavity_2(x, y):  
    return 0.0

def boundary_value_function_cavity_2(x, y):  
    if abs(y - 1.0) < 1e-10:
        return (1.0, 0.0)
    return (0.0, 0.0)

######### Option 3: Confined Jet Impingement #################
def forcing_function_jet_3(x, y, nu=1, sigma=0): 
    return 0.0, 0.0

def exact_solution_jet_3(x, y): 
    return 0.0, 0.0

def exact_solution_l2_jet_3(x, y):  
    return 0.0

def boundary_value_function_jet_3(x, y):  # inflow at top (x<=D/2=0.5), no-slip elsewhere
    # if abs(y - 1.0) < 1e-10 and x <= 0.5 + 1e-10: 
    if abs(y - 1.0) < 1e-10 and x <= 0.5 + 1e-10: 
        return (0.0, -1.0)   # downward jet U=1 
    return (0.0, 0.0)        # no-slip walls; right outflow uses outflow_faces (not boundary_value)
#############################################################


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

unit_max_knot_xi = 1
unit_max_knot_eta = 1


n_quad              = max(degree1+1, degree2+2)+1
interval            = [0, 1]
quad                = gq_nD.GaussQuadrature2D(n_quad, n_quad, interval, interval)
quad_1D             = gq_nD.GaussQuadrature1D(n_quad, start_pt=interval[0], end_pt=interval[1])
gamma               =  5 * (degree1 + 1) # 20 * max(degree1, degree2)**3
ifID                = True 
USE_CURVED_GEOMETRY = False
option_number       = 3
is_L2Projection     = False
is_Stokes           = False
is_NavierStokes     = False
is_JetNavierStokes  = True


######## Square domain
# cpts = spline.grevillePoints( unitkv1, unitkv2, degree1, degree2 )#/ max_knot # make the domain a unit square   

######## Curve domain
# cpts = np.array([
#     [0.5, 0.5, 1.0,   0.25, 0.25, 1.0,   0.0, 0.0, 1.0 ], 
#     [0.0, 0.5, 0.5,   0.0, 0.75, 0.75,   0.0, 1.0, 1.0 ]
# ])

def map_xi_eta_to_xy(xi, eta):
    x = -3*eta + 3*xi**2 + 3*xi**2*eta - xi**3

    y = ( 3*xi - 3*xi*eta + 9*xi*eta**2 - 3*xi*eta**3
          + 15*xi**2*eta - 18*xi**2*eta**2 + 6*xi**2*eta**3
          - xi**3 - 9*xi**3*eta + 9*xi**3*eta**2 - 3*xi**3*eta**3 )
    return x, y

def make_cpts(kv1_, kv2_, deg1_, deg2_, min_knot, unit_max_knot_xi, unit_max_knot_eta):                                                                                                        
    cpts_sq = spline.grevillePoints(kv1_, kv2_, deg1_, deg2_)                    
    if not USE_CURVED_GEOMETRY:                                                    
        return cpts_sq 
                                                          
    cpts_ = cpts_sq.copy()
    for i in range(cpts_.shape[1]):
        xi_cp  = cpts_sq[0, i]
        eta_cp = cpts_sq[1, i]
    
        # if your knot range is [min_knot,max_knot]=[0,1], this is just xi_cp, eta_cp
        xi_n  = (xi_cp  - min_knot) / (unit_max_knot_xi - min_knot)
        eta_n = (eta_cp - min_knot) / (unit_max_knot_eta - min_knot)
    
        X, Y = map_xi_eta_to_xy(xi_n, eta_n)
        cpts_[0, i] = X
        cpts_[1, i] = Y
        
    return cpts_                                                      
                                                               
cpts = make_cpts(unitkv1, unitkv2, degree1, degree2, min_knot, unit_max_knot_xi, unit_max_knot_eta)

basis = spline.NavierStokesTPDiscretization( kv1, kv2, degree1, degree2, cpts)

#################### Refinement ####################
num_elems = 2

refined_basis = spline.globallyHRefine(basis, num_divisions=num_elems, parametric_tolerance=1e-5)
kv1_refined, kv2_refined = refined_basis.knotVectors()
cpts_refined = refined_basis.control_points

