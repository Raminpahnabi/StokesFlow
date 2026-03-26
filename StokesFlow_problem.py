#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:44:23 2026

@author: raminpahnabi
"""

import sys
import os
import numpy as np
sys.path.insert(0, '/Users/raminpahnabi/Documents/BYU/sweeps/build/src/api')
sys.path.append(os.path.join(os.getcwd(), 'HWs'))
sys.path.append(os.path.join(os.getcwd(), 'Required'))

import splines as spline
import Gaussian_Quadrature_2D_Solution as gq_nD
import StokesFlow_Solver as ss
import Plotting as p
import Convergence as cn

KINEMATIC_VISCOSITY = 1

########################################################################
########################### Input  #####################################
########################################################################

############################## Option No.1
# def forcing_function( x, y):
#     return y,x

# def exact_solution(x,y):
#     return y,x

# def exact_solution_l2(x,y):
#     return 0

# ############################## Option No.2
# def forcing_function(x, y):
#     return np.sin(np.pi * y), np.sin(np.pi * x)

# def exact_solution(x, y):
#     return np.sin(np.pi * y), np.sin(np.pi * x) # np.sin(np.pi * x) * np.cos(np.pi * y), -np.cos(np.pi * x) * np.sin(np.pi * y)

# def exact_solution_l2(x, y):
#     return np.sin(np.pi * x) #* np.sin(np.pi * y) - (4/((np.pi)**2))

# ############################## Option No.3
# def forcing_function(x, y):
#     f_x = np.cos(np.pi * x) * np.sin(np.pi * y) + np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
#     f_y = -np.sin(np.pi * x) * np.cos(np.pi * y) + np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
#     return f_x, f_y

# def exact_solution(x, y):
#     u_x = np.cos(np.pi * x) * np.sin(np.pi * y)
#     u_y = -np.sin(np.pi * x) * np.cos(np.pi * y)
#     return u_x, u_y

# def exact_solution_l2(x,y):
#     return np.sin(np.pi * x)* np.sin(np.pi * y) - (4/((np.pi)**2))

############################## Option No.4
def forcing_function(x, y):
    f1 = np.exp(x) * (2 * y * (-2 + 7 * y - 6 * y * y + y * y * y) 
                      - 8 * x * y * (-2 + 7 * y - 6 * y * y + y * y * y) 
                      + x * x * (12 - 36 * y + 13 * y * y - 2 * y * y * y + y * y * y * y) 
                      + x * x * x * x * (12 - 36 * y + 13 * y * y - 2 * y * y * y + y * y * y * y) 
                      + x * x * x * (-24 + 56 * y + 30 * y * y - 44 * y * y * y + 6 * y * y * y * y))


    f2 = 2 * (228 - 456 * y + np.exp(x) * (x * x * x * x * (-5 + 7 * y + 3 * y * y + 2 * y * y * y) 
                                            + 2 * x * (115 - 233 * y + y * y + 6 * y * y * y - 2 * y * y * y * y) 
                                            - 3 * (76 - 152 * y + y * y - 2 * y * y * y + y * y * y * y) 
                                            + 2 * x * x * x * (19 - 41 * y + 5 * y * y - 2 * y * y * y + 2 * y * y * y * y) 
                                            + x * x * (-119 + 253 * y - 3 * y * y - 34 * y * y * y + 12 * y * y * y * y)))    
    return f1, f2

def exact_solution(x, y):
    u1 = 2 * np.exp(x) * (-1 + x)*(-1 + x)* x**2 * (y**2 - y) * (-1 + 2 * y)
    u2 = -np.exp(x) * (-1 + x) * x * (-2 + x * (3 + x)) * (-1 + y)**2 * y**2
    return u1, u2

def exact_solution_l2(x, y):
    # p = -424+156*np.exp(1)+(-y+y*y)*(-456+np.exp(x)*(456+2*x*(-228-y+y*y)+2*x*x*x*(-36-y+y*y)+x*x*x*x*(12-y+y*y)+x*x*(228-5*(-y+y*y))))
    # p = -424+156*np.exp(1)+(-y+y*y)*(-456+np.exp(x)*(456++x*x*(228-5*(-y+y*y))+2*x*(-228+(-y+y*y))+2*x*x*x*(-36+(-y+y*y))+x*x*x*x*(12+(-y+y*y))))
    p = -424+156*np.exp(1)+(-y+y**2)*(-456+np.exp(x)*(456+x**2*(228-5*(-y+y**2))+2*x*(-228+(-y+y**2))+2*x**3*(-36+(-y+y**2))+x**4*(12+(-y+y**2))))
    # p = (-424 +156*np.exp(1)+(-y+y**2)*(-456+np.exp(x)*(456+2*x*(-228-y+y*y)+2*x*x*x*(-36-y+y*y)+x*x*x*x*(12-y+y*y)+x*x*(228-5*(-y+y*y))))) #0.0032#-.00444
    return p

max_knot = 1
min_knot = 0
degree1 = 2
degree2 = 2
nelem1 = 2
nelem2 = 2
degs = [degree1,degree2]
kv1_init = list(np.linspace(0,max_knot,nelem1+1))
kv2_init = list(np.linspace(0,max_knot,nelem2+1))
kv1 = spline.KnotVector([0]*degree1 + kv1_init + [max_knot]*(degree1), 1e-9)
kv2 = spline.KnotVector([0]*degree2 + kv2_init + [max_knot]*(degree2), 1e-9)

######## Square domain
cpts = spline.grevillePoints( kv1, kv2, degree1, degree2 )#/ max_knot # make the domain a unit square   

######## Curve domain
# cpts = np.array([
#     [0.5, 0.5, 1.0,   0.25, 0.25, 1.0,   0.0, 0.0, 1.0 ], 
#     [0.0, 0.5, 0.5,   0.0, 0.75, 0.75,   0.0, 1.0, 1.0 ]
# ])

basis = spline.NavierStokesTPDiscretization( kv1, kv2, degree1, degree2, cpts)


n_quad   = max(degree1, degree2) + 1
interval = [0, 1]
quad     = gq_nD.GaussQuadrature2D(n_quad, n_quad, interval, interval)
quad_1D  = gq_nD.GaussQuadrature1D(n_quad, start_pt=interval[0], end_pt=interval[1])
gamma    =  20 * max(degree1, degree2)**2
ifID = True

def boundary_value_function(x, y):
    return exact_solution(x, y)

# Quick check
example_d = ss.Stokes(basis, degree1, quad, quad_1D, gamma,
                   forcing_function, exact_solution,
                   boundary_conditions=None,
                   boundary_value_function=boundary_value_function,ifID=ifID)
print("example_d:", example_d)

quad_plus = gq_nD.GaussQuadrature2D(n_quad, n_quad, interval, interval)

nref = 3
refined_basis = spline.globallyHRefine(basis, nelem1*nelem2*nref, parametric_tolerance=1e-5)
dtotal = ss.Stokes(refined_basis, degs, quad, quad_1D, gamma,
                forcing_function, exact_solution,
                boundary_conditions=None,
                boundary_value_function=boundary_value_function,ifID=ifID)


p.PlotSolution(refined_basis, dtotal, quad_plus, quad_1D, gamma,
              forcing_function, nelem1*2, exact_solution, exact_solution_l2)

num_iterations = 2
cn.run_convergence_study(num_iterations, basis, degs, quad, quad_1D, gamma,
                      forcing_function, exact_solution, exact_solution_l2,
                      boundary_value_function,nelem1,nelem2)