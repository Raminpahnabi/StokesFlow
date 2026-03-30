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
import matplotlib.pyplot as plt  #NEWCODE

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

################################ STOKES

def stokes_exact_solution(x, y):
    # Stream function: psi = x^2*(1-x)^2*y^2*(1-y)^2  =>  div(u) = 0, u = 0 on all boundaries
    # POLYNOMIAL (degree 4 in each variable) -- works for degrees 2 and 3.
    # For degrees >= 4 the exact solution lies inside the approximation space,
    # so errors collapse to machine precision and the computed slope is meaningless.
    u1 = 2*x**2*(1-x)**2*y*(1-y)*(1-2*y)
    u2 = -2*x*(1-x)*(1-2*x)*y**2*(1-y)**2
    return u1, u2

def stokes_exact_solution_l2(x, y):
    # Zero pressure for this manufactured solution
    return 0.0

def stokes_forcing_function(x, y):
    # f = -nabla^2 u  (pure Stokes, p=0)
    # u1 = A(x)*B(y),  A=2x^2(1-x)^2,  B=y(1-y)(1-2y)
    # A''(x) = 4 - 24x + 24x^2
    # B''(y) = -6 + 12y
    f1 = -((4 - 24*x + 24*x**2)*y*(1-y)*(1-2*y)
            + 2*x**2*(1-x)**2*(-6 + 12*y))
    # u2 = C(x)*D(y),  C=-2x(1-x)(1-2x),  D=y^2(1-y)^2
    # C''(x) = 12 - 24x
    # D''(y) = 2 - 12y + 12y^2
    f2 = -((12 - 24*x)*y**2*(1-y)**2
            + (-2*x*(1-x)*(1-2*x))*(2 - 12*y + 12*y**2))
    return f1, f2

############################## Option No.4 # Driven Cavity
def forcing_function_0(x, y):
    # Zero forcing function
    return 0, 0

def exact_solution_0(r, theta):
    # Exact solution using the provided expressions in polar coordinates
    lambda_value = 0.54448373678246
    phi = (np.sin((1 + lambda_value) * theta) * np.cos(lambda_value * theta) / (1 + lambda_value) -
           np.cos((1 + lambda_value) * theta) / (1 - lambda_value))
    
    # Calculating the components of the velocity field
    u_r = r**lambda_value * ((1 + lambda_value) * np.sin(theta) * phi + np.cos(theta) * phi)
    u_theta = r**lambda_value * (-(1 + lambda_value) * np.cos(theta) * phi + np.sin(theta) * phi)
    
    return u_r, u_theta

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
# cpts = spline.grevillePoints( kv1, kv2, degree1, degree2 )#/ max_knot # make the domain a unit square   

######## Curve domain
cpts = np.array([
    [0.5, 0.5, 1.0,   0.25, 0.25, 1.0,   0.0, 0.0, 1.0 ], 
    [0.0, 0.5, 0.5,   0.0, 0.75, 0.75,   0.0, 1.0, 1.0 ]
])

basis = spline.NavierStokesTPDiscretization( kv1, kv2, degree1, degree2, cpts)


n_quad   = max(degree1+1, degree2+2)+1
interval = [0, 1]
quad     = gq_nD.GaussQuadrature2D(n_quad, n_quad, interval, interval)
quad_1D  = gq_nD.GaussQuadrature1D(n_quad, start_pt=interval[0], end_pt=interval[1])
gamma    =  20 * max(degree1, degree2)**3
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



# p.PlotSolution(refined_basis, dtotal, quad_plus, quad_1D, gamma,
#               forcing_function, nelem1*2, exact_solution, exact_solution_l2)

##########################################################################
#NEWCODE: Trig manufactured solution — derived from stream function      #
#         psi = sin^2(pi*x)*sin^2(pi*y), so div(u)=0 and u=0 on all    #
#         four edges. f = -nabla^2(u) (pure Stokes, zero pressure).     #
#         This solution is NEVER polynomial, so it gives true O(h^p)    #
#         convergence for all degrees (no superconvergence saturation).  #
##########################################################################

def trig_exact_solution(x, y):  #NEWCODE
    u1 =  np.pi * np.sin(np.pi * x)**2 * np.sin(2 * np.pi * y)  #NEWCODE
    u2 = -np.pi * np.sin(2 * np.pi * x) * np.sin(np.pi * y)**2  #NEWCODE
    return u1, u2  #NEWCODE

def trig_forcing_function(x, y):  #NEWCODE
    # f = -nabla^2(u), pure Stokes with zero pressure  #NEWCODE
    f1 = 2 * np.pi**3 * (4 * np.sin(np.pi * x)**2 - 1) * np.sin(2 * np.pi * y)  #NEWCODE
    f2 = 2 * np.pi**3 * np.sin(2 * np.pi * x) * (1 - 4 * np.sin(np.pi * y)**2)  #NEWCODE
    return f1, f2  #NEWCODE

def trig_boundary_value(x, y):  #NEWCODE
    # trig exact solution is exactly zero on all four edges  #NEWCODE
    return 0.0, 0.0  #NEWCODE


def manufactured_sol_degrees_clean():  #NEWCODE
    degrees = [2,3]  #NEWCODE
    colors  = ['b', 'g', 'r', 'c']  #NEWCODE
    refinement_levels = [8, 16, 32, 64]  # 4 data points -> robust slope  #NEWCODE
    interval_d = [0,1]  #NEWCODE
    max_knot_d = 1  #NEWCODE
    is_trig = False

    plt.figure(figsize=(8, 6))  #NEWCODE

    for idx, deg in enumerate(degrees):  #NEWCODE
        print(f"\n{'='*60}")  #NEWCODE
        print(f"Processing degree {deg}...")  #NEWCODE
        print(f"{'='*60}")  #NEWCODE

        # Build quadrature for this degree  #NEWCODE
        n_quad_d  = deg + 1  #NEWCODE
        quad_d    = gq_nD.GaussQuadrature2D(n_quad_d, n_quad_d, interval_d, interval_d)  #NEWCODE
        quad_1D_d = gq_nD.GaussQuadrature1D(n_quad_d, start_pt=interval_d[0], end_pt=interval_d[1])  #NEWCODE
        gamma_d   = 20 * deg**3  #NEWCODE

        # Build coarsest single-element basis for this degree  #NEWCODE
        kv1_d = spline.KnotVector([0]*deg + [0, max_knot_d] + [max_knot_d]*deg, 1e-9)  #NEWCODE
        kv2_d = spline.KnotVector([0]*deg + [0, max_knot_d] + [max_knot_d]*deg, 1e-9)  #NEWCODE
        cpts_d = spline.grevillePoints(kv1_d, kv2_d, deg, deg)  #NEWCODE
        basis_d = spline.NavierStokesTPDiscretization(kv1_d, kv2_d, deg, deg, cpts_d)  #NEWCODE

        errors   = []  #NEWCODE
        h_values = []  #NEWCODE

        print("level | n_divisions | h           | error")  #NEWCODE
        for ilevel, n_div in enumerate(refinement_levels):  #NEWCODE
            rb = spline.globallyHRefine(basis_d, n_div, parametric_tolerance=1e-5)  #NEWCODE

            if is_trig:
                d = ss.Stokes(rb, [deg, deg], quad_d, quad_1D_d, gamma_d,  #NEWCODE
                              trig_forcing_function, trig_exact_solution,  #NEWCODE
                              boundary_conditions=None,  #NEWCODE
                              boundary_value_function=trig_boundary_value,  #NEWCODE
                              ifID=True)  #NEWCODE
    
                e = cn.compute_convergence_error(rb, d, quad_d, trig_exact_solution, isHDIV=True)  #NEWCODE
            
            if not is_trig:
                d = ss.Stokes(rb, [deg, deg], quad_d, quad_1D_d, gamma_d,  #NEWCODE
                              stokes_forcing_function, stokes_exact_solution,  #NEWCODE
                              boundary_conditions=None,  #NEWCODE
                              boundary_value_function=trig_boundary_value,  #NEWCODE
                              ifID=True)  #NEWCODE
    
                e = cn.compute_convergence_error(rb, d, quad_d, stokes_exact_solution, isHDIV=True)
            h = np.sqrt(cn.compute_largest_element_area(rb, quad_d))  #NEWCODE
            errors.append(e)  #NEWCODE
            h_values.append(h)  #NEWCODE
            print(f"{ilevel:5d} | {n_div:11d} | {h:.6e} | {e:.6e}")  #NEWCODE

        log_h = np.log(h_values)  #NEWCODE
        log_e = np.log(errors)  #NEWCODE
        slope, _ = np.polyfit(log_h, log_e, 1)  # least-squares fit through all 4 points  #NEWCODE
        print(f"Degree {deg}: convergence slope = {slope:.4f}")  #NEWCODE

        plt.plot(log_h, log_e, marker='o', linestyle='--',  #NEWCODE
                 color=colors[idx],  #NEWCODE
                 label=f'Degree {deg} (slope ≈ {round(slope)})')  #NEWCODE

    plt.xlabel(r'$\log(h)$')  #NEWCODE
    plt.ylabel(r'$\log(\|e\|_0)$')  #NEWCODE
    plt.title(r'Velocity Convergence: $\|e\|_0$ vs $\log(h)$ for Degrees 2–5')  #NEWCODE
    plt.legend()  #NEWCODE
    plt.grid(True)  #NEWCODE
    plt.show()  #NEWCODE


manufactured_sol_degrees_clean()  #NEWCODE