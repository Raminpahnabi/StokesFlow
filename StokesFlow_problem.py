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
import matplotlib.pyplot as plt  
import Inputfile as inp

KINEMATIC_VISCOSITY = 1


max_knot                = inp.max_knot
min_knot                = inp.min_knot
degree1                 = inp.degree1
degree2                 = inp.degree2
nelem1                  = inp.nelem1
nelem2                  = inp.nelem2
kv1                     = inp.kv1
kv2                     = inp.kv2
degs                    = inp.degs
cpts                    = inp.cpts
quad                    = inp.quad
quad_1D                 = inp.quad_1D
gamma                   = inp.gamma
forcing_function        = inp.forcing_function
exact_solution          = inp.exact_solution
exact_solution_l2       = inp.exact_solution_l2
boundary_value_function = inp.boundary_value_function
ifID                    = inp.ifID

 
basis = spline.NavierStokesTPDiscretization( kv1, kv2, degree1, degree2, cpts)


# Quick check
example_d = ss.Stokes(basis, degree1, quad, quad_1D, gamma,
                   forcing_function, exact_solution,
                   boundary_conditions=None,
                   boundary_value_function=boundary_value_function,ifID=ifID)
print("example_d:", example_d)

nref = 3
refined_basis = spline.globallyHRefine(basis, nelem1*nelem2*nref, parametric_tolerance=1e-5)
dtotal = ss.Stokes(refined_basis, degs, quad, quad_1D, gamma,
                forcing_function, exact_solution,
                boundary_conditions=None,
                boundary_value_function=boundary_value_function,ifID=ifID)

# p.PlotSolution(refined_basis, dtotal, quad, quad_1D, gamma, forcing_function, nelem1*2, exact_solution, exact_solution_l2)


def manufactured_sol_degrees_clean():  #NEWCODE
    degrees = [2,3]  
    colors  = ['b', 'g', 'r', 'c']  
    refinement_levels = [8, 16, 32, 64]  
    interval_d = [0,1]  
    max_knot_d = 1  

    plt.figure(figsize=(8, 6))

    for idx, deg in enumerate(degrees): 
        print(f"\n{'='*60}")  
        print(f"Processing degree {deg}...") 
        print(f"{'='*60}") 

        # Build quadrature for this degree  
        n_quad_d  = deg + 1 
        quad_d    = gq_nD.GaussQuadrature2D(n_quad_d, n_quad_d, interval_d, interval_d)  
        quad_1D_d = gq_nD.GaussQuadrature1D(n_quad_d, start_pt=interval_d[0], end_pt=interval_d[1])  
        gamma_d   = 20 * deg**3  

        # Build coarsest single-element basis for this degree  
        kv1_d = spline.KnotVector([0]*deg + [0, max_knot_d] + [max_knot_d]*deg, 1e-9)  
        kv2_d = spline.KnotVector([0]*deg + [0, max_knot_d] + [max_knot_d]*deg, 1e-9)  
        cpts_d = spline.grevillePoints(kv1_d, kv2_d, deg, deg)  
        basis_d = spline.NavierStokesTPDiscretization(kv1_d, kv2_d, deg, deg, cpts_d)  

        errors   = []  
        h_values = []  

        print("level | n_divisions | h           | error")  
        for ilevel, n_div in enumerate(refinement_levels):  
            rb = spline.globallyHRefine(basis_d, n_div, parametric_tolerance=1e-5)  

            d = ss.Stokes(rb, [deg, deg], quad_d, quad_1D_d, gamma_d,  
                          forcing_function, exact_solution, 
                          boundary_conditions=None,  
                          boundary_value_function=boundary_value_function, 
                          ifID=True)

            e = cn.compute_convergence_error(rb, d, quad_d, exact_solution, isHDIV=True)  
            
            h = np.sqrt(cn.compute_largest_element_area(rb, quad_d)) 
            errors.append(e)  
            h_values.append(h) 
            print(f"{ilevel:5d} | {n_div:11d} | {h:.6e} | {e:.6e}")  

        log_h = np.log(h_values)  
        log_e = np.log(errors)  
        slope, _ = np.polyfit(log_h, log_e, 1)  # least-squares fit through all 4 points  
        print(f"Degree {deg}: convergence slope = {slope:.4f}")  

        plt.plot(log_h, log_e, marker='o', linestyle='--',  
                 color=colors[idx],  
                 label=f'Degree {deg} (slope ≈ {round(slope)})')  

    plt.xlabel(r'$\log(h)$')  
    plt.ylabel(r'$\log(\|e\|_0)$')  
    plt.title(r'Velocity Convergence: $\|e\|_0$ vs $\log(h)$ for Degrees 2–5')  
    plt.legend()  
    plt.grid(True)  
    plt.show()  


manufactured_sol_degrees_clean()  