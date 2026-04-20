#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:44:23 2026

@author: raminpahnabi
"""
import NS_Inputfile as inp
import sys
import os
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, inp.sweepspath)
sys.path.append(str(PROJECT_ROOT / 'HWs'))
sys.path.append(str(PROJECT_ROOT / 'Required'))

import splines as spline
import Gaussian_Quadrature_2D_Solution as gq_nD
import Convergence as cn
import matplotlib.pyplot as plt  
import NormalizedPressure as npre
import Plotting as pl
import L2Projection_solver as ls
import StokesFlow_Solver as ss

max_knot_xi        = inp.max_knot_xi
max_knot_eta       = inp.max_knot_eta
min_knot           = inp.min_knot
degree1            = inp.degree1
degree2            = inp.degree2
nelem1             = inp.nelem1
nelem2             = inp.nelem2
kv1                = inp.kv1
kv2                = inp.kv2
degs               = inp.degs
cpts               = inp.cpts
quad               = inp.quad
quad_1D            = inp.quad_1D
gamma              = inp.gamma
ifID               = inp.ifID
nu                 = inp.KINEMATIC_VISCOSITY
use_curve_geometry = inp.USE_CURVED_GEOMETRY
L2Projection       = inp.is_L2Projection
Stokes             = inp.is_Stokes
NavierStokes       = inp.is_NavierStokes
option             = inp.option_number

if L2Projection:
    if use_curve_geometry   == False:
        if option == 0:
            forcing_function        = inp.forcing_function_l2projection_0
            exact_solution          = inp.exact_solution_0
            exact_solution_l2       = inp.exact_solution_l2_0
            boundary_value_function = inp.boundary_value_function_0
        elif option == 1:
            forcing_function        = inp.forcing_function_l2projection_1
            exact_solution          = inp.exact_solution_1
            exact_solution_l2       = inp.exact_solution_l2_1
            boundary_value_function = inp.boundary_value_function_1
    elif use_curve_geometry == True: # TODO: Are these the same as square?
        if option == 0:
            forcing_function        = inp.forcing_function_l2projection_0
            exact_solution          = inp.exact_solution_0
            exact_solution_l2       = inp.exact_solution_l2_0
            boundary_value_function = inp.boundary_value_function_0
        elif option == 1:
            forcing_function        = inp.forcing_function_l2projection_1
            exact_solution          = inp.exact_solution_1
            exact_solution_l2       = inp.exact_solution_l2_1
            boundary_value_function = inp.boundary_value_function_1

elif Stokes:
    if use_curve_geometry   == False:
        forcing_function        = inp.forcing_function_s_1
        exact_solution          = inp.exact_solution_1
        exact_solution_l2       = inp.exact_solution_l2_1
        boundary_value_function = inp.boundary_value_function_1
    elif use_curve_geometry == True:
        forcing_function        = inp.forcing_function_s_1_curve
        exact_solution          = inp.exact_solution_1_curve
        exact_solution_l2       = inp.exact_solution_1_l2_curve
        boundary_value_function = inp.boundary_value_function_1_curve

elif NavierStokes:
    if use_curve_geometry   == False:
        forcing_function        = inp.forcing_function_ns_1
        exact_solution          = inp.exact_solution_1
        exact_solution_l2       = inp.exact_solution_l2_1
        boundary_value_function = inp.boundary_value_function_1
    elif use_curve_geometry == True:
        forcing_function        = inp.forcing_function_ns_1_curve
        exact_solution          = inp.exact_solution_1_curve
        exact_solution_l2       = inp.exact_solution_l2_1_curve
        boundary_value_function = inp.boundary_value_function_1_curve
        
basis = spline.NavierStokesTPDiscretization( kv1, kv2, degree1, degree2, cpts)

# Quick check ..
if L2Projection:
    example_d_check = ls.L2Projection(basis, degree1, quad, quad_1D, gamma,
                        forcing_function, exact_solution,
                        boundary_conditions=None, boundary_value_function=boundary_value_function, ifID=ifID,nu=nu,use_curve_geometry =use_curve_geometry)
elif Stokes:
    example_d_check = ss.Stokes(basis, degs, quad, quad_1D, gamma,
                               forcing_function, exact_solution,
                               boundary_conditions=None,
                               boundary_value_function=boundary_value_function,
                               ifID=True,nu=nu)
    
print("example_d:", example_d_check)
########### START of NORMALIZING Pressure
alpha      = npre.EvaluateAveragePressure(basis, example_d_check, quad)      #(α = int_Ω p_h dΩ)
print("avg pressure before normalization:", alpha)                          
example_d_check = npre.NormalizePressureCoefficients(basis, example_d_check, degree1, quad, quad_1D)
average_pressure_after = npre.EvaluateAveragePressure(basis, example_d_check, quad)  
print("avg pressure after  normalization:", average_pressure_after)          
########### END of NORMALIZING Pressure
# pl.PlotSolution(basis, example_d_check, quad, quad_1D, gamma, forcing_function, nelem1*2, exact_solution, exact_solution_l2)


def manufactured_sol_degrees_clean():  
    degrees = [2,3,4]  
    colors  = ['b', 'g', 'r', 'c']  
    refinement_levels = [8, 16, 32]#, 64]  
    interval_d = [0,1]  
    max_knot_d_xi = max_knot_xi  
    max_knot_d_eta = max_knot_eta 

    plt.figure(figsize=(8, 6))
    fig_pres, ax_pres = plt.subplots(figsize=(8, 6)) 

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
        kv1_d = spline.KnotVector([0]*deg + [0, max_knot_d_xi] + [max_knot_d_xi]*deg, 1e-9)  
        kv2_d = spline.KnotVector([0]*deg + [0, max_knot_d_eta] + [max_knot_d_eta]*deg, 1e-9)  
        unitkv1_d = spline.KnotVector([0]*deg + [0, 1] + [1]*deg, 1e-9)
        unitkv2_d = spline.KnotVector([0]*deg + [0, 1] + [1]*deg, 1e-9)
        cpts_d = inp.make_cpts(unitkv1_d, unitkv2_d, deg, deg, min_knot, 1, 1)
        basis_d = spline.NavierStokesTPDiscretization(kv1_d, kv2_d, deg, deg, cpts_d)  

        errors   = []
        errors_p = []  
        h_values = []

        print("level | n_divisions | h           | error")  
        for ilevel, n_div in enumerate(refinement_levels):  
            rb = spline.globallyHRefine(basis_d, n_div, parametric_tolerance=1e-5)  

            if L2Projection:
                d = ls.L2Projection(rb, [deg, deg], quad_d, quad_1D_d, gamma_d,  
                          forcing_function, exact_solution, 
                          boundary_conditions=None,  
                          boundary_value_function=boundary_value_function, 
                          ifID=True,nu=nu,use_curve_geometry=use_curve_geometry)
            
            elif Stokes:
                d = ss.Stokes(rb, [deg, deg], quad_d, quad_1D_d, gamma_d,
                                           forcing_function, exact_solution,
                                           boundary_conditions=None,
                                           boundary_value_function=boundary_value_function,
                                           ifID=True,nu=nu)

            e = cn.compute_convergence_error(rb, d, quad_d, exact_solution, isHDIV=True)  
            
            ########### START of NORMALIZING Pressure
            alpha_rb      = npre.EvaluateAveragePressure(rb, d, quad_d)               
            d = npre.NormalizePressureCoefficients(rb, d, [deg, deg], quad_d, quad_1D_d)
            ########### END of NORMALIZING Pressure

            e_p = cn.compute_pressure_convergence_error(rb, d, quad_d, exact_solution_l2)  

            h = np.sqrt(cn.compute_largest_element_area(rb, quad_d))
            errors.append(e)
            errors_p.append(e_p)  
            h_values.append(h)
            print(f"{ilevel:5d} | {n_div:11d} | {h:.6e} | vel {e:.6e} | pres {e_p:.6e}")  

        log_h = np.log(h_values)
        log_e = np.log(errors)
        log_e_p = np.log(errors_p)  
        slope, _ = np.polyfit(log_h, log_e, 1)  # least-squares fit through all 4 points
        slope_p, _ = np.polyfit(log_h, log_e_p, 1)  
        print(f"Degree {deg}: velocity slope = {slope:.4f} | pressure slope = {slope_p:.4f}")  

        plt.plot(log_h, log_e, marker='o', linestyle='--',
                 color=colors[idx],
                 label=f'Degree {deg} (slope ≈ {round(slope)})')
        ax_pres.plot(log_h, log_e_p, marker='s', linestyle='--',  
                     color=colors[idx],                             
                     label=f'Degree {deg} (slope_p ≈ {round(slope_p)})')  

    plt.xlabel(r'$\log(h)$')
    plt.ylabel(r'$\log(\|e\|_0)$')
    plt.title(r'Velocity Convergence: $\|e\|_0$ vs $\log(h)$ for Degrees 2–5')
    plt.legend()
    plt.grid(True)
    plt.show()

    ax_pres.set_xlabel(r'$\log(h)$')                                                  
    ax_pres.set_ylabel(r'$\log(\|e_p\|_0)$')                                         
    ax_pres.set_title(r'Pressure Convergence: $\|e_p\|_0$ vs $\log(h)$ for Degrees 2–5') 
    ax_pres.legend()                                                                    
    ax_pres.grid(True)                                                                
    fig_pres.show()                                                                   

manufactured_sol_degrees_clean()
