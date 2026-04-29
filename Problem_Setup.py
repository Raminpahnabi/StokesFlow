#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:44:23 2026

@author: raminpahnabi
"""
import Inputfile as inp
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
import Solver_L2Projection as ls
import Solver_StokesFlow as ss
import Solver_NonlinearNavierStokes as nss
import export_vtk as vtk

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
JetNavierStokes    = inp.is_JetNavierStokes
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
    
    elif use_curve_geometry == True: 
        if option == 0:
            forcing_function        = inp.forcing_function_l2projection_0
            exact_solution          = inp.exact_solution_0
            exact_solution_l2       = inp.exact_solution_l2_0
            boundary_value_function = inp.boundary_value_function_0


elif Stokes:
    if use_curve_geometry   == False:
        if option == 1:
            forcing_function        = inp.forcing_function_s_1
            exact_solution          = inp.exact_solution_1
            exact_solution_l2       = inp.exact_solution_l2_1
            boundary_value_function = inp.boundary_value_function_1
        
        elif option == 2:
            forcing_function        = inp.forcing_function_cavity_2
            exact_solution          = inp.exact_solution_cavity_2
            exact_solution_l2       = inp.exact_solution_l2_cavity_2
            boundary_value_function = inp.boundary_value_function_cavity_2


elif NavierStokes:
    if use_curve_geometry   == False:
        if option == 1:
            forcing_function        = inp.forcing_function_s_1
            f_ns                    = inp.forcing_function_ns_1
            exact_solution          = inp.exact_solution_1
            exact_solution_l2       = inp.exact_solution_l2_1
            boundary_value_function = inp.boundary_value_function_1
        
        elif option == 2:
            forcing_function        = inp.forcing_function_cavity_2
            f_ns                    = inp.forcing_function_cavity_2
            exact_solution          = inp.exact_solution_cavity_2
            exact_solution_l2       = inp.exact_solution_l2_cavity_2
            boundary_value_function = inp.boundary_value_function_cavity_2


# # Quick check ..
# if L2Projection:
#     example_d_check = ls.L2Projection(basis, degree1, quad, quad_1D, gamma,
#                                 forcing_function, exact_solution,
#                                 boundary_conditions=None, 
#                                 boundary_value_function=boundary_value_function, 
#                                 ifID=ifID,nu=nu,
#                                 use_curve_geometry =use_curve_geometry)
#     print("example_d_L2Projection:", example_d_check)

# elif Stokes:
#     example_d_check = ss.Stokes(basis, degs, quad, quad_1D, gamma,
#                                 forcing_function, exact_solution,
#                                 boundary_conditions=None,
#                                 boundary_value_function=boundary_value_function,
#                                 ifID=ifID,nu=nu)
#     print("example_d_Stokes:", example_d_check)
    
# elif NavierStokes:
#     d_initial = ss.Stokes(basis, degs, quad, quad_1D, gamma,
#                                 forcing_function, exact_solution,
#                                 boundary_conditions=None,
#                                 boundary_value_function=boundary_value_function,
#                                 ifID=ifID,nu=nu)
    
#     example_d_check = nss.NavierStokes(basis, degree1, quad, quad_1D, gamma, 
#                                 forcing_function, f_ns, exact_solution,
#                                 boundary_conditions=None, 
#                                 boundary_value_function=boundary_value_function, 
#                                 ifID=ifID, d_initial=d_initial, nu=nu) 
#     print("NavierStokes solution:", example_d_check)  
    
    
# ########### START of NORMALIZING Pressure
# alpha      = npre.EvaluateAveragePressure(basis, example_d_check, quad)      #(α = int_Ω p_h dΩ)
# print("avg pressure before normalization:", alpha)                          
# example_d_check = npre.NormalizePressureCoefficients(basis, example_d_check, degree1, quad, quad_1D)
# average_pressure_after = npre.EvaluateAveragePressure(basis, example_d_check, quad)  
# print("avg pressure after  normalization:", average_pressure_after)          
# ########### END of NORMALIZING Pressure
# pl.PlotSolution(basis, example_d_check, quad, quad_1D, gamma, forcing_function, nelem1*2, exact_solution, exact_solution_l2)


def check_local_refinement_vtk():
    """
    Build a 3-level hierarchically refined mesh on a unit-square domain
    and export it as a Bezier-cell VTK (one cell per Bezier element,
    colored by 'refinement_level': 0 = base, 1 = once refined, etc.)

    Open 'bezier_mesh_refinement_check.vtk' in ParaView and colour by
    'refinement_level' to visually verify the nested refinement pattern.

    Index scheme (0-based, virtualised grids):
      Level 1 elements live in the  8×8 base grid  → rows/cols 2–5
      Level 2 elements live in a 16×16 virtual grid → rows/cols 6–9
      Level 3 elements live in a 32×32 virtual grid → rows/cols 14–17
    Each finer level subdivides the inner 4×4 block of the previous one,
    producing three concentric nested squares.
    """
    # -- Unit-square base with 2 elements per direction --
    kv_1d = spline.KnotVector(
        [0]*degree1 + list(np.linspace(0, 1, 3)) + [1]*degree1, 1e-9
    )
    kv_2d = spline.KnotVector(
        [0]*degree2 + list(np.linspace(0, 1, 3)) + [1]*degree2, 1e-9
    )
    cpts_sq = spline.grevillePoints(kv_1d, kv_2d, degree1, degree2)
    basis_sq = spline.NavierStokesTPDiscretization(kv_1d, kv_2d, degree1, degree2, cpts_sq)

    # -- Globally refine to 8×8 elements (4 divisions × 2 base elements) --
    rb = spline.globallyHRefine(basis_sq, 4, parametric_tolerance=1e-5)
    kv1_r, kv2_r = rb.knotVectors()
    cpts_r = rb.control_points

    # -- Three-level concentric local refinement --
    # Level 1: 4×4 central block on the 8×8 base grid (rows/cols 2–5, 0-indexed)
    L1 = [[r, c] for r in range(2, 6) for c in range(2, 6)]

    # Level 2: inner 4×4 block on the 16×16 virtual grid.
    # Level-1 element [r,c] spawns children at [2r,2c]…[2r+1,2c+1] in the 16×16 grid.
    # The inner 2×2 of the L1 block (base rows 3–4) maps to 16×16 rows 6–9.
    L2 = [[r, c] for r in range(6, 10) for c in range(6, 10)]

    # Level 3: inner 4×4 block on the 32×32 virtual grid.
    # Level-2 element [r,c] spawns children at [2r,2c]…[2r+1,2c+1] in the 32×32 grid.
    # The inner 2×2 of the L2 block (16×16 rows 7–8) maps to 32×32 rows 14–17.
    L3 = [[r, c] for r in range(14, 18) for c in range(14, 18)]

    hier = spline.NavierStokesHierarchicalDiscretization(
        kv2_r, kv1_r, degree1, degree2, cpts_r, [L1, L2, L3]
    )

    n_elems = len(list(hier.elements()))
    print(f"Refinement check mesh: {n_elems} total Bezier elements")
    vtk.export_bezier_mesh_vtk(hier, vtk_path='bezier_mesh_refinement_check.vtk')


def manufactured_sol_degrees_clean():
    degrees = [2,3,4]  
    colors  = ['b', 'g', 'r', 'c']  
    refinement_levels = [8, 16, 32]#, 64]  
    # refinement_levels = [32, 64, 128]
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
        n_quad_d  = deg + 3 
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
        
        num_elems = 1
        basis_d = spline.globallyHRefine(basis_d, num_divisions=num_elems, parametric_tolerance=1e-5)
        kv1_d, kv2_d = basis_d.knotVectors()
        cpts_d = basis_d.control_points

        errors   = []
        errors_p = []  
        h_values = []

        print("level | n_divisions | h           | error")  
        for ilevel, n_div in enumerate(refinement_levels):  
            rb_local = spline.globallyHRefine(basis_d, n_div, parametric_tolerance=1e-5)
            
            kv1_d, kv2_d = rb_local.knotVectors()
            cpts_d = rb_local.control_points
            # Scale indices with n_div so the locally refined region stays at
            # the same physical fraction [1/8, 5/8] of the domain for every n_div.
            # At n_div=8: indices 1–4 → physical [0.125, 0.625].
            # At n_div=16: indices 2–9 → same physical region.
            # At n_div=32: indices 4–19 → same physical region.
            r_start = n_div // 8
            r_end   = 2 * n_div // 8
            L1 = [[r, c] for r in range(r_start, r_end) for c in range(r_start, r_end)]
            elems_to_refine = [L1]
            
            # elems_to_refine = []
            # # elems_to_refine.append([[1,1],[1,2],[1,3],[1,4],[2,1],[2,2],[2,3],[2,4],[3,1],[3,2],[3,3],[3,4],[4,1],[4,2],[4,3],[4,4]])#, [q, 5], [q, 6]])  # Refine one element column
            # elems_to_refine.append([[1,1],[1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]])
          
            # rb_local = spline.NavierStokesHierarchicalDiscretization(
            #     kv2_d, kv1_d, deg, deg, cpts_d, elems_to_refine
            # )
            
            if L2Projection:
                d = ls.L2Projection(rb_local, [deg, deg], quad_d, quad_1D_d, gamma_d,  
                          forcing_function, exact_solution, 
                          boundary_conditions=None,  
                          boundary_value_function=boundary_value_function, 
                          ifID=True,nu=nu,use_curve_geometry=use_curve_geometry)
            
            elif Stokes:
                d = ss.Stokes(rb_local, [deg, deg], quad_d, quad_1D_d, gamma_d,
                                           forcing_function, exact_solution,
                                           boundary_conditions=None,
                                           boundary_value_function=boundary_value_function,
                                           ifID=True,nu=nu)

            elif NavierStokes:
                d_initial = ss.Stokes(rb_local, [deg, deg], quad_d, quad_1D_d, gamma_d,
                                           forcing_function, exact_solution,
                                           boundary_conditions=None,
                                           boundary_value_function=boundary_value_function,
                                           ifID=True,nu=nu)
                
                d = nss.NavierStokes(rb_local, [deg, deg], quad_d, quad_1D_d, gamma_d, forcing_function, f_ns, exact_solution,
                                    boundary_conditions=None, boundary_value_function=boundary_value_function, ifID=ifID,
                                    d_initial=d_initial, nu=nu)
             
            e = cn.compute_convergence_error(rb_local, d, quad_d, exact_solution, isHDIV=True)  
            
            ########### START of NORMALIZING Pressure
            alpha_rb      = npre.EvaluateAveragePressure(rb_local, d, quad_d)               
            d = npre.NormalizePressureCoefficients(rb_local, d, [deg, deg], quad_d, quad_1D_d)
            ########### END of NORMALIZING Pressure
            
            ########### START of VTK file 
            vtk.export_as_vtk(rb_local, d, true_velocity=exact_solution, true_pressure=exact_solution_l2,vtk_path=f"bezier_mesh_level{ilevel}_deg{deg}.vtk")
            # vtk.export_bezier_mesh_vtk(rb_local, vtk_path=f"bezier_mesh_level{ilevel}_deg{deg}.vtk")

            e_p = cn.compute_pressure_convergence_error(rb_local, d, quad_d, exact_solution_l2)  

            h = np.sqrt(cn.compute_largest_element_area(rb_local, quad_d))
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

if __name__ == '__main__':
    # check_local_refinement_vtk()
    manufactured_sol_degrees_clean()
