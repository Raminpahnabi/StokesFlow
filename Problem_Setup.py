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
import Solver_L2Projection as ls
import Solver_StokesFlow as ss
import Solver_NonlinearNavierStokes as nss
# import export_vtk as vtk
import Inputfile_force_exactsol as inpfe

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


forcing_function        = inpfe.forcing_function
exact_solution          = inpfe.exact_solution
exact_solution_l2       = inpfe.exact_solution_l2
boundary_value_function = inpfe.boundary_value_function
if NavierStokes or JetNavierStokes:
    f_ns                = inpfe.f_ns

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

        ############ NON_Uniform Knot Vectors ################
        # kv1_d = spline.KnotVector(
        #     [0]*deg + [0, 0.2*max_knot_d_xi, 0.5*max_knot_d_xi, max_knot_d_xi] + [max_knot_d_xi]*deg,1e-9)
        
        # kv2_d = spline.KnotVector(
        #     [0]*deg + [0, 0.1*max_knot_d_eta, 0.7*max_knot_d_eta, max_knot_d_eta] + [max_knot_d_eta]*deg,1e-9)
        
        # unitkv1_d = spline.KnotVector([0]*deg + [0,0.2, 0.5,1] + [1]*deg,1e-9)
        # unitkv2_d = spline.KnotVector([0]*deg + [0,0.1, 0.7,1] + [1]*deg,1e-9)
        
        ############ Uniform Knot Vectors ################
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
            rb = spline.globallyHRefine(basis_d, n_div, parametric_tolerance=1e-5)
            
            kv1_d, kv2_d = rb.knotVectors()
            cpts_d = rb.control_points
            
            # count distinct knot spans = number of elements per direction
            N_xi  = sum(1 for i in range(kv1_d.size()-1) if kv1_d.knot(i+1) - kv1_d.knot(i) > 1e-9)
            N_eta = sum(1 for i in range(kv2_d.size()-1) if kv2_d.knot(i+1) - kv2_d.knot(i) > 1e-9)
            
            N = N_xi          # elements per direction on this mesh (2-elem base × n_div)
            buf = deg - 1     # admissibility buffer for 2-admissible, degree p
        
            # Central quarter of the mesh, padded by buf to ensure admissibility
            # Target inner zone: [N//4, 3*N//4], outer zone adds buf on each side
            inner_s = max(0,   N // 8 - buf)
            inner_e = min(N,   3 * N // 8 + buf)
            L1 = [[r, c] for r in range(inner_s, inner_e)
                          for c in range(inner_s, inner_e)]
            elems_to_refine = [L1]
            
            # r_start = n_div // 8
            # r_end   = 3 * n_div // 8
            # L1 = [[r, c] for r in range(r_start, r_end) for c in range(r_start, r_end)]
            # elems_to_refine = [L1]
            
            # elems_to_refine = []
            # elems_to_refine.append([[1,1],[1,2],[1,3],[1,4],[2,1],[2,2],[2,3],[2,4],[3,1],[3,2],[3,3],[3,4],[4,1],[4,2],[4,3],[4,4]])#, [q, 5], [q, 6]])  # Refine one element column
            # # elems_to_refine.append([[1,1],[1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]])
          
            rb_local = spline.NavierStokesHierarchicalDiscretization(
                kv2_d, kv1_d, deg, deg, cpts_d, elems_to_refine
            )
            
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
                                           ifID=True,nu=nu,use_curve_geometry=use_curve_geometry)

            elif NavierStokes:
                d_initial = ss.Stokes(rb_local, [deg, deg], quad_d, quad_1D_d, gamma_d,
                                           forcing_function, exact_solution,
                                           boundary_conditions=None,
                                           boundary_value_function=boundary_value_function,
                                           ifID=True,nu=nu,use_curve_geometry=use_curve_geometry)
                
                d = nss.NavierStokes(rb_local, [deg, deg], quad_d, quad_1D_d, gamma_d, forcing_function, f_ns, exact_solution,
                                    boundary_conditions=None, boundary_value_function=boundary_value_function, ifID=ifID,
                                    d_initial=d_initial, nu=nu)
             
            e = cn.compute_convergence_error(rb_local, d, quad_d, exact_solution, isHDIV=True)  
            
            ########### START of NORMALIZING Pressure
            alpha_rb      = npre.EvaluateAveragePressure(rb_local, d, quad_d)               
            d = npre.NormalizePressureCoefficients(rb_local, d, [deg, deg], quad_d, quad_1D_d)
            ########### END of NORMALIZING Pressure
            
            ########### START of VTK file 
            # vtk.export_as_vtk(rb_local, d, true_velocity=exact_solution, true_pressure=exact_solution_l2,vtk_path=f"bezier_mesh_level{ilevel}_deg{deg}.vtk")
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
    manufactured_sol_degrees_clean()
