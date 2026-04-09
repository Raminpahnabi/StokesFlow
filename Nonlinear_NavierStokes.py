#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:51:16 2026

@author: raminpahnabi
"""
import sys
import os
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from pathlib import Path
import matplotlib.pyplot as plt  

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from sweeps_path import ensure_sweeps_api_on_path

ensure_sweeps_api_on_path()
sys.path.append(str(PROJECT_ROOT / 'HWs'))
sys.path.append(str(PROJECT_ROOT / 'Required'))

import LocalAssembly as la
import Nitsche as ni
import CommonFuncs as cf
import BoundaryConditions as bc
import splines as spline
import StokesFlow_Solver as ss
import NS_Inputfile as inp

import Gaussian_Quadrature_2D_Solution as gq_nD
import Convergence as cn
import NormalizedPressure as npre


max_knot_xi             = inp.max_knot_xi
max_knot_eta            = inp.max_knot_eta
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
use_curved_geometry     = inp.USE_CURVED_GEOMETRY
f_ns                    = inp.forcing_function_ns
nu                      = inp.KINEMATIC_VISCOSITY

basis = spline.NavierStokesTPDiscretization( kv1, kv2, degree1, degree2, cpts)

# Initial Solution
example_d_check = ss.Stokes(basis, degree1, quad, quad_1D, gamma,
                    forcing_function, exact_solution,
                    boundary_conditions=None, boundary_value_function=boundary_value_function, ifID=ifID,nu=nu)
print("Initial_Solu:", example_d_check) # With the Stokes equation(NavierStokes equation with Re<<<1)


#######################################################################
##################     NavierStokes_Flow_div_free     #################
#######################################################################
def NavierStokes(basis, deg, gaussian, quad_1D, gamma, f, f_ns, u_exact, boundary_conditions, boundary_value_function,
           ifID=True, curve_domain=False, geo_map_func=None, d_initial=None, nu=nu):  #ns nu=1 default; override when calling (e.g. nu=0.1)

    n_hdiv_1_comp = cf.GetNumberH1FirstComponent(basis)[0]
    n_hdiv_2_comp = cf.GetNumberH1FirstComponent(basis)[1]
    n_hdiv = n_hdiv_1_comp + n_hdiv_2_comp
    # n_hdiv = basis.HDIV.numTotalFunctions()
    n_l2 = basis.L2.numTotalFunctions()
    n_total_funcs = n_hdiv + n_l2

    boundary_dofs = bc.GetBoundaryDOFs(basis)
    all_tangential = set(boundary_dofs['all_tangential'])
    all_normal = set(boundary_dofs['all_normal'])
    
    # Compute prescribed values for normal DOFs
    prescribed = bc.ComputePrescribedNormalDOFValues(basis, boundary_dofs, boundary_value_function, quad_1D)
    # Build ID array (marks normal DOFs as -1)
    ID = cf.ID_array(basis.HDIV, basis.L2, boundary_dofs, prescribed)
    
    n = max(ID)+1
    
    f_ns_nu = f_ns  #ns bind nu into the NS forcing so LocalForceStokes gets the right nu
    f_stokes_nu = f #ns bind nu into the Stokes forcing used for the initial guess

    if d_initial is not None:
        d_prev = d_initial
    else:
        d_prev = ss.Stokes(basis, deg, gaussian, quad_1D, gamma, f_stokes_nu, u_exact,  
                           boundary_conditions, boundary_value_function, ifID=ifID, nu=nu)
    
    max_iter = 40  
    tol = 1e-8  

    for k in range(max_iter):  
        K = lil_matrix((n, n))  
        F = np.zeros(n)  

        for e in basis.elements():  
            basis.localizeElement(e) 
            ke = la.LocalStiffnessStokes(basis, deg, gaussian, quad_1D, e, boundary_conditions, nu=nu)  
            fe = la.LocalForceStokes(basis, deg, gaussian, quad_1D, gamma, e, f_ns_nu, nu=nu)  
            ke_Nitsche = ni.LocalStiffnessMatrix_Nitsche_IGA_2D(basis, deg, gaussian, quad_1D, gamma, e, nu=nu)  
            fe_Nitsche = ni.LocalForceVector_Nitsche_IGA_2D(basis, deg, gaussian, quad_1D, gamma, e, f_ns_nu, u_exact, boundary_value_function, nu=nu)  
            
            # ke_adv = LocalAdvectionPicard(basis, gaussian, e, d_prev)
            ke_adv = la.LocalAdvectionPicard(basis, deg, gaussian, quad_1D, e, d_prev, boundary_conditions)

            local_IEN_HDIV = basis.HDIV.connectivity(e)  
            n_local_hdiv = len(local_IEN_HDIV)  
            local_IEN_L2 = basis.L2.connectivity(e)  
            n_local_L2 = len(local_IEN_L2)  
            
            # ke_total = ke + ke_Nitsche + ke_adv  
            # fe_total = fe + fe_Nitsche  
        
            ke_total2 = ke + ke_adv  

            for a in range(n_local_hdiv):
                A = local_IEN_HDIV[a]
                P = ID[A]
                if P == -1:
                    continue
                F[P] += fe[a] + fe_Nitsche[a]

                for b in range(n_local_hdiv):
                    B = local_IEN_HDIV[b]
                    Q = ID[B]
                    if Q == -1:
                        u_B = prescribed.get(B, 0.0)
                        F[P] -= ke_total2[a, b] * u_B       # ke + ke_adv contribution
                        F[P] -= ke_Nitsche[a, b] * u_B      # Nitsche contribution (matches Stokes)
                        continue
                    K[P, Q] += ke_total2[a, b]              # ke + ke_adv
                    K[P, Q] += ke_Nitsche[a, b]             # Nitsche velocity-velocity

                for b in range(n_local_L2):
                    B = n_hdiv + local_IEN_L2[b]
                    Q = ID[B]
                    if Q == -1:
                        u_B = prescribed.get(B, 0.0)
                        F[P] -= ke_total2[a, b + n_local_hdiv] * u_B
                        continue
                    K[P, Q] += ke_total2[a, b + n_local_hdiv]   # pressure-velocity (ke only, same as Stokes)
                    K[Q, P] += ke_total2[b + n_local_hdiv, a]   # symmetric

        d_reduced = spsolve(K.tocsr(), F)  
        
        d_picard = cf.ExtractTotalD(ID, d_reduced, prescribed, n_hdiv, n_l2) 
        d_next = d_picard
         
        rel_change = np.linalg.norm(d_next[:n_hdiv] - d_prev[:n_hdiv]) / max(np.linalg.norm(d_next[:n_hdiv]), 1e-14)  
        print(f"Picard iter {k+1:02d}: rel_change = {rel_change:.3e}")  
        d_prev = d_next  
        if rel_change < tol:  
            print(f"Picard converged in {k+1} iterations.")  
            break  

    return d_prev  


d_ns = NavierStokes(basis, degree1, quad, quad_1D, gamma, forcing_function, f_ns, exact_solution,
                    boundary_conditions=None, boundary_value_function=boundary_value_function, ifID=ifID,
                    d_initial=example_d_check, nu=nu) 
print("NavierStokes solution:", d_ns)  


def manufactured_sol_degrees_clean_ns():  
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
        n_quad_d  = deg * 2 
        quad_d    = gq_nD.GaussQuadrature2D(n_quad_d, n_quad_d, interval_d, interval_d)  
        quad_1D_d = gq_nD.GaussQuadrature1D(n_quad_d, start_pt=interval_d[0], end_pt=interval_d[1])  
        gamma_d   = 20 * deg**3  #ns use same gamma as Stokes (which gives perfect rates); 5*(deg+1) was far too small for degrees 3+ causing Nitsche BC enforcement to fail

        # Build coarsest single-element basis for this degree  
        kv1_d = spline.KnotVector([0]*deg + [0, max_knot_d_xi] + [max_knot_d_xi]*deg, 1e-9)  
        kv2_d = spline.KnotVector([0]*deg + [0, max_knot_d_eta] + [max_knot_d_eta]*deg, 1e-9)  
        unitkv1_d = spline.KnotVector([0]*deg + [0, 1] + [1]*deg, 1e-9)
        unitkv2_d = spline.KnotVector([0]*deg + [0, 1] + [1]*deg, 1e-9)
        cpts_d = spline.grevillePoints(unitkv1_d, unitkv2_d, deg, deg)  
        basis_d = spline.NavierStokesTPDiscretization(kv1_d, kv2_d, deg, deg, cpts_d)  

        errors   = []
        errors_p = []  
        h_values = []

        print("level | n_divisions | h           | error")  
        for ilevel, n_div in enumerate(refinement_levels):  
            rb = spline.globallyHRefine(basis_d, n_div, parametric_tolerance=1e-5)  

            d = NavierStokes(rb, [deg,deg], quad_d, quad_1D_d, gamma_d, forcing_function, f_ns, exact_solution, boundary_conditions=None,
                             boundary_value_function=boundary_value_function,
                             ifID=True, curve_domain=False, geo_map_func=None, nu=nu)  

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

manufactured_sol_degrees_clean_ns()
