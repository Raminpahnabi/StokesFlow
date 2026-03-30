#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:47:20 2026

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
import CommonFuncs as cf

def export_as_vtk(basis, dtotal, true_velocity = None, true_pressure = None):        
    
    # -----------------------------------------------------------------------
    # Sample solution on a uniform parametric grid per element
    # -----------------------------------------------------------------------
    N_SAMPLE = 10          # grid points per element per direction (>=2)
    
    n_hdiv_1_comp = cf.GetNumberH1FirstComponent(basis)[0]
    n_hdiv_2_comp = cf.GetNumberH1FirstComponent(basis)[1]
    n_hdiv_total  = n_hdiv_1_comp + n_hdiv_2_comp
    
    xi_vals  = np.linspace(0.0, 1.0, N_SAMPLE)
    eta_vals = np.linspace(0.0, 1.0, N_SAMPLE)
    
    all_points   = []   # (x, y, 0)
    all_velocity = []   # (ux, uy, 0)
    all_pressure = []   # scalar
    all_velocity_true = []
    all_velocity_err = []
    all_pressure_err = []
    
    quad_cells   = []   # list of (p0, p1, p2, p3) VTK_QUAD
    
    global_pt_idx = 0
    
    for elem in basis.elements():
        basis.localizeElement(elem)
        local_IEN_hdiv = basis.HDIV.connectivity(elem)
        local_IEN_L2   = basis.L2.connectivity(elem)
    
        # Index grid for this element: shape (N_SAMPLE, N_SAMPLE)
        grid_idx = np.empty((N_SAMPLE, N_SAMPLE), dtype=int)
    
        for i, xi in enumerate(xi_vals):
            for j, eta in enumerate(eta_vals):
                basis.localizePoint([xi, eta])
    
                # --- velocity ---
                phi_hdiv = basis.piolaTransformedHDIVBasis()
                uh = dtotal[np.asarray(local_IEN_hdiv)] @ phi_hdiv

                # --- pressure ---
                phi_l2 = np.asarray(basis.piolaTransformedL2()).ravel()
                IEN_L2 = np.array([int(B) for B in local_IEN_L2])
                p_coeffs = np.asarray(dtotal[IEN_L2 + n_hdiv_total]).ravel()
                ph = float(np.dot(phi_l2, p_coeffs))
    
                # --- physical coordinates ---
                xy = basis.mapping()
    
                all_points.append((float(xy[0]), float(xy[1]), 0.0))
                all_velocity.append((float(uh[0]), float(uh[1]), 0.0))
                all_pressure.append(float(ph))
                
                if true_velocity != None:
                    u = true_velocity(xy[0],xy[1])
                    all_velocity_true.append((float(u[0]),float(u[1]),0.0))
                    all_velocity_err.append((float(u[0]-uh[0]), float(u[1]-uh[1]), 0.0))
                if true_pressure != None:
                    p = true_pressure(xy[0],xy[1])
                    all_pressure_err.append(float(p-ph))
    
                grid_idx[i, j] = global_pt_idx
                global_pt_idx += 1
    
        # Build quads from the grid
        for i in range(N_SAMPLE - 1):
            for j in range(N_SAMPLE - 1):
                quad_cells.append((grid_idx[i,   j],
                                    grid_idx[i+1, j],
                                    grid_idx[i+1, j+1],
                                    grid_idx[i,   j+1]))
    
    n_pts   = len(all_points)
    n_cells = len(quad_cells)
    
    # -----------------------------------------------------------------------
    # Write legacy VTK ASCII file
    # -----------------------------------------------------------------------
    vtk_path = 'stokes_solution.vtk'
    
    print(f"Writing {vtk_path}  ({n_pts} points, {n_cells} quads) ...")
    
    with open(vtk_path, 'w') as f:
        # Header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Stokes Flow Solution\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n\n")
    
        pts_arr = np.array(all_points)
        vel_arr = np.array(all_velocity)

        # Points
        f.write(f"POINTS {n_pts} float\n")
        np.savetxt(f, pts_arr, fmt="%.8f")
        f.write("\n")

        # Cells  (each VTK_QUAD entry: count p0 p1 p2 p3)
        total_int = n_cells * 5   # 1 count + 4 indices per cell
        f.write(f"CELLS {n_cells} {total_int}\n")
        cells_arr = np.hstack([np.full((n_cells, 1), 4, dtype=int),
                               np.array(quad_cells, dtype=int)])
        np.savetxt(f, cells_arr, fmt="%d")
        f.write("\n")

        # Cell types  (9 = VTK_QUAD)
        f.write(f"CELL_TYPES {n_cells}\n")
        np.savetxt(f, np.full((n_cells, 1), 9, dtype=int), fmt="%d")
        f.write("\n")

        # Point data
        f.write(f"POINT_DATA {n_pts}\n")

        # Velocity vector
        f.write("VECTORS velocity float\n")
        np.savetxt(f, vel_arr, fmt="%.8f")
        f.write("\n")

        # Velocity vector error
        if true_velocity != None:
            f.write("VECTORS velocity_err float\n")
            np.savetxt(f, np.array(all_velocity_err), fmt="%.8f")
            f.write("\n")
            f.write("VECTORS velocity_true float\n")
            np.savetxt(f, np.array(all_velocity_true), fmt="%.8f")
            f.write("\n")

        # Pressure scalar
        f.write("SCALARS pressure float 1\n")
        f.write("LOOKUP_TABLE default\n")
        np.savetxt(f, np.array(all_pressure), fmt="%.8f")
        f.write("\n")

        # Pressure scalar error
        if true_pressure != None:
            f.write("SCALARS pressure_err float 1\n")
            f.write("LOOKUP_TABLE default\n")
            np.savetxt(f, np.array(all_pressure_err), fmt="%.8f")
            f.write("\n")

        # Velocity magnitude (handy for colour plots)
        vel_mag = np.sqrt(vel_arr[:, 0]**2 + vel_arr[:, 1]**2)
        f.write("SCALARS velocity_magnitude float 1\n")
        f.write("LOOKUP_TABLE default\n")
        np.savetxt(f, vel_mag, fmt="%.8f")
        f.write("\n")

        # Velocity magnitude err (handy for colour plots)
        if true_velocity != None:
            vel_err_arr = np.array(all_velocity_err)
            vel_mag_err = np.sqrt(vel_err_arr[:, 0]**2 + vel_err_arr[:, 1]**2)
            f.write("SCALARS velocity_magnitude_err float 1\n")
            f.write("LOOKUP_TABLE default\n")
            np.savetxt(f, vel_mag_err, fmt="%.8f")
            f.write("\n")
        
    
    print(f"Done.  Open  {os.path.abspath(vtk_path)}  in ParaView.")


# -----------------------------------------------------------------------
# Switch:  True  -> trig manufactured solution (zero pressure, square domain)
#          False -> exponential manufactured solution 
# -----------------------------------------------------------------------
USE_TRIG = False

# -----------------------------------------------------------------------
# Exponential manufactured solution  (Option 4)
# -----------------------------------------------------------------------
KINEMATIC_VISCOSITY = 1

def forcing_function(x, y):
    f1 = np.exp(x) * (2 * y * (-2 + 7*y - 6*y*y + y**3)
                      - 8*x*y * (-2 + 7*y - 6*y*y + y**3)
                      + x*x * (12 - 36*y + 13*y*y - 2*y**3 + y**4)
                      + x**4 * (12 - 36*y + 13*y*y - 2*y**3 + y**4)
                      + x**3 * (-24 + 56*y + 30*y*y - 44*y**3 + 6*y**4))
    f2 = 2 * (228 - 456*y + np.exp(x) * (
              x**4 * (-5 + 7*y + 3*y*y + 2*y**3)
              + 2*x * (115 - 233*y + y*y + 6*y**3 - 2*y**4)
              - 3 * (76 - 152*y + y*y - 2*y**3 + y**4)
              + 2*x**3 * (19 - 41*y + 5*y*y - 2*y**3 + 2*y**4)
              + x*x * (-119 + 253*y - 3*y*y - 34*y**3 + 12*y**4)))
    return f1, f2

def exact_solution(x, y):
    u1 = 2 * np.exp(x) * (-1+x)**2 * x**2 * (y**2-y) * (-1+2*y)
    u2 = -np.exp(x) * (-1+x) * x * (-2+x*(3+x)) * (-1+y)**2 * y**2
    return u1, u2

def exact_solution_l2(x, y):
    p = (-424 + 156*np.exp(1)
          + (-y + y**2) * (-456 + np.exp(x) * (
              456 + x**2*(228 - 5*(-y+y**2))
              + 2*x*(-228 + (-y+y**2))
              + 2*x**3*(-36 + (-y+y**2))
              + x**4*(12 + (-y+y**2)))))
    return p

def boundary_value_function(x, y):
    return exact_solution(x, y)

# -----------------------------------------------------------------------
# Trig manufactured solution  (zero pressure, zero boundary values)
# -----------------------------------------------------------------------
def trig_exact_solution(x, y):
    u1 =  np.pi * np.sin(np.pi * x)**2 * np.sin(2 * np.pi * y)
    u2 = -np.pi * np.sin(2 * np.pi * x) * np.sin(np.pi * y)**2
    return u1, u2

def trig_forcing_function(x, y):
    f1 = 2 * np.pi**3 * (4 * np.sin(np.pi * x)**2 - 1) * np.sin(2 * np.pi * y)
    f2 = 2 * np.pi**3 * np.sin(2 * np.pi * x) * (1 - 4 * np.sin(np.pi * y)**2)
    return f1, f2

def trig_boundary_value(x, y):
    return 0.0, 0.0

def trig_exact_pressure(x, y):
    return 0.0

# -----------------------------------------------------------------------
# Solver setup
# -----------------------------------------------------------------------
interval = [0, 1]

if USE_TRIG:
    USE_CURVED  = True   # True → curved domain cpts  |  False → unit square
    degree      = 2 if USE_CURVED else 3   # curved cpts is 3×3 → requires degree 2
    n_div       = 16     # elements per direction → 16×16 mesh, h = 1/16

    n_quad  = degree + 1
    quad    = gq_nD.GaussQuadrature2D(n_quad, n_quad, interval, interval)
    quad_1D = gq_nD.GaussQuadrature1D(n_quad, start_pt=interval[0], end_pt=interval[1])
    gamma   = 20 * degree**3

    kv1 = spline.KnotVector([0]*degree + [0, 1] + [1]*degree, 1e-9)
    kv2 = spline.KnotVector([0]*degree + [0, 1] + [1]*degree, 1e-9)

    if USE_CURVED:
        # Curved physical domain — u_trig is NOT zero on this boundary,
        # so we pass trig_exact_solution as the boundary value function.
        cpts = np.array([
            [0.5, 0.5, 1.0,   0.25, 0.25, 1.0,   0.0, 0.0, 1.0],
            [0.0, 0.5, 0.5,   0.0,  0.75, 0.75,  0.0, 1.0, 1.0]
        ])
        bvf = trig_exact_solution
        domain_label = "curved"
    else:
        # Unit square — u_trig is exactly zero on the boundary.
        cpts = spline.grevillePoints(kv1, kv2, degree, degree)
        bvf = trig_boundary_value
        domain_label = "square"

    basis = spline.NavierStokesTPDiscretization(kv1, kv2, degree, degree, cpts)
    refined_basis = spline.globallyHRefine(basis, n_div, parametric_tolerance=1e-5)

    print(f"Solving Stokes system — TRIG/{domain_label} (degree={degree}, n_div={n_div}) ...")
    dtotal = ss.Stokes(refined_basis, [degree, degree], quad, quad_1D, gamma,
                       trig_forcing_function, trig_exact_solution,
                       boundary_conditions=None,
                       boundary_value_function=bvf,
                       ifID=True)
    print("Solver done.")

    export_as_vtk(refined_basis, dtotal,
                  true_velocity=trig_exact_solution,
                  true_pressure=trig_exact_pressure)

else:
    degree1 = 2
    degree2 = 2
    nelem1  = 2
    nelem2  = 2
    degs    = [degree1, degree2]

    n_quad  = max(degree1, degree2) + 1
    quad    = gq_nD.GaussQuadrature2D(n_quad, n_quad, interval, interval)
    quad_1D = gq_nD.GaussQuadrature1D(n_quad, start_pt=interval[0], end_pt=interval[1])
    gamma   = 20 * max(degree1, degree2)**2

    kv1_init = list(np.linspace(0, 1, nelem1+1))
    kv2_init = list(np.linspace(0, 1, nelem2+1))
    kv1 = spline.KnotVector([0]*degree1 + kv1_init + [1]*degree1, 1e-9)
    kv2 = spline.KnotVector([0]*degree2 + kv2_init + [1]*degree2, 1e-9)
    cpts = spline.grevillePoints(kv1, kv2, degree1, degree2)
    # cpts = np.array([
    #     [0.5, 0.5, 1.0,   0.25, 0.25, 1.0,   0.0, 0.0, 1.0],
    #     [0.0, 0.5, 0.5,   0.0,  0.75, 0.75,  0.0, 1.0, 1.0]
    # ])
    basis = spline.NavierStokesTPDiscretization(kv1, kv2, degree1, degree2, cpts)

    nref = 2**3
    refined_basis = spline.globallyHRefine(basis, nelem1*nelem2*nref, parametric_tolerance=1e-5)

    print(f"Solving Stokes system — EXPONENTIAL solution (degree={degree1}) ...")
    dtotal = ss.Stokes(refined_basis, degs, quad, quad_1D, gamma,
                       forcing_function, exact_solution,
                       boundary_conditions=None,
                       boundary_value_function=boundary_value_function,
                       ifID=True)
    print("Solver done.")

    export_as_vtk(refined_basis, dtotal,
                  true_velocity=exact_solution,
                  true_pressure=exact_solution_l2)

