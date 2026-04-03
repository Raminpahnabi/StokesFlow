#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:15:18 2026

@author: raminpahnabi
"""
import sys
import os
import numpy as np
from pathlib import Path
from sweeps_path import ensure_sweeps_api_on_path

PROJECT_ROOT = Path(__file__).resolve().parent
ensure_sweeps_api_on_path()
sys.path.append(str(PROJECT_ROOT / 'HWs'))
sys.path.append(str(PROJECT_ROOT / 'Required'))

import splines as spline
import StokesFlow_Solver as ss
import CommonFuncs as cf
import Inputfile as inp
import Convergence as cn
import NormalizedPressure as npre

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

basis = spline.NavierStokesTPDiscretization(kv1, kv2, degree1, degree2, cpts)

nref = 2**3
refined_basis = spline.globallyHRefine(basis, nelem1*nelem2*nref, parametric_tolerance=1e-5)

print(f"Solving Stokes system — EXPONENTIAL solution (degree={degree1}) ...")
dtotal = ss.Stokes(refined_basis, degs, quad, quad_1D, gamma,
                   forcing_function, exact_solution,
                   boundary_conditions=None,
                   boundary_value_function=boundary_value_function,
                   ifID=True)
n_l2       = refined_basis.L2.numTotalFunctions()                                    #NEWCODE
alpha      = npre.EvaluateAveragePressure(refined_basis, dtotal, quad)      #NEWCODE  (α = ∫_Ω p_h dΩ)
area_value = sum(cn.compute_all_element_areas(refined_basis, quad))                  #NEWCODE  (vol = ∫_Ω dΩ)
print("avg pressure before normalization:", alpha)                            #NEWCODE
dtotal[-n_l2:] -= alpha / area_value                                #NEWCODE  (d_n = d_p - α/vol)
average_pressure_after = npre.EvaluateAveragePressure(refined_basis, dtotal, quad)  #NEWCODE
print("avg pressure after  normalization:", average_pressure_after) 
print("Solver done.")

export_as_vtk(refined_basis, dtotal, true_velocity=exact_solution, true_pressure=exact_solution_l2)
