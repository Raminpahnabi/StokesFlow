#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:15:18 2026

@author: raminpahnabi
"""
import sys
import os
import math
import numpy as np
from pathlib import Path

os.environ["SWEEPS_API_PATH"] = "/Users/raminpahnabi/Documents/BYU/sweeps/build/src/api"

from sweeps_path import ensure_sweeps_api_on_path

PROJECT_ROOT = Path(__file__).resolve().parent
ensure_sweeps_api_on_path()
sys.path.append(str(PROJECT_ROOT / 'HWs'))
sys.path.append(str(PROJECT_ROOT / 'Required'))

import splines as spline
import CommonFuncs as cf
import BoundaryConditions as bc 
import Inputfile as inp
import NormalizedPressure as npre
import Solver_L2Projection as ls
import Solver_StokesFlow as ss
import Solver_NonlinearNavierStokes as nss
# import Problem_Setup as ps


def export_as_vtk(basis, dtotal, true_velocity=None, true_pressure=None, vtk_path=None): 
    
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
    if vtk_path is None:
        if L2Projection:
            vtk_path = 'solution_l2projection.vtk'
        elif Stokes:
            vtk_path = 'solution_stokes.vtk'
        elif NavierStokes:
            vtk_path = 'solution_navierstokes.vtk'
        elif JetNavierStokes:  
            vtk_path = 'solution_jet_navierstokes.vtk'
        
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


def _quad_area(corners):
    """Signed area of a planar quad using the shoelace formula."""
    total = 0.0
    n = len(corners)
    for i in range(n):
        j = (i + 1) % n
        total += corners[i][0] * corners[j][1]
        total -= corners[j][0] * corners[i][1]
    return abs(total) * 0.5


def export_bezier_mesh_vtk(basis, vtk_path=None):
    """
    Write a VTK file with one quad cell per Bezier element.
    CELL_DATA field 'refinement_level' encodes hierarchy depth:
      level 0 = coarsest (no refinement),
      level 1 = once refined, etc.

    Level is inferred from element physical area relative to the
    largest element: level = round(log4(area_max / area_elem)).
    This is exact for smooth geometries where each h-refinement
    step halves element size in both parametric directions.

    vtk_path: optional filename override; if None the name is derived
              from the active problem type flags.
    """
    # -----------------------------------------------------------------------
    # Gather element corners in physical space
    # -----------------------------------------------------------------------
    elem_corners = []   # list of [(x0,y0),(x1,y1),(x2,y2),(x3,y3)]

    LOCAL_CORNERS = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    for elem in basis.elements():
        basis.localizeElement(elem)
        corners = []
        for xi, eta in LOCAL_CORNERS:
            basis.localizePoint([xi, eta])
            xy = basis.mapping()
            corners.append((float(xy[0]), float(xy[1])))
        elem_corners.append(corners)

    # -----------------------------------------------------------------------
    # Determine refinement level from physical element area
    # -----------------------------------------------------------------------
    areas = [_quad_area(c) for c in elem_corners]
    max_area = max(areas) if areas else 1.0

    levels = []
    for a in areas:
        if a <= 0.0 or max_area <= 0.0:
            levels.append(0)
        else:
            ratio = max_area / a
            # level = log_4(ratio) ; round to nearest integer
            level = int(round(math.log(max(ratio, 1.0)) / math.log(4.0)))
            levels.append(level)

    # -----------------------------------------------------------------------
    # Build VTK arrays — one unique set of 4 points per element so that
    # non-conforming interfaces are represented faithfully.
    # -----------------------------------------------------------------------
    n_elems = len(elem_corners)
    all_points = []
    quad_cells = []
    pt_idx = 0

    for corners in elem_corners:
        indices = []
        for c in corners:
            all_points.append((c[0], c[1], 0.0))
            indices.append(pt_idx)
            pt_idx += 1
        quad_cells.append(indices)

    n_pts   = len(all_points)
    n_cells = n_elems

    # -----------------------------------------------------------------------
    # Choose output filename to mirror the solution file naming
    # -----------------------------------------------------------------------
    if vtk_path is None:
        if L2Projection:
            vtk_path = 'bezier_mesh_l2projection.vtk'
        elif Stokes:
            vtk_path = 'bezier_mesh_stokes.vtk'
        elif NavierStokes:
            vtk_path = 'bezier_mesh_navierstokes.vtk'
        elif JetNavierStokes:
            vtk_path = 'bezier_mesh_jet_navierstokes.vtk'
        else:
            vtk_path = 'bezier_mesh.vtk'

    print(f"Writing Bezier mesh {vtk_path}  ({n_cells} elements) ...")

    with open(vtk_path, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Bezier Mesh\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n\n")

        # Points
        f.write(f"POINTS {n_pts} float\n")
        np.savetxt(f, np.array(all_points), fmt="%.8f")
        f.write("\n")

        # Cells (VTK_QUAD = type 9, 4 nodes each)
        total_int = n_cells * 5
        f.write(f"CELLS {n_cells} {total_int}\n")
        cells_arr = np.hstack([np.full((n_cells, 1), 4, dtype=int),
                               np.array(quad_cells, dtype=int)])
        np.savetxt(f, cells_arr, fmt="%d")
        f.write("\n")

        f.write(f"CELL_TYPES {n_cells}\n")
        np.savetxt(f, np.full((n_cells, 1), 9, dtype=int), fmt="%d")
        f.write("\n")

        # Cell data: refinement level
        f.write(f"CELL_DATA {n_cells}\n")
        f.write("SCALARS refinement_level int 1\n")
        f.write("LOOKUP_TABLE default\n")
        np.savetxt(f, np.array(levels, dtype=int).reshape(-1, 1), fmt="%d")
        f.write("\n")

    print(f"Done.  Open  {os.path.abspath(vtk_path)}  in ParaView.")


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
ifID                    = inp.ifID
nu                      = inp.KINEMATIC_VISCOSITY
use_curve_geometry      = inp.USE_CURVED_GEOMETRY
L2Projection            = inp.is_L2Projection
Stokes                  = inp.is_Stokes
NavierStokes            = inp.is_NavierStokes
JetNavierStokes         = inp.is_JetNavierStokes
option                  = inp.option_number

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

elif JetNavierStokes:
            forcing_function        = inp.forcing_function_jet_3
            f_ns                    = inp.forcing_function_jet_3
            exact_solution          = inp.exact_solution_jet_3
            exact_solution_l2       = inp.exact_solution_l2_jet_3
            boundary_value_function = inp.boundary_value_function_jet_3 



############# Stokes ############
if L2Projection:
    basis = spline.NavierStokesTPDiscretization(kv1, kv2, degree1, degree2, cpts)
    nref = 2**2
    refined_basis = spline.globallyHRefine(basis, nelem1*nelem2*nref, parametric_tolerance=1e-5)

    print(f"Solving L2_Projection system — EXPONENTIAL solution (degree={degree1}) ...")
    dtotal = ls.L2Projection(refined_basis, degs, quad, quad_1D, gamma,
                        forcing_function, exact_solution,
                        boundary_conditions=None, boundary_value_function=boundary_value_function, ifID=ifID,nu=nu,use_curve_geometry =use_curve_geometry)
elif Stokes:
    kv1_unit = spline.KnotVector([0]*degree1 + [0, 1] + [1]*degree1, 1e-9)
    kv2_unit = spline.KnotVector([0]*degree2 + [0, 1] + [1]*degree2, 1e-9)
    cpts_base = spline.grevillePoints(kv1_unit, kv2_unit, degree1, degree2)

    basis_coarse = spline.NavierStokesTPDiscretization(kv1_unit, kv2_unit, degree1, degree2, cpts_base)

    n_div = 2**4
    global_refined_basis = spline.globallyHRefine(basis_coarse, n_div, parametric_tolerance=1e-5)
    
    kv1_d, kv2_d = global_refined_basis.knotVectors()
    cpts_d = global_refined_basis.control_points
    elems_to_refine = []
    elems_to_refine.append([[1,1],[1,2],[1,3],[1,4],[2,1],[2,2],[2,3],[2,4],[3,1],[3,2],[3,3],[3,4],[4,1],[4,2],[4,3],[4,4]])#, [q, 5], [q, 6]])  # Refine one element column
  
    refined_basis = spline.NavierStokesHierarchicalDiscretization(
        kv2_d, kv1_d, degree1, degree2, cpts_d, elems_to_refine
    )


    print("\nSolving Stokes (initial guess) ...")
    dtotal = ss.Stokes(refined_basis, degs, quad, quad_1D, gamma,
                         forcing_function, exact_solution,
                         boundary_conditions=None,
                         boundary_value_function=boundary_value_function,
                         ifID=True, nu=nu) 

elif NavierStokes:
    basis = spline.NavierStokesTPDiscretization(kv1, kv2, degree1, degree2, cpts)
    nref = 2**2
    refined_basis = spline.globallyHRefine(basis, nelem1*nelem2*nref, parametric_tolerance=1e-5)

    print(f"Solving NavierStokes system — EXPONENTIAL solution (degree={degree1}) ...")
    dinitial = ss.Stokes(refined_basis, degs, quad, quad_1D, gamma,
                               forcing_function, exact_solution,
                               boundary_conditions=None,
                               boundary_value_function=boundary_value_function,
                               ifID=True,nu=nu)
    
    dtotal = nss.NavierStokes(refined_basis, degs, quad, quad_1D, gamma, forcing_function, f_ns, exact_solution,
                                      boundary_conditions=None, boundary_value_function=boundary_value_function, ifID=ifID,
                                      d_initial=dinitial, nu=nu) 

elif JetNavierStokes:
    L      = 8.0    # domain length
    H      = 1.0    # domain height
    D_half = 0.5    # half-width of inflow nozzle (D/2)
    U      = 1.0    # inflow speed
    Re     = 50     
    nu     = U * H / Re   # kinematic viscosity = 0.02

    kv1_unit = spline.KnotVector([0]*degree1 + [0, 1] + [1]*degree1, 1e-9)
    kv2_unit = spline.KnotVector([0]*degree2 + [0, 1] + [1]*degree2, 1e-9)
    cpts_base = spline.grevillePoints(kv1_unit, kv2_unit, degree1, degree2)
    cpts_base[0, :] *= L    # scale x-coordinates from [0,1] to [0,L=8]
                            # y-coordinates stay in [0,1] = [0,H] since H=1

    basis_coarse = spline.NavierStokesTPDiscretization(kv1_unit, kv2_unit, degree1, degree2, cpts_base)

    n_div = 2**5  # nozzle is 0.5/8 of domain; need ≥16 elements
    refined_basis = spline.globallyHRefine(basis_coarse, n_div, parametric_tolerance=1e-5)

    #  skip_nitsche_faces: don't apply Nitsche tangential penalty on these faces
    #  left  → symmetry wall: free tangential, only u.n=0 is strongly enforced
    #  right → outflow:       natural BC, nothing enforced
    skip_nitsche    = ['left', 'right']

    #  skip_prescribed_faces: don't prescribe normal DOFs on these faces (they are free)
    #  right → outflow: normal flux determined by NS equations
    skip_prescribed = ['right']
    

    print("\nSolving Stokes (initial guess) ...")
    d_stokes = ss.Stokes(refined_basis, degs, quad, quad_1D, gamma,
                         forcing_function, exact_solution,
                         boundary_conditions=None,
                         boundary_value_function=boundary_value_function,
                         ifID=True, nu=nu) 
    print(f"Stokes done. |d|={np.linalg.norm(d_stokes):.4e}")

    print(f"\nSolving Navier-Stokes  Re={Re}  nu={nu:.4f} ...")
    dtotal = nss.NavierStokes(refined_basis, degs, quad, quad_1D, gamma,
                            forcing_function, f_ns, exact_solution,
                            boundary_conditions=None,
                            boundary_value_function=boundary_value_function,
                            ifID=True, nu=nu,
                            d_initial=d_stokes)  
    print(f"NS done. |d|={np.linalg.norm(dtotal):.4e}")


n_l2       = refined_basis.L2.numTotalFunctions()                                  
alpha      = npre.EvaluateAveragePressure(refined_basis, dtotal, quad)      #(α = ∫_Ω p_h dΩ)
# area_value = sum(cn.compute_all_element_areas(refined_basis, quad))       #(vol = ∫_Ω dΩ)
print("avg pressure before normalization:", alpha)  

alpha_rb      = npre.EvaluateAveragePressure(refined_basis, dtotal, quad)               
average_pressure_after = npre.NormalizePressureCoefficients(refined_basis, dtotal, degs, quad, quad_1D)

# dtotal[-n_l2:] -= alpha / area_value                                
# average_pressure_after = npre.EvaluateAveragePressure(refined_basis, dtotal, quad) 
print("avg pressure after  normalization:", average_pressure_after) 
print("Solver done.")

if __name__ == '__main__':
    export_as_vtk(refined_basis, dtotal, true_velocity=exact_solution, true_pressure=exact_solution_l2)
    # export_bezier_mesh_vtk(refined_basis)
