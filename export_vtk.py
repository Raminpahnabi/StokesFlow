#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Stokes flow solution to VTK format for ParaView visualization.

Run this script (same working directory as StokesFlow_problem.py).
It produces  stokes_solution.vtk  which you can open directly in ParaView.

Each IGA element is sampled on a uniform N_SAMPLE x N_SAMPLE grid in
parametric space.  The quads are written as VTK_QUAD cells so ParaView
renders a smooth surface.  Point-data arrays exported:
  - velocity  (vector, 3-component)
  - pressure  (scalar)
  - velocity_magnitude (scalar, convenient for colour-mapping)
"""

import sys
import os
sys.path.insert(0, '/Users/raminpahnabi/Documents/BYU/sweeps/build/src/api')
sys.path.append(os.path.join(os.getcwd(), 'HWs'))
sys.path.append(os.path.join(os.getcwd(), 'Required'))

import splines as spline
import numpy as np
import Gaussian_Quadrature_2D_Solution as gq_nD
import StokesFlow_Solver as ss
import CommonFuncs as cf

# -----------------------------------------------------------------------
# Solver setup  (mirror of StokesFlow_problem.py)
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

max_knot = 1
min_knot = 0
degree1  = 2
degree2  = 2
nelem1   = 2
nelem2   = 2
degs     = [degree1, degree2]

kv1_init = list(np.linspace(0, max_knot, nelem1+1))
kv2_init = list(np.linspace(0, max_knot, nelem2+1))
kv1 = spline.KnotVector([0]*degree1 + kv1_init + [max_knot]*degree1, 1e-9)
kv2 = spline.KnotVector([0]*degree2 + kv2_init + [max_knot]*degree2, 1e-9)

cpts  = spline.grevillePoints(kv1, kv2, degree1, degree2)
basis = spline.NavierStokesTPDiscretization(kv1, kv2, degree1, degree2, cpts)

n_quad   = max(degree1, degree2) + 1
interval = [0, 1]
quad     = gq_nD.GaussQuadrature2D(n_quad, n_quad, interval, interval)
quad_1D  = gq_nD.GaussQuadrature1D(n_quad, start_pt=interval[0], end_pt=interval[1])
gamma    = 20 * max(degree1, degree2)**2
ifID     = True

nref = 3
refined_basis = spline.globallyHRefine(basis, nelem1*nelem2*nref, parametric_tolerance=1e-5)

print("Solving Stokes system ...")
dtotal = ss.Stokes(refined_basis, degs, quad, quad_1D, gamma,
                   forcing_function, exact_solution,
                   boundary_conditions=None,
                   boundary_value_function=boundary_value_function,
                   ifID=ifID)
print("Solver done.")

# -----------------------------------------------------------------------
# Sample solution on a uniform parametric grid per element
# -----------------------------------------------------------------------
N_SAMPLE = 10          # grid points per element per direction (>=2)

n_hdiv_1_comp = cf.GetNumberH1FirstComponent(refined_basis)[0]
n_hdiv_2_comp = cf.GetNumberH1FirstComponent(refined_basis)[1]
n_hdiv_total  = n_hdiv_1_comp + n_hdiv_2_comp

xi_vals  = np.linspace(0.0, 1.0, N_SAMPLE)
eta_vals = np.linspace(0.0, 1.0, N_SAMPLE)

all_points   = []   # (x, y, 0)
all_velocity = []   # (ux, uy, 0)
all_pressure = []   # scalar

quad_cells   = []   # list of (p0, p1, p2, p3) VTK_QUAD

global_pt_idx = 0

for elem in refined_basis.elements():
    refined_basis.localizeElement(elem)
    local_IEN_hdiv = refined_basis.HDIV.connectivity(elem)
    local_IEN_L2   = refined_basis.L2.connectivity(elem)

    # Index grid for this element: shape (N_SAMPLE, N_SAMPLE)
    grid_idx = np.empty((N_SAMPLE, N_SAMPLE), dtype=int)

    for i, xi in enumerate(xi_vals):
        for j, eta in enumerate(eta_vals):
            refined_basis.localizePoint([xi, eta])

            # --- velocity ---
            phi_hdiv = refined_basis.piolaTransformedHDIVBasis()
            uh = np.zeros(2)
            for a, A in enumerate(local_IEN_hdiv):
                uh += dtotal[A] * phi_hdiv[a]

            # --- pressure ---
            phi_l2 = refined_basis.piolaTransformedL2()
            ph = 0.0
            for b, B in enumerate(local_IEN_L2):
                ph += dtotal[B + n_hdiv_total] * phi_l2[b]

            # --- physical coordinates ---
            xy = refined_basis.mapping()

            all_points.append((float(xy[0]), float(xy[1]), 0.0))
            all_velocity.append((float(uh[0]), float(uh[1]), 0.0))
            all_pressure.append(float(ph))

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

    # Points
    f.write(f"POINTS {n_pts} float\n")
    for (x, y, z) in all_points:
        f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")
    f.write("\n")

    # Cells  (each VTK_QUAD entry: count p0 p1 p2 p3)
    total_int = n_cells * 5   # 1 count + 4 indices per cell
    f.write(f"CELLS {n_cells} {total_int}\n")
    for (p0, p1, p2, p3) in quad_cells:
        f.write(f"4 {p0} {p1} {p2} {p3}\n")
    f.write("\n")

    # Cell types  (9 = VTK_QUAD)
    f.write(f"CELL_TYPES {n_cells}\n")
    for _ in quad_cells:
        f.write("9\n")
    f.write("\n")

    # Point data
    f.write(f"POINT_DATA {n_pts}\n")

    # Velocity vector
    f.write("VECTORS velocity float\n")
    for (ux, uy, uz) in all_velocity:
        f.write(f"{ux:.8f} {uy:.8f} {uz:.8f}\n")
    f.write("\n")

    # Pressure scalar
    f.write("SCALARS pressure float 1\n")
    f.write("LOOKUP_TABLE default\n")
    for p in all_pressure:
        f.write(f"{p:.8f}\n")
    f.write("\n")

    # Velocity magnitude (handy for colour plots)
    f.write("SCALARS velocity_magnitude float 1\n")
    f.write("LOOKUP_TABLE default\n")
    for (ux, uy, _) in all_velocity:
        f.write(f"{np.sqrt(ux*ux + uy*uy):.8f}\n")
    f.write("\n")

print(f"Done.  Open  {os.path.abspath(vtk_path)}  in ParaView.")
