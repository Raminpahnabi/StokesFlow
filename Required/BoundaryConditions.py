#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:39:29 2026

@author: raminpahnabi
"""

import sys
import os
import numpy as np
sys.path.insert(0, '/Users/raminpahnabi/Documents/BYU/sweeps/build/src/api')
sys.path.append(os.path.join(os.getcwd(), '../HWs'))

import Quadrature_Operations_Solutions_boundary as gq_bc
import CommonFuncs as cf

def GetBoundaryDOFs(basis):
    """
    Returns a dict of global DOF index sets for each face, split by role.

    comp1: n_bf_x × (n_bf_y-1)  e.g. 4×3 grid  (/\ arrows)
           TANGENTIAL  on BOTTOM and TOP
           NORMAL on LEFT  and RIGHT

    comp2: (n_bf_x-1) × n_bf_y  e.g. 3×4 grid  (> arrows)
           NORMAL on BOTTOM and TOP
           TANGENTIAL   on LEFT  and RIGHT

    Returns:
        {
          'bottom': {'normal': [...], 'tangential': [...]},
          'top':    {'normal': [...], 'tangential': [...]},
          'left':   {'normal': [...], 'tangential': [...]},
          'right':  {'normal': [...], 'tangential': [...]},
          'all_normal':     set of all normal DOFs,
          'all_tangential': set of all tangential DOFs,
        }
    """
    degs = cf.GetSplineDegree(basis)
    knot_vecs = basis.knotVectors()
    num_h1_bfs = []
    for i in range(len(degs)):
        num_h1_bfs.append(
            len(str(knot_vecs[i]).replace("}","").replace("{","").split(",")[:-1]) - (degs[i]+1)
        )

    n_bf_x = num_h1_bfs[0]   # e.g. 4
    n_bf_y = num_h1_bfs[1]   # e.g. 4

    # comp1 grid dimensions  (normal on bottom/top)
    nc1 = n_bf_x          # 4 columns
    nr1 = n_bf_y - 1      # 3 rows
    # comp2 grid dimensions  (normal on left/right)
    nc2 = n_bf_x - 1      # 3 columns
    nr2 = n_bf_y          # 4 rows

    offset = nc1 * nr1    # where comp2 starts in global numbering  (= 12)

    # ── BOTTOM face ──────────────────────────────────────────────────────────
    # Tangential = comp2 first row  (row 0 of 3×4 grid)
    bottom_tangential     = [k for k in range(nc1)]                # [0,1,2,3]
    # Normal  = comp1 first row  (row 0 of 4×3 grid)
    bottom_normal = [offset + k for k in range(nc2)]               # [12,13,14]

    # ── TOP face ─────────────────────────────────────────────────────────────
    # Tangential = comp2 last row  (row nr2-1)
    top_normal_start  = (nr1 - 1) * nc1
    top_tangential  = [top_normal_start + k for k in range(nc1)]     # [8,9,10,11]
    
    # Normal  = comp1 last row  (row nr1-1)
    top_tang_start    = offset + (nr2 - 1) * nc2
    top_normal    = [top_tang_start + k for k in range(nc2)]       # [21,22,23]

    # ── LEFT face ────────────────────────────────────────────────────────────
    # Tangential = comp1 first column (col 0 of 4×3 grid)
    left_tangential       = [offset + i * nc2 for i in range(nr2)]         # [12,15,18,21]
    # Normal  = comp2 first column (col 0 of 3×4 grid)
    left_normal   = [i * nc1 for i in range(nr1)]                  # [0,4,8]

    # ── RIGHT face ───────────────────────────────────────────────────────────
    # Tangential = comp1 last column (col nc1-1 of 4×3 grid)
    right_tangential      = [offset + (i+1)*nc2 - 1 for i in range(nr2)]  # [14,17,20,23]
    
    # Normal  = comp2 last column (col nc2-1 of 3×4 grid)
    right_normal  = [(i+1)*nc1 - 1 for i in range(nr1)]           # [3,7,11]

    all_normal     = set(bottom_normal + top_normal + left_normal + right_normal)
    all_tangential = set(bottom_tangential + top_tangential + left_tangential + right_tangential)

    return {
        'bottom': {'normal': bottom_normal,   'tangential': bottom_tangential},
        'top':    {'normal': top_normal,       'tangential': top_tangential},
        'left':   {'normal': left_normal,      'tangential': left_tangential},
        'right':  {'normal': right_normal,     'tangential': right_tangential},
        'all_normal':     all_normal,
        'all_tangential': all_tangential,
    }


# ===========================================================================
#   Compute PRESCRIBED VALUES for normal DOFs (g value)
#       (needed for strong enforcement)
# ===========================================================================
def ComputePrescribedNormalDOFValues(basis, boundary_dofs, boundary_value_function, quad_1D):
    """
    For each normal boundary DOF, compute the prescribed value by L2 projection
    of (u_exact · n) onto the boundary.

    Returns:
        prescribed: dict {global_dof_index: value}
    """
    n_hdiv_1, n_hdiv_2 = cf.GetNumberH1FirstComponent(basis)
    n_hdiv = n_hdiv_1 + n_hdiv_2

    # We accumulate: mass_vec[A] and rhs_vec[A] for each normal DOF A
    # then prescribed[A] = rhs_vec[A] / mass_vec[A]
    # (scalar L2 projection along boundary)
    all_normal = boundary_dofs['all_normal']

    mass_vec = np.zeros(n_hdiv)
    rhs_vec  = np.zeros(n_hdiv)

    bdry_map = {
        gq_bc.BoundaryFace.BOTTOM: boundary_dofs['bottom']['normal'],
        gq_bc.BoundaryFace.TOP:    boundary_dofs['top']['normal'],
        gq_bc.BoundaryFace.LEFT:   boundary_dofs['left']['normal'],
        gq_bc.BoundaryFace.RIGHT:  boundary_dofs['right']['normal'],
    }

    for elem in basis.elements():
        basis.localizeElement(elem)
        local_IEN = basis.HDIV.connectivity(elem)

        for bdry, normal_global_dofs in bdry_map.items():
            normal_global_set = set(normal_global_dofs)
            xi_vals = gq_bc.GetFaceQuadraturePoints(quad_1D, bdry)

            # rotation matrix for outward normal
            if bdry in (gq_bc.BoundaryFace.TOP, gq_bc.BoundaryFace.LEFT):
                R = np.array([[0,-1],[1,0]])
            else:
                R = np.array([[0,1],[-1,0]])

            for g in range(len(xi_vals)):
                quad_pts = xi_vals[g]
                basis.localizePoint(quad_pts)

                jac    = basis.jacobian()
                tangent = cf.DifferentialVector(jac, bdry)
                jac_1d  = np.linalg.norm(tangent)
                normal_unit = np.dot(R, tangent) / jac_1d

                qpt_mapped = basis.mapping()
                x_g, y_g   = qpt_mapped[0], qpt_mapped[1]
                u_exact_val = np.array(boundary_value_function(x_g, y_g))
                u_n_exact   = np.dot(u_exact_val, normal_unit)   # scalar normal component

                phi_all = basis.piolaTransformedHDIVBasis()       # shape: (n_local, 2)
                wt = quad_1D.quad_wts[g]

                for a, A_global in enumerate(local_IEN):
                    if A_global in normal_global_set:
                        phi_a_dot_n = np.dot(phi_all[a], normal_unit)  # scalar
                        mass_vec[A_global] += phi_a_dot_n * phi_a_dot_n * wt * jac_1d
                        rhs_vec[A_global]  += phi_a_dot_n * u_n_exact  * wt * jac_1d

    # Compute prescribed values (avoid division by zero for interior DOFs)
    prescribed = {}
    for A in all_normal:
        if abs(mass_vec[A]) > 1e-14:
            prescribed[A] = rhs_vec[A] / mass_vec[A]
        else:
            prescribed[A] = 0.0
    
    # try:
    #     K_sp = csr_matrix(mass_vec)
    #     c = spsolve(K_sp, rhs_vec)
    # except Exception:
    #     c = np.zeros(len(all_normal))

    # for i, A in enumerate(all_normal):
    #     prescribed[A] = c[i]
    
    # print("prescribed:",prescribed)
    return prescribed