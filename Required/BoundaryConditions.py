#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:39:29 2026

@author: raminpahnabi
"""

import sys
import os
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from sweeps_path import ensure_sweeps_api_on_path

ensure_sweeps_api_on_path()
sys.path.append(str(PROJECT_ROOT / 'HWs'))

import Quadrature_Operations_Solutions_boundary as gq_bc
import CommonFuncs as cf


def is_hierarchical(basis):  # True when basis is NavierStokesHierarchicalDiscretization (no knotVectors) 
    try:  
        basis.knotVectors()
        return False          # tensor-product: knotVectors() succeeded
    except Exception:       
        return True           # hierarchical: knotVectors() not available


def _lr_to_boundary_dofs(bc_lr):  # convert find_boundary_elements() output to boundary_dofs-compatible dict
    """
    Convert the find_boundary_elements() format
        bc_lr[face] = [normal_dofs_list, elements_set, ...]
    to the standard boundary_dofs format used by ComputePrescribedNormalDOFValues / ID_array:
        {'bottom': {'normal': [...], 'tangential': []}, ..., 'all_normal': set, 'all_tangential': set}

    NOTE: tangential sets are empty for hierarchical bases — Nitsche handles tangential weakly
    via boundary-face detection, not via a tangential-DOF list.
    """  
    result = {}  
    all_normal = set()  
    for face in ('bottom', 'top', 'left', 'right'):  
        normal = list(bc_lr[face][0])   # bc_lr[face][0] is the sorted list of normal-DOF indices
        result[face] = {'normal': normal, 'tangential': []}  
        all_normal |= set(normal)        
    result['all_normal']     = all_normal   
    result['all_tangential'] = set()        # no tangential DOF list needed for LR
    return result  


def GetBoundaryDOFs(basis, degs):
    """
    Returns a dict of global DOF index sets for each face, split by role.
    NOTE: only valid for tensor-product (TP) bases that expose knotVectors().
    For hierarchical (LR) bases use _lr_to_boundary_dofs(find_boundary_elements(basis)) instead.

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
    if is_hierarchical(basis):  # safety guard: LR bases must use _lr_to_boundary_dofs instead
        raise RuntimeError(  
            "GetBoundaryDOFs() called on a hierarchical (LR) basis — "  
            "use _lr_to_boundary_dofs(find_boundary_elements(basis)) instead."  
        )  
    degs = cf.GetSplineDegree(basis)
    # degs = degs
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
def ComputePrescribedNormalDOFValues(basis, boundary_dofs, boundary_value_function, quad_1D,
                                     skip_faces=None, use_curve_geometry=False):  #PE skip_faces: list of face names whose normal DOFs are left free (e.g. ['right'] for outflow)
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
    skip = set(skip_faces or [])  #PE set of face names to leave free (no strong enforcement)

    # only include faces that are not skipped
    all_normal = set()  #PE rebuild to exclude skipped-face DOFs
    for face in ('bottom', 'top', 'left', 'right'):
        if face not in skip:
            all_normal.update(boundary_dofs[face]['normal'])

    mass_vec = np.zeros(n_hdiv)
    rhs_vec  = np.zeros(n_hdiv)

    # build bdry_map only for faces that need strong enforcement
    bdry_map = {}
    if 'bottom' not in skip:
        bdry_map[gq_bc.BoundaryFace.BOTTOM] = boundary_dofs['bottom']['normal']
    if 'top' not in skip:
        bdry_map[gq_bc.BoundaryFace.TOP]    = boundary_dofs['top']['normal']
    if 'left' not in skip:
        bdry_map[gq_bc.BoundaryFace.LEFT]   = boundary_dofs['left']['normal']
    if 'right' not in skip:
        bdry_map[gq_bc.BoundaryFace.RIGHT]  = boundary_dofs['right']['normal']
    bounds = cf._physical_domain_bounds(basis)
    boundary_sets = cf._boundary_element_sets(basis) if use_curve_geometry else None  #1 precompute once for curved domain

    for elem in basis.elements():
        basis.localizeElement(elem)
        local_IEN = basis.HDIV.connectivity(elem)

        for bdry, normal_global_dofs in bdry_map.items():
            normal_global_set = set(normal_global_dofs)
            xi_vals = gq_bc.GetFaceQuadraturePoints(quad_1D, bdry)
            if not cf._is_boundary_face(basis, elem, bdry, quad_1D, bounds, boundary_sets):
                continue

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
                # u_exact_val = np.array(boundary_value_function(quad_pts[0], quad_pts[1])) 
                # if use_curve_geometry:  #CS curved domain: boundary condition defined in parametric space
                #     u_exact_val = np.array(boundary_value_function(quad_pts[0], quad_pts[1]))  
                # else:  
                #     u_exact_val = np.array(boundary_value_function(x_g, y_g))
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
    
    return prescribed
