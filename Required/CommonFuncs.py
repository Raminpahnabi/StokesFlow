#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:35:00 2026

@author: raminpahnabi
"""
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'Required'))

import Quadrature_Operations_Solutions_boundary as gq_bc

# from scipy.sparse import lil_matrix
# from scipy.sparse.linalg import spsolve
from pathlib import Path

os.environ["SWEEPS_API_PATH"] = "/Users/raminpahnabi/Documents/BYU/sweeps/build/src/api"
import splines as spline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from sweeps_path import ensure_sweeps_api_on_path

ensure_sweeps_api_on_path()
sys.path.append(str(PROJECT_ROOT / 'HWs'))
sys.path.append(str(PROJECT_ROOT / 'Required'))




def _boundary_element_sets(basis):  #CS
    """Topological boundary detection — correct for both straight-sided and curved domains.
    Returns {BoundaryFace: set_of_element_dart_ids}. Call once per solve (O(n_elems))."""  
    bdry_elems = find_boundary_elements(basis)  
    return {  
        gq_bc.BoundaryFace.BOTTOM: bdry_elems['bottom'][1],  
        gq_bc.BoundaryFace.TOP:    bdry_elems['top'][1],  
        gq_bc.BoundaryFace.LEFT:   bdry_elems['left'][1],  
        gq_bc.BoundaryFace.RIGHT:  bdry_elems['right'][1],  
    }  

def _physical_domain_bounds(basis):
    control_points = np.asarray(basis.control_points)
    x_min = float(np.min(control_points[0]))
    x_max = float(np.max(control_points[0]))
    y_min = float(np.min(control_points[1]))
    y_max = float(np.max(control_points[1]))
    return x_min, x_max, y_min, y_max


def _is_boundary_face(basis, elem, bdry, quad_1D, bounds, boundary_sets=None):
    if boundary_sets is not None:  # curved domain: topological detection via DOF connectivity
        return elem.dart in boundary_sets[bdry]  # avoids physical bounding-box check that fails for curved domains
    xi_vals = gq_bc.GetFaceQuadraturePoints(quad_1D, bdry)
    face_midpoint = xi_vals[len(xi_vals) // 2]

    basis.localizeElement(elem)
    basis.localizePoint(face_midpoint)
    x_f, y_f = basis.mapping()[:2]

    x_min, x_max, y_min, y_max = bounds
    span = max(x_max - x_min, y_max - y_min, 1.0)
    tol = 1e-10 * span

    if bdry == gq_bc.BoundaryFace.BOTTOM:
        return abs(y_f - y_min) <= tol
    if bdry == gq_bc.BoundaryFace.TOP:
        return abs(y_f - y_max) <= tol
    if bdry == gq_bc.BoundaryFace.LEFT:
        return abs(x_f - x_min) <= tol
    if bdry == gq_bc.BoundaryFace.RIGHT:
        return abs(x_f - x_max) <= tol
    return False


def GetSplineDegree(basis):
    try: 
        kv_iter = basis.knotVectors()
    except AttributeError:  
        return []  # hierarchical basis: no global knot vectors
    
    degs = []
    for knotvec in kv_iter:
        knot_list = str(knotvec).replace("}","").replace("{","").split(",")
        count = 0
        first_knot = float(knot_list[0])
        for i in range(1, len(knot_list)):
            if float(knot_list[i]) == first_knot:
                count += 1
            else:
                break
        degs.append(count)
    return degs

def GetNumberH1FirstComponent(basis):
    try:  # LR bases have no knotVectors()
        basis.knotVectors()
    except AttributeError:  # hierarchical basis: return total HDIV count as comp1, 0 as comp2
        n_hdiv = basis.HDIV.numTotalFunctions()  
        return n_hdiv, 0  

    degs = GetSplineDegree(basis)
    knot_vecs = basis.knotVectors()
    num_h1_bfs = []
    for i in range(0, len(degs)):
        num_h1_bfs.append(
            len(str(knot_vecs[i]).replace("}","").replace("{","").split(",")[:-1]) - (degs[i] + 1)
        )
    # NOTE: from connectivity we know:
    #   comp1 (DOFs 0..nc1*nr1-1) is the 4×3 (/\ normal on bottom/top) component
    #   comp2 (DOFs nc1*nr1..)    is the 3×4 (>  normal on left/right) component
    # GetNumberH1FirstComponent returns (comp1_count, comp2_count)
    num_hdiv_comp1 = (num_h1_bfs[0]) * (num_h1_bfs[1] - 1)   # n_bf_x × (n_bf_y-1) = 4×3
    num_hdiv_comp2 = (num_h1_bfs[0] - 1) * (num_h1_bfs[1])   # (n_bf_x-1) × n_bf_y = 3×4
    return num_hdiv_comp1, num_hdiv_comp2


# extract the column that matches the varying direction for further calculations
def DifferentialVector(def_grad, bdry_face):
    varying_coord = gq_bc.__BdryFaceToVaryingCoordinate__(bdry_face)
    
    return def_grad[:,varying_coord] 


# Compute the Jacobian of a curve given points on the face,
# control points that define the parent to spatial mapping,
# a basis function object, and the face of interest
def JacobianOneD(def_grad, bdry_face):
    diff_vect = DifferentialVector(def_grad, bdry_face)
    
    return np.sqrt(diff_vect[0]**2 + diff_vect[1]**2)


def compute_face_length(basis, xi_vals, quad_1D, bdry_face):
    length = 0.0
    for g in range(len(xi_vals)):
        xi = xi_vals[g]
        basis.localizePoint(xi)
        jac = basis.jacobian()    
        diff_vect = DifferentialVector(jac, bdry_face)   # shape (2,)
        jac_1d = np.linalg.norm(diff_vect)        # np.sqrt(dx/dxi**2 + dy/dxi**2)
        quad_wt_1d = quad_1D.quad_wts[g]
        
        length += quad_1D.jacobian * quad_wt_1d * jac_1d
        
    return length

def ID_array(HDiv, L2, boundary_dofs, free_faces=None):  # free_faces = outflow_faces (normal DOFs solved, not eliminated)
    total_basis_functions_HDIV = HDiv.numTotalFunctions()
    total_basis_functions_L2 = L2.numTotalFunctions()
    total_basis = total_basis_functions_HDIV + total_basis_functions_L2

    ID = np.zeros(total_basis, dtype=int)

    counter = 0

    #build constrained set: all normal DOFs minus those on free faces
    free = set(free_faces or [])
    constrained_normal = set(boundary_dofs['all_normal'])
    for face in free:
        constrained_normal -= set(boundary_dofs[face]['normal'])  #remove free-face DOFs so they are solved, not prescribed

    for i in range(total_basis_functions_HDIV):
        if i in constrained_normal:
            ID[i] = -1
        else:
            ID[i] = counter
            counter += 1


    # Mark first pressure DOF as fixed to avoid singularity
    ID[total_basis_functions_HDIV] = -1  
    
    for i in range(total_basis_functions_HDIV + 1, total_basis):
        ID[i] = counter
        counter += 1

    return ID

def ID_array_l2projection(HDiv, L2, boundary_dofs):
    total_basis_functions_HDIV = HDiv.numTotalFunctions()
    total_basis_functions_L2 = L2.numTotalFunctions()
    total_basis = total_basis_functions_HDIV + total_basis_functions_L2

    ID = np.zeros(total_basis, dtype=int)

    counter = 0
    
    for i in range(total_basis_functions_HDIV):
        ID[i] = counter
        counter += 1

    ID[total_basis_functions_HDIV] = -1
    
    for i in range(total_basis_functions_HDIV + 1, total_basis):
        ID[i] = counter
        counter += 1

    return ID

def ExtractTotalD(ID, d_reduced, prescribed, n_hdiv, n_l2):
    d_total = np.zeros(len(ID))
    
    for A in range(len(d_total)):
        if ID[A] == -1:
            if A in prescribed:
                d_total[A] = prescribed[A]
            elif A == n_hdiv:
                d_total[A] = 0.0
                
        else:
            d_total[A] = d_reduced[ID[A]]
            
    return d_total

def ExtractTotalD_l2projection(ID, d_reduced, prescribed, n_hdiv, n_l2):
    d_total = np.zeros(len(ID))
    
    for A in range(len(d_total)):
        if ID[A] == -1:
            d_total[A] = 0.0
                
        else:
            d_total[A] = d_reduced[ID[A]]
            
    return d_total

def find_normal_parallel_boundary_conditions(discretization):
    # Initialize boundary conditions storage
    boundary_conditions = {
        'left': {'hdiv_normal': [], 'hdiv_parallel': []},
        'right': {'hdiv_normal': [], 'hdiv_parallel': []},
        'bottom': {'hdiv_normal': [], 'hdiv_parallel': []},
        'top': {'hdiv_normal': [], 'hdiv_parallel': []}
    }

    # Define patch sides using the correct PatchSide enum values
    sides = {
        'left': spline.PatchSide.S0,
        'right': spline.PatchSide.S1,
        'bottom': spline.PatchSide.T0,
        'top': spline.PatchSide.T1
    }

    # Loop over each side and find perpendicular and parallel functions
    all_funcs = list(range(discretization.HDIV.numTotalFunctions()))  # Get total number of HDiv functions
    
    for side_name, patch_side in sides.items():
        try:
            # Find perpendicular HDiv functions (normal to the boundary) on the given side
            perpendicular_funcs = discretization.boundaryPerpendicularHDivFuncs(patch_side)
            #print(f"Perpendicular functions on {side_name}: {perpendicular_funcs}")
            boundary_conditions[side_name]['hdiv_normal'].extend(perpendicular_funcs)

            # # Identify the parallel functions (those not perpendicular) for this side
            # parallel_funcs = set(all_funcs) - set(perpendicular_funcs)
            # boundary_conditions[side_name]['hdiv_parallel'].extend(parallel_funcs)

        except Exception as e:
            print(f"Error processing side {side_name}: {e}")

    return boundary_conditions


    
def find_boundary_elements(basis):
    # Get normal boundary conditions using an existing function
    normal_boundary = find_normal_parallel_boundary_conditions(basis)
    
    # Dictionary to store indices on each boundary
    # store normal basis function indicies in the first list, elements in the 2nd

    # first index is global bf indices that are normal on this boundary, 
    # second index is elements that are on this boundary
    # third is the dirichlet boundary condition normal to this boundary
    # fourth is the dirichlet boundary condition (weakly enforced) tangential to this boundary condition 
    # fifth is the normal dirichlet boundary condition to this boundary condition being appended later
    boundary_indices = {
        'left': [set(),set(),[],[],[]], 
        'right': [set(),set(),[],[],[]],
        'top': [set(),set(),[],[],[]],
        'bottom': [set(),set(),[],[],[]]
    }

    # Loop over all elements to check their connectivity
    for elem in basis.elements():
        # Get connectivity for the current element in HDIV space
        local_IEN_HDIV = set(basis.HDIV.connectivity(elem))  # Connectivity for HDIV space

        # Check if the indices of the element are on any of the boundaries and collect them
        if local_IEN_HDIV & set(normal_boundary.get('left', {}).get('hdiv_normal', [])):
            boundary_indices['left'][0].update(local_IEN_HDIV & set(normal_boundary['left']['hdiv_normal']))
            boundary_indices['left'][1].add(elem.dart)
        if local_IEN_HDIV & set(normal_boundary.get('right', {}).get('hdiv_normal', [])):
            boundary_indices['right'][0].update(local_IEN_HDIV & set(normal_boundary['right']['hdiv_normal']))
            boundary_indices['right'][1].add(elem.dart)
        if local_IEN_HDIV & set(normal_boundary.get('top', {}).get('hdiv_normal', [])):
            boundary_indices['top'][0].update(local_IEN_HDIV & set(normal_boundary['top']['hdiv_normal']))
            boundary_indices['top'][1].add(elem.dart)
        if local_IEN_HDIV & set(normal_boundary.get('bottom', {}).get('hdiv_normal', [])):
            boundary_indices['bottom'][0].update(local_IEN_HDIV & set(normal_boundary['bottom']['hdiv_normal']))
            boundary_indices['bottom'][1].add(elem.dart)

    # Convert sets to sorted lists for easier interpretation
    for key in boundary_indices:
        boundary_indices[key][0] = sorted(boundary_indices[key][0])

    return boundary_indices

    