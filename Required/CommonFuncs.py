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

KINEMATIC_VISCOSITY = 1


def GetSplineDegree(basis):
    degs = []
    for knotvec in basis.knotVectors():
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
        
        length += quad_wt_1d * jac_1d
        
    return length

def ID_array(HDiv, L2, boundary_dofs, prescribed):
    total_basis_functions_HDIV = HDiv.numTotalFunctions()
    total_basis_functions_L2 = L2.numTotalFunctions()
    total_basis = total_basis_functions_HDIV + total_basis_functions_L2

    ID = np.zeros(total_basis, dtype=int)

    counter = 0
    
    all_normal = boundary_dofs['all_normal']
    
    for i in range(total_basis_functions_HDIV):
        if i in all_normal:
            ID[i] = -1 
        else:
            ID[i] = counter
            counter += 1

    # Assign indices to L2 space functions
    # Mark first pressure DOF as fixed to avoid singularity
    ID[total_basis_functions_HDIV] = -1  # First pressure DOF
    
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