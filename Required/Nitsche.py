#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:21:18 2026

@author: raminpahnabi
"""
import sys
import os
sys.path.insert(0, '/Users/raminpahnabi/Documents/BYU/sweeps/build/src/api')
sys.path.append(os.path.join(os.getcwd(), '../HWs'))
sys.path.append(os.path.join(os.getcwd(), 'Required'))

import numpy as np
import Quadrature_Operations_Solutions_boundary as gq_bc
import CommonFuncs as cf

KINEMATIC_VISCOSITY = 1

# ===========================================================================
#   Nitsche functions modified to act ONLY on TANGENTIAL DOFs
# ===========================================================================
def LocalForceVector_Nitsche_IGA_2D(basis, deg, quad,quad_1D, gamma, elem, forcing_function,u_exact, boundary_value_function):
    kinematic_viscosity = KINEMATIC_VISCOSITY
    
    local_IEN_HDIV = basis.HDIV.connectivity(elem)
    n_local_hdiv = len(local_IEN_HDIV)
    local_IEN_L2 = basis.L2.connectivity(elem)
    n_local_L2 = len(local_IEN_L2)
    n_local_total = n_local_hdiv + n_local_L2

    fe = np.zeros(n_local_total)
    
    bdries = [gq_bc.BoundaryFace.BOTTOM,gq_bc.BoundaryFace.TOP,gq_bc.BoundaryFace.LEFT,gq_bc.BoundaryFace.RIGHT]

    for bdry in bdries:
        xi_vals = gq_bc.GetFaceQuadraturePoints(quad_1D, bdry)
        face_length = cf.compute_face_length(basis, xi_vals, quad_1D, bdry)
        for g in range(len(xi_vals)):
            quad_pts = xi_vals[g]
            basis.localizePoint(quad_pts)
            qpt_mapped_bdry = basis.mapping()
            jac = basis.jacobian()
            jac_1d = cf.JacobianOneD(jac, bdry)
            
            tangent = cf.DifferentialVector(jac, bdry)
            if bdry == gq_bc.BoundaryFace.TOP or bdry == gq_bc.BoundaryFace.LEFT:
                R = np.array(([0,-1],
                              [1,0]))
            elif bdry == gq_bc.BoundaryFace.BOTTOM or bdry == gq_bc.BoundaryFace.RIGHT:
                R = np.array(([0,1],
                              [-1,0]))
            
            boundary_normal = np.dot(R, tangent)
            normal_unit = boundary_normal / jac_1d
            
            tangent_unit = tangent / jac_1d
            
            transformed_basis_bc = basis.piolaTransformedHDIVBasis()
            transformed_basis_L2_bc = basis.piolaTransformedL2()

            quad_wt_1d = quad_1D.quad_wts[g]
            
            x_g, y_g = qpt_mapped_bdry[0], qpt_mapped_bdry[1]
            if boundary_value_function is not None:
                u_boundary = np.array(boundary_value_function(x_g, y_g))
            else:
                u_boundary = np.array([0.0, 0.0])  # Default: no-slip
                
            # u_val = u_boundary
            u_val = np.dot(u_boundary, tangent_unit) * tangent_unit
            
            # NEW TERM from LHS
            n1, n2 = normal_unit[0], normal_unit[1]
            
            gradient_basis_bc = basis.piolaTransformedHDIVFirstDerivatives()
            
            # gradient_basis structure: [du1/dx, du1/dy, du2/dx, du2/dy]
            # Transpose: [du1/dx, du2/dx, du1/dy, du2/dy] = [row[0], row[2], row[1], row[3]]
            gradient_basis_transpose_bc = np.array([[row[0], row[2], row[1], row[3]] for row in gradient_basis_bc])
            sym_gradients_bc = (gradient_basis_bc + gradient_basis_transpose_bc) / 2.0
            
            # sym_gradients_bc structure: [eps11, eps12, eps21, eps22] where:
            # eps11 = du1/dx, eps12 = eps21 = (du1/dy + du2/dx)/2, eps22 = du2/dy
            # To compute (2ν∇^s u)·n, we form the matrix-vector product:
            # [eps11  eps12] [n1]   [eps11*n1 + eps12*n2]
            # [eps21  eps22] [n2] = [eps21*n1 + eps22*n2]
            
            # Compute (2*kinematic*sym_grad).n for each basis function
            sym_grad_basis_dot_n = []
            for sym_grad in sym_gradients_bc:
                sym_grad_dot_n = 2 * kinematic_viscosity * np.array([
                    sym_grad[0] * n1 + sym_grad[1] * n2, 
                    sym_grad[2] * n1 + sym_grad[3] * n2   
                ])
                sym_grad_basis_dot_n.append(sym_grad_dot_n)
            
            quad_wt_1d = quad_1D.quad_wts[g]
            

            # velocity-velocity coupling terms
            for a in range(n_local_hdiv):
                phi_a = transformed_basis_bc[a]
                sym_grad_a_dot_n = sym_grad_basis_dot_n[a]
                                
                    
                consistency_term = -np.dot(sym_grad_a_dot_n, u_val)
                penalty_term = (gamma * kinematic_viscosity / face_length) * np.dot(phi_a, u_val)
                
                fe[a] += (consistency_term + penalty_term) * quad_wt_1d * jac_1d

    return fe



def LocalStiffnessMatrix_Nitsche_IGA_2D(basis, deg, quad, quad_1D, gamma, elem, boundary_value_function=None):
    kinematic_viscosity = KINEMATIC_VISCOSITY
    
    local_IEN_HDIV = basis.HDIV.connectivity(elem)
    n_local_hdiv = len(local_IEN_HDIV)
    local_IEN_L2 = basis.L2.connectivity(elem)
    n_local_L2 = len(local_IEN_L2)
    n_local_total = n_local_hdiv + n_local_L2

    ke = np.zeros((n_local_total, n_local_total))
    
    bdries = [gq_bc.BoundaryFace.BOTTOM, gq_bc.BoundaryFace.TOP, 
              gq_bc.BoundaryFace.LEFT, gq_bc.BoundaryFace.RIGHT]

    for bdry in bdries:
        xi_vals = gq_bc.GetFaceQuadraturePoints(quad_1D, bdry)
        face_length = cf.compute_face_length(basis, xi_vals, quad_1D, bdry)
        
        for g in range(len(xi_vals)):
            quad_pts = xi_vals[g]
            basis.localizePoint(quad_pts)

            jac = basis.jacobian()
            jac_1d = cf.JacobianOneD(jac, bdry)
            
            # compute boundary normal
            tangent = cf.DifferentialVector(jac, bdry)
            tangent_unit = tangent / (jac_1d)
            if bdry == gq_bc.BoundaryFace.TOP or bdry == gq_bc.BoundaryFace.LEFT:
                R = np.array(([0,-1], [1,0]))
            elif bdry == gq_bc.BoundaryFace.BOTTOM or bdry == gq_bc.BoundaryFace.RIGHT:
                R = np.array(([0,1], [-1,0]))
            
            boundary_normal = np.dot(R, tangent)
            normal_unit = boundary_normal / (jac_1d)
            n1, n2 = normal_unit[0], normal_unit[1]
            
            transformed_basis_bc = basis.piolaTransformedHDIVBasis()
            transformed_basis_L2_bc = basis.piolaTransformedL2()
            gradient_basis_bc = basis.piolaTransformedHDIVFirstDerivatives()
            
            # gradient_basis structure: [du1/dx, du1/dy, du2/dx, du2/dy]
            # Transpose: [du1/dx, du2/dx, du1/dy, du2/dy] = [row[0], row[2], row[1], row[3]]
            gradient_basis_transpose_bc = np.array([[row[0], row[2], row[1], row[3]] for row in gradient_basis_bc])
            sym_gradients_bc = (gradient_basis_bc + gradient_basis_transpose_bc) / 2.0
            
            # sym_gradients_bc structure: [eps11, eps12, eps21, eps22] where:
            # eps11 = du1/dx, eps12 = eps21 = (du1/dy + du2/dx)/2, eps22 = du2/dy
            # To compute (2ν∇^s u)·n, we form the matrix-vector product:
            # [eps11  eps12] [n1]   [eps11*n1 + eps12*n2]
            # [eps21  eps22] [n2] = [eps21*n1 + eps22*n2]
            
            # Compute (2*kinematic*sym_grad).n for each basis function
            sym_grad_basis_dot_n = []
            for sym_grad in sym_gradients_bc:
                sym_grad_dot_n = 2 * kinematic_viscosity * np.array([
                    sym_grad[0] * n1 + sym_grad[1] * n2, 
                    sym_grad[2] * n1 + sym_grad[3] * n2   
                ])
                sym_grad_basis_dot_n.append(sym_grad_dot_n)
            
            quad_wt_1d = quad_1D.quad_wts[g]
            

            # velocity-velocity coupling terms
            for a in range(n_local_hdiv):
                phi_a = transformed_basis_bc[a]
                sym_grad_a_dot_n = sym_grad_basis_dot_n[a]
                
                for b in range(n_local_hdiv):
                    phi_b = transformed_basis_bc[b]
                    sym_grad_b_dot_n = sym_grad_basis_dot_n[b]
                    
                    consistency_term = -np.dot(sym_grad_a_dot_n, phi_b)
                    
                    symmetry_term = -np.dot(sym_grad_b_dot_n, phi_a)
                    
                    penalty_term = (gamma * 2 * kinematic_viscosity / face_length) * np.dot(phi_a, phi_b)
                    
                    # penalty_term = (gamma * kinematic_viscosity / face_length) * np.dot(phi_a, tangent_unit) * np.dot(phi_b, tangent_unit)
                    # consistency_term = -np.dot(sym_grad_a_dot_n, tangent_unit) * np.dot(phi_b, tangent_unit)
                    # symmetry_term    = -np.dot(sym_grad_b_dot_n, tangent_unit) * np.dot(phi_a, tangent_unit)
                    
                    ke[a, b] += (consistency_term + symmetry_term + penalty_term) * quad_wt_1d * jac_1d


            # for a in range(n_local_hdiv):
            #     phi_a_dot_n = np.dot(transformed_basis_bc[a], normal_unit)
            #     for b in range(n_local_L2):
            #         phi_p = transformed_basis_L2_bc[b]
            #         ke[a, b + n_local_hdiv] += phi_p * phi_a_dot_n * quad_wt_1d * jac_1d 
            #         # ke[b + n_local_hdiv, a] += -phi_p * phi_a_dot_n * quad_wt_1d * jac_1d  #TODO: should I cinsider this sym section? Symmetric part with negative sign

    return ke
