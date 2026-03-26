#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:21:18 2026

@author: raminpahnabi
"""
import sys
import os
import numpy as np
sys.path.insert(0, '/Users/raminpahnabi/Documents/BYU/sweeps/build/src/api')
sys.path.append(os.path.join(os.getcwd(), '../HWs'))
sys.path.append(os.path.join(os.getcwd(), 'Required'))

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

            gradient_basis_transpose_bc = gradient_basis_bc[:, [0, 2, 1, 3]]
            sym_gradients_bc = (gradient_basis_bc + gradient_basis_transpose_bc) / 2.0

            # sigma_n[a] = 2ν * ε(φ_a) · n, shape (n_local_hdiv, 2)
            sg = sym_gradients_bc.reshape(-1, 2, 2)
            n_vec = np.array([n1, n2])
            sigma_n = 2 * kinematic_viscosity * (sg @ n_vec)

            phi = transformed_basis_bc  # (n_local_hdiv, 2)
            quad_wt_1d = quad_1D.quad_wts[g]
            scale = quad_wt_1d * jac_1d

            fe[:n_local_hdiv] += scale * (
                -(sigma_n @ u_val)
                + (gamma * kinematic_viscosity / face_length) * (phi @ u_val)
            )

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
            
            gradient_basis_transpose_bc = gradient_basis_bc[:, [0, 2, 1, 3]]
            sym_gradients_bc = (gradient_basis_bc + gradient_basis_transpose_bc) / 2.0

            # sigma_n[a] = 2ν * ε(φ_a) · n, shape (n_local_hdiv, 2)
            sg = sym_gradients_bc.reshape(-1, 2, 2)
            n_vec = np.array([n1, n2])
            sigma_n = 2 * kinematic_viscosity * (sg @ n_vec)

            phi = transformed_basis_bc  # (n_local_hdiv, 2)
            quad_wt_1d = quad_1D.quad_wts[g]
            scale = quad_wt_1d * jac_1d

            # velocity-velocity coupling (vectorized)
            ke[:n_local_hdiv, :n_local_hdiv] += scale * (
                -(sigma_n @ phi.T)                                                # consistency
                -(phi @ sigma_n.T)                                                # symmetry
                + (gamma * kinematic_viscosity / face_length) * (phi @ phi.T)    # penalty
            )

            # pressure-velocity coupling (vectorized)
            phi_dot_n = phi @ normal_unit  # (n_local_hdiv,)
            ke[:n_local_hdiv, n_local_hdiv:] += scale * np.outer(phi_dot_n, transformed_basis_L2_bc)

    return ke
