#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:17:49 2026

@author: raminpahnabi
"""

import numpy as np

def LocalForceStokes(basis, deg, quad, quad_1D, gamma, elem, forcing_function, nu):  
    local_IEN_HDIV = basis.HDIV.connectivity(elem)
    n_local_hdiv = len(local_IEN_HDIV)
    local_IEN_L2 = basis.L2.connectivity(elem)
    n_local_L2 = len(local_IEN_L2)
    n_local_total = n_local_hdiv + n_local_L2

    fe = np.zeros(n_local_total)
    
    for g in range(len(quad.quad_wts)):
        quad_wts = quad.quad_wts[g]
        quad_pts = quad.quad_pts[g]
        quad_jacobian = quad.jacobian
        basis.localizePoint(quad_pts)
        jac_det = basis.jacobianDeterminant()

        transformed_basis = basis.piolaTransformedHDIVBasis()
        
        # Mapping from reference space to global space
        qpt_mapped = basis.mapping()
        x_g, y_g = qpt_mapped[0], qpt_mapped[1]
        force = np.array(forcing_function(x_g, y_g, nu))  

        for a in range(n_local_hdiv):
            fe[a] += np.dot(transformed_basis[a], force) * jac_det * quad_wts * quad_jacobian
    
    return fe
    
    
def LocalStiffnessStokes(basis, deg, quad, quad_1D, elem, boundary_condition, parent_domain_fudge_factor=1, nu=1):  
    kinematic_viscosity = nu 

    local_IEN_HDIV = basis.HDIV.connectivity(elem)
    n_local_hdiv = len(local_IEN_HDIV)
    local_IEN_L2 = basis.L2.connectivity(elem)
    n_local_L2 = len(local_IEN_L2)
    n_local_total = n_local_hdiv + n_local_L2

    ke = np.zeros((n_local_total, n_local_total))

    for iqpt in range(len(quad.quad_wts)):
        weight = quad.quad_wts[iqpt]
        qpt = quad.quad_pts[iqpt]
        quad_jacobian = quad.jacobian
        basis.localizePoint(qpt)
        jac_det = basis.jacobianDeterminant()


        transformed_basis_L2 = basis.piolaTransformedL2()
        gradient_basis = basis.piolaTransformedHDIVFirstDerivatives()
       
        gradient_basis_transpose = gradient_basis[:, [0, 2, 1, 3]]
        sym_gradients = (gradient_basis + gradient_basis_transpose) / 2

        # Viscous contributions (vectorized)
        scale = 2 * kinematic_viscosity * jac_det * weight * quad_jacobian
        ke[:n_local_hdiv, :n_local_hdiv] += scale * (sym_gradients @ sym_gradients.T)

        # Pressure coupling (vectorized)
        trace_grad = gradient_basis[:, 0] + gradient_basis[:, 3]
        scale_p = jac_det * weight * quad_jacobian
        block = np.outer(trace_grad, transformed_basis_L2) * scale_p
        ke[:n_local_hdiv, n_local_hdiv:] -= block
        ke[n_local_hdiv:, :n_local_hdiv] = -ke[:n_local_hdiv, n_local_hdiv:].T
    
    return ke