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

def LocalAdvectionPicard(basis, deg, quad, quad_1D, elem, previous_d_coeffs, boundary_condition):
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
        transformed_basis = basis.piolaTransformedHDIVBasis()
        gradient_basis = basis.piolaTransformedHDIVFirstDerivatives()

        uh_x_prev = sum(previous_d_coeffs[local_IEN_HDIV[i]] * transformed_basis[i][0] for i in range(len(transformed_basis)))
        uh_y_prev = sum(previous_d_coeffs[local_IEN_HDIV[i]] * transformed_basis[i][1] for i in range(len(transformed_basis)))
        u = [uh_x_prev,uh_y_prev]

        for a in range(n_local_hdiv):
            phi_a = transformed_basis[a]  # Current transformed basis function
        
            for b in range(n_local_hdiv):
                grad_v = np.reshape(gradient_basis[b], (2, 2))  # Gradient of velocity basis function
        
                convective_term = 0
                for i in range(2):
                    for j in range(2):
                        convective_term += u[i] * grad_v[i, j] * phi_a[j]
                ke[a, b] += convective_term * jac_det * weight * quad_jacobian
    return ke


def LocalAdvectionNewton(basis, deg, quad, quad_1D, elem, previous_d_coeffs, boundary_condition):  
    """                                                                                            
    # NNS Newton tangent stiffness for the advection term (u·∇)u.                                 
    # Full linearisation at u^k:                                                                   
    #   K^Newton_ab = ∫ [(u^k·∇)φ_b · φ_a  +  (φ_b·∇)u^k · φ_a] dΩ                            
    #              =  Picard term            +  extra Newton term                                  
    # The extra term requires grad(u^k) at each quadrature point.                                  
    """                                                                                            
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
        transformed_basis = basis.piolaTransformedHDIVBasis()
        gradient_basis = basis.piolaTransformedHDIVFirstDerivatives()

        # Previous velocity at this quadrature point (same as Picard)
        uh_x_prev = sum(previous_d_coeffs[local_IEN_HDIV[i]] * transformed_basis[i][0] for i in range(n_local_hdiv))
        uh_y_prev = sum(previous_d_coeffs[local_IEN_HDIV[i]] * transformed_basis[i][1] for i in range(n_local_hdiv))
        u_k = [uh_x_prev, uh_y_prev]

        # gradient of previous velocity — grad_u_k[i,j] = ∂u^k_{i+1}/∂x_{j+1}
        # gradient_basis layout per basis fn: [∂φ_1/∂x, ∂φ_1/∂y, ∂φ_2/∂x, ∂φ_2/∂y]
        grad_u_k = np.zeros((2, 2))                                                              
        for c in range(n_local_hdiv):                                                            
            d_c = previous_d_coeffs[local_IEN_HDIV[c]]                                         
            grad_u_k[0, 0] += d_c * gradient_basis[c][0]   # ∂u_1/∂x                          
            grad_u_k[0, 1] += d_c * gradient_basis[c][1]   # ∂u_1/∂y                          
            grad_u_k[1, 0] += d_c * gradient_basis[c][2]   # ∂u_2/∂x                          
            grad_u_k[1, 1] += d_c * gradient_basis[c][3]   # ∂u_2/∂y                          

        scale = jac_det * weight * quad_jacobian

        for a in range(n_local_hdiv):
            phi_a = transformed_basis[a]

            for b in range(n_local_hdiv):
                grad_v = np.reshape(gradient_basis[b], (2, 2))
                phi_b  = transformed_basis[b]                                                    

                # Picard term: (u^k·∇)φ_b · φ_a — identical to LocalAdvectionPicard
                picard_term = 0.0
                newton_extra2 = 0.0
                for i in range(2):
                    for j in range(2):
                        picard_term   += u_k[i] * grad_v[i, j] * phi_a[j]
                        newton_extra2 += u_k[i] * grad_u_k[i, j] * phi_a[j]  

                # extra Newton term: (φ_b·∇)u^k · φ_a = phi_a · (grad_u_k @ phi_b)
                newton_extra  = float(np.dot(phi_a, grad_u_k @ phi_b)) 
                                        

                ke[a, b] += (picard_term + newton_extra - newton_extra2) * scale                                 

    return ke


def LocalForceNS_Newton(basis, deg, quad, quad_1D, elem, previous_d_coeffs):  # NNS
    """                                                                         # NNS
    # NNS Newton nonlinear residual force vector.                               # NNS
    #   F^NL_a = ∫ (u^k·∇)u^k · φ_a dΩ                                       # NNS
    #                                                                           # NNS
    # PURPOSE: balances the extra Newton stiffness term so the fixed point      # NNS
    # of the iteration satisfies the full NS equations.  Add this to the       # NNS
    # external force in the solver:                                             # NNS
    #   (K_Stokes + K_Newton) u^{k+1} = F_ext + F^NL(u^k)                    # NNS
    """                                                                         # NNS
    local_IEN_HDIV = basis.HDIV.connectivity(elem)                              # NNS
    n_local_hdiv   = len(local_IEN_HDIV)                                        # NNS
    n_local_L2     = len(basis.L2.connectivity(elem))                           # NNS
    n_local_total  = n_local_hdiv + n_local_L2                                  # NNS

    fe = np.zeros(n_local_total)                                                # NNS

    for iqpt in range(len(quad.quad_wts)):                                      # NNS
        weight       = quad.quad_wts[iqpt]                                      # NNS
        qpt          = quad.quad_pts[iqpt]                                      # NNS
        quad_jacobian = quad.jacobian                                            # NNS

        basis.localizePoint(qpt)                                                # NNS
        jac_det          = basis.jacobianDeterminant()                          # NNS
        transformed_basis = basis.piolaTransformedHDIVBasis()                   # NNS
        gradient_basis    = basis.piolaTransformedHDIVFirstDerivatives()        # NNS

        # Previous velocity at this quadrature point                            # NNS
        uh_x = sum(previous_d_coeffs[local_IEN_HDIV[i]] * transformed_basis[i][0] for i in range(n_local_hdiv))  # NNS
        uh_y = sum(previous_d_coeffs[local_IEN_HDIV[i]] * transformed_basis[i][1] for i in range(n_local_hdiv))  # NNS
        u_k  = np.array([uh_x, uh_y])                                          # NNS

        # Gradient of previous velocity                                         # NNS
        grad_u_k = np.zeros((2, 2))                                             # NNS
        for c in range(n_local_hdiv):                                           # NNS
            d_c = previous_d_coeffs[local_IEN_HDIV[c]]                         # NNS
            grad_u_k[0, 0] += d_c * gradient_basis[c][0]                       # NNS
            grad_u_k[0, 1] += d_c * gradient_basis[c][1]                       # NNS
            grad_u_k[1, 0] += d_c * gradient_basis[c][2]                       # NNS
            grad_u_k[1, 1] += d_c * gradient_basis[c][3]                       # NNS

        # (u^k·∇)u^k = grad_u_k @ u_k                                         # NNS
        adv_uk = grad_u_k @ u_k                                                 # NNS

        scale = jac_det * weight * quad_jacobian                                # NNS
        for a in range(n_local_hdiv):                                           # NNS
            fe[a] += float(np.dot(transformed_basis[a], adv_uk)) * scale       # NNS

    return fe                                                                    # NNS


def LocalStiffnessL2Projection(basis, deg, quad, quad_1D, elem):
    """
    # Local stiffness for L2-projection: u + grad(p) = f, div(u) = 0.
    # u-u block : mass matrix  ∫ φ_a · φ_b dΩ   (NOT 2ν ε:ε — no viscous term in LP)
    # u-p block : -∫ (∇·φ_a) ψ_b dΩ             (same divergence coupling as Stokes)
    # p-u block : skew-symmetric transpose of u-p block
    """
    local_IEN_HDIV = basis.HDIV.connectivity(elem)
    n_local_hdiv = len(local_IEN_HDIV)
    local_IEN_L2 = basis.L2.connectivity(elem)
    n_local_L2 = len(local_IEN_L2)
    n_local_total = n_local_hdiv + n_local_L2

    ke = np.zeros((n_local_total, n_local_total))

    for g in range(len(quad.quad_wts)):
        quad_wts      = quad.quad_wts[g]
        quad_pts      = quad.quad_pts[g]
        quad_jacobian = quad.jacobian
        basis.localizePoint(quad_pts)
        jac_det = basis.jacobianDeterminant()

        transformed_basis    = basis.piolaTransformedHDIVBasis()            # (n_local_hdiv, 2) vector
        transformed_basis_L2 = basis.piolaTransformedL2()                   # (n_local_L2,) scalar
        gradient_basis       = basis.piolaTransformedHDIVFirstDerivatives() # (n_local_hdiv, 4): [v,xx v,xy v,yx v,yy]

        # u-u block: L2 mass matrix — identity operator on velocity, NO viscosity
        for a in range(n_local_hdiv):
            for b in range(n_local_hdiv):
                ke[a, b] += np.dot(transformed_basis[a], transformed_basis[b]) * jac_det * quad_wts * quad_jacobian

        # u-p block: -∫ (∇·φ_a) ψ_b dΩ  — same divergence coupling as Stokes
        for a in range(n_local_hdiv):
            div_phi_a = gradient_basis[a][0] + gradient_basis[a][3]  # div(φ_a) = v,xx + v,yy = trace(∇φ)
            for b in range(n_local_L2):
                ke[a, b + n_local_hdiv] -= div_phi_a * transformed_basis_L2[b] * jac_det * quad_wts * quad_jacobian

        # p-u block: skew-symmetric transpose of u-p block
        for a in range(n_local_L2):
            for b in range(n_local_hdiv):
                div_phi_b = gradient_basis[b][0] + gradient_basis[b][3]
                ke[a + n_local_hdiv, b] += div_phi_b * transformed_basis_L2[a] * jac_det * quad_wts * quad_jacobian

    return ke

def LocalForceStokesL2Projection(basis, deg, quad, quad_1D, gamma, elem, forcing_function, nu, use_curve_geometry):  
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
        
        qpt_mapped = basis.mapping()
        x_g, y_g = qpt_mapped[0], qpt_mapped[1]
        force = np.array(forcing_function(x_g, y_g, nu))  

        for a in range(n_local_hdiv):
            fe[a] += np.dot(transformed_basis[a], force) * jac_det * quad_wts * quad_jacobian
    
    return fe