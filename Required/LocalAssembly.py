#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:17:49 2026

@author: raminpahnabi
"""

import numpy as np

def LocalForceStokes(basis, deg, quad, quad_1D, gamma, elem, forcing_function, nu):  #CSn removed unused use_curve_geometry param
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
    
    
def LocalStiffnessStokes(basis, deg, quad, quad_1D, elem, boundary_condition, nu=1):  
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

#################### UPWIND #####################
def EvalLocalStiffnessStokes_boundary(basis, deg, quad, quad_1D, elem, boundary_condition, nu=1):
    local_IEN_HDIV = basis.HDIV.connectivity(elem)
    n_local_hdiv = len(local_IEN_HDIV)
    local_IEN_L2 = basis.L2.connectivity(elem)
    n_local_L2=len(local_IEN_L2)
    n_local_total = n_local_hdiv + n_local_L2

    ke = np.zeros((n_local_total, n_local_total))

    if boundary_condition is not None:
        for boundary, indices in boundary_condition.items():
                
                # Generate quadrature points for the boundary
                if boundary == 'left':
                    if elem.dart not in indices[1]:
                        continue
                    else:
                        quad_pts = [[0, val] for val in quad_1D.quad_pts]
                elif boundary == 'right':
                    if elem.dart not in indices[1]:
                        continue
                    else:
                        quad_pts = [[1, val] for val in quad_1D.quad_pts]
                elif boundary == 'top':
                    if elem.dart not in indices[1]:
                        continue
                    else:
                        quad_pts = [[val, 1] for val in quad_1D.quad_pts]
                elif boundary == 'bottom':
                    if elem.dart not in indices[1]:
                        continue
                    else:
                        quad_pts = [[val, 0] for val in quad_1D.quad_pts]
                
                
                jac_det_bdry = []
                temp_weight = []
                
                ubc = boundary_condition[boundary][4]
                ubc_global_indices = boundary_condition[boundary][0]
                ubc_mapped = {}
                for c in range(0,len(local_IEN_HDIV)):
                    if local_IEN_HDIV[c] in ubc_global_indices:
                        C=local_IEN_HDIV[c]
                        ubc_mapped[c] = C
                # ubc_mapped = {local_IEN_HDIV.index(A): ubc[i] for i, A in enumerate(ubc_global_indices) if A in local_IEN_HDIV}
                ubc_globalidx_to_sideidx = {}
                for i in range(0,len(ubc_global_indices)):
                    ubc_globalidx_to_sideidx[ubc_global_indices[i]] = i
                
                for i_1D, weight_1D in enumerate(quad_1D.quad_wts):
                    qpt_boundary = quad_pts[i_1D]  # Using the generated quadrature points for the boundary
                    basis.localizePoint(qpt_boundary)
                    deformation_gradients = basis.jacobian()
                    if boundary == 'left' or boundary == 'right':
                        deformation_gradient_boundary = deformation_gradients[:,1]
                    elif boundary == 'top' or boundary == 'bottom':
                        deformation_gradient_boundary = deformation_gradients[:,0]
                    
                    jac_det_boundary = np.sqrt(deformation_gradient_boundary[0]**2 + deformation_gradient_boundary[1]**2)
                    jac_det_bdry.append(jac_det_boundary)
                    temp_weight.append(weight_1D)
                
               
                    tangent_vector_boundary = deformation_gradient_boundary
                    
                    # Calculate the normal vector for the boundary
                    normal_vector_boundary = np.array([-tangent_vector_boundary[1], tangent_vector_boundary[0]])
                    magnitude_normal_boundary = np.linalg.norm(normal_vector_boundary)
                    normal_unit_vector_boundary = normal_vector_boundary / magnitude_normal_boundary if magnitude_normal_boundary != 0 else np.array([0, 0])
    
                    # Set direction scalar based on the boundary
                    dir_change_scalar = 1
                    if boundary in ['bottom', 'right']:
                        dir_change_scalar = -1
                    elif boundary in ['top', 'left']:
                          dir_change_scalar = 1
    
                    # Adjust the normal vector based on the boundary's unit vector
                    adjusted_normal = normal_unit_vector_boundary * dir_change_scalar
                    
                    transformed_basis = basis.piolaTransformedHDIVBasis()

                    u_bc = np.zeros(2)
                    for c in range(n_local_hdiv):
                        if c in ubc_mapped:
                            C = ubc_mapped[c]
                            side_idx = ubc_globalidx_to_sideidx[C]
                            coeff = ubc[side_idx]
                            u_bc += coeff * transformed_basis[c]
                                    
                    upwind_term = np.dot(u_bc,adjusted_normal)
                    
                    if upwind_term > 0:
                        for a in range(n_local_hdiv):                        
                            for b in range(n_local_hdiv):        
                                    # Add contributions to the stiffness matrix
                                    ke_boundary = (upwind_term *np.dot(transformed_basis[a],transformed_basis[b])
                                        
                                    ) * weight_1D * jac_det_boundary *quad_1D.jacobian
                                    
                                    if isinstance(ke_boundary, np.ndarray):
                                        ke_boundary = np.sum(ke_boundary)
                    
                                    ke[a, b] += ke_boundary                   
    return ke

##################### Force UPWIND #####################
def EvalLocalforceStokes_boundary(basis,deg, quad, quad_1D, elem, forcing_function, boundary_conditions=None, nu=1):
    local_IEN_HDIV = basis.HDIV.connectivity(elem)
    n_local_hdiv = len(local_IEN_HDIV)
    local_IEN_L2 = basis.L2.connectivity(elem)
    n_local_L2 = len(local_IEN_L2)
    n_local_total = n_local_hdiv + n_local_L2

    # Initialize local force vector
    fe = np.zeros(n_local_total)
    deformation_gradients = basis.jacobian()

    if boundary_conditions is not None:
        for boundary, indices in boundary_conditions.items():
            # Generate quadrature points for the boundary
            if boundary == 'left':
                if elem.dart not in indices[1]:
                    continue
                else:
                    quad_pts = [[0, val] for val in quad_1D.quad_pts]
            elif boundary == 'right':
                if elem.dart not in indices[1]:
                    continue
                else:
                    quad_pts = [[1, val] for val in quad_1D.quad_pts]
            elif boundary == 'top':
                if elem.dart not in indices[1]:
                    continue
                else:
                    quad_pts = [[val, 1] for val in quad_1D.quad_pts]
            elif boundary == 'bottom':
                if elem.dart not in indices[1]:
                    continue
                else:
                    quad_pts = [[val, 0] for val in quad_1D.quad_pts]
           
           
           
            # on this boundary type
            jac_det_bdry = []
            temp_weight = []
           
            for i_1D, weight_1D in enumerate(quad_1D.quad_wts):
                qpt_boundary = quad_pts[i_1D]  # Using the generated quadrature points for the boundary
                basis.localizePoint(qpt_boundary)
                deformation_gradients = basis.jacobian()
                if boundary == 'left' or boundary == 'right':
                    deformation_gradient_boundary = deformation_gradients[:,1]
                elif boundary == 'top' or boundary == 'bottom':
                    deformation_gradient_boundary = deformation_gradients[:,0]
                    
                ubc = boundary_conditions[boundary][4]
                ubc_global_indices = boundary_conditions[boundary][0]
                ubc_mapped = {}
                for c in range(0,len(local_IEN_HDIV)):
                    if local_IEN_HDIV[c] in ubc_global_indices:
                        C=local_IEN_HDIV[c]
                        ubc_mapped[c] = C
                ubc_globalidx_to_sideidx = {}
                for i in range(0,len(ubc_global_indices)):
                    ubc_globalidx_to_sideidx[ubc_global_indices[i]] = i
               
                jac_det_boundary = np.sqrt(deformation_gradient_boundary[0]**2 + deformation_gradient_boundary[1]**2)
                jac_det_bdry.append(jac_det_boundary)
                temp_weight.append(weight_1D)

                # Calculate the tangent vector
                tangent_vector_boundary = deformation_gradient_boundary
                magnitude_tangent_boundary = np.linalg.norm(tangent_vector_boundary)
                tangent_unit_vector_boundary = tangent_vector_boundary / magnitude_tangent_boundary if magnitude_tangent_boundary != 0 else np.array([0, 0])
    
                # Calculate the normal vector for the boundary
                normal_vector_boundary = np.array([-tangent_vector_boundary[1], tangent_vector_boundary[0]])
                magnitude_normal_boundary = np.linalg.norm(normal_vector_boundary)
                normal_unit_vector_boundary = normal_vector_boundary / magnitude_normal_boundary if magnitude_normal_boundary != 0 else np.array([0, 0])
    
                # Set direction scalar based on the boundary
                dir_change_scalar = 1
                if boundary in ['bottom', 'right']:
                    dir_change_scalar = -1
    
                # Adjust the normal vector based on the boundary's unit vector
                adjusted_normal = normal_unit_vector_boundary * dir_change_scalar
               
                transformed_basis = basis.piolaTransformedHDIVBasis()

                qpt_mapped = basis.mapping()
                x_g, y_g = qpt_mapped[0], qpt_mapped[1]

                u_bc_normal = np.zeros(2)
                for c in range(n_local_hdiv):
                    if c in ubc_mapped:
                        C = ubc_mapped[c]
                        side_idx = ubc_globalidx_to_sideidx[C]
                        coeff = ubc[side_idx]
                        u_bc_normal += coeff * transformed_basis[c]
                                
                        
                upwind_term = np.dot(u_bc_normal,adjusted_normal)

                given_boundary= boundary_conditions[boundary][3](x_g,y_g)*tangent_unit_vector_boundary +u_bc_normal
              
                if upwind_term<=0:
          
                    for a in range(n_local_hdiv):                        
                        fe_boundary= (
                            upwind_term* np.dot(given_boundary,transformed_basis[a])
                        ) * weight_1D * quad_1D.jacobian*jac_det_boundary
                        
                        if isinstance(fe_boundary, np.ndarray):
                            fe_boundary = np.sum(fe_boundary)
                        fe[a] -= fe_boundary
                       
                else:
                    continue
    
    return fe

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
                for i in range(2):
                    for j in range(2):
                        picard_term   += u_k[i] * grad_v[i, j] * phi_a[j]  

                # extra Newton term: (φ_b·∇)u^k · φ_a = phi_a · (grad_u_k @ phi_b)
                newton_extra  = float(np.dot(phi_a, grad_u_k @ phi_b)) 
                                        

                ke[a, b] += (picard_term + newton_extra) * scale                                 

    return ke


def LocalForceNS_Newton(basis, deg, quad, quad_1D, elem, previous_d_coeffs):  
    """                                                                         
    # Newton nonlinear residual force vector.
    #   F^NL_a = ∫ (u^k·∇)u^k · φ_a dΩ                                       
    #                                                                           
    # PURPOSE: balances the extra Newton stiffness term so the fixed point      
    # of the iteration satisfies the full NS equations.  Add this to the       
    # external force in the solver:                                             
    #   (K_Stokes + K_Newton) u^{k+1} = F_ext + F^NL(u^k)                   
    """                                                                         
    local_IEN_HDIV = basis.HDIV.connectivity(elem)                              
    n_local_hdiv   = len(local_IEN_HDIV)                                        
    n_local_L2     = len(basis.L2.connectivity(elem))                           
    n_local_total  = n_local_hdiv + n_local_L2                                  

    fe = np.zeros(n_local_total)                                                

    for iqpt in range(len(quad.quad_wts)):                                      
        weight        = quad.quad_wts[iqpt]                                      
        qpt           = quad.quad_pts[iqpt]                                      
        quad_jacobian = quad.jacobian                                            

        basis.localizePoint(qpt)                                                
        jac_det           = basis.jacobianDeterminant()                         
        transformed_basis = basis.piolaTransformedHDIVBasis()                   
        gradient_basis    = basis.piolaTransformedHDIVFirstDerivatives()        

        # Previous velocity at this quadrature point                            
        uh_x = sum(previous_d_coeffs[local_IEN_HDIV[i]] * transformed_basis[i][0] for i in range(n_local_hdiv)) 
        uh_y = sum(previous_d_coeffs[local_IEN_HDIV[i]] * transformed_basis[i][1] for i in range(n_local_hdiv))  
        u_k  = np.array([uh_x, uh_y])                                          

        # Gradient of previous velocity                                         
        grad_u_k = np.zeros((2, 2))                                             
        for c in range(n_local_hdiv):                                           
            d_c = previous_d_coeffs[local_IEN_HDIV[c]]                        
            grad_u_k[0, 0] += d_c * gradient_basis[c][0]                    
            grad_u_k[0, 1] += d_c * gradient_basis[c][1]                     
            grad_u_k[1, 0] += d_c * gradient_basis[c][2]                       
            grad_u_k[1, 1] += d_c * gradient_basis[c][3]                       

        # (u^k·∇)u^k = grad_u_k @ u_k                                         
        adv_uk = grad_u_k @ u_k                                             

        scale = jac_det * weight * quad_jacobian                              
        for a in range(n_local_hdiv):                                          
            fe[a] += float(np.dot(transformed_basis[a], adv_uk)) * scale       

    return fe                                                                    


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
        # if use_curve_geometry:  #CS curved domain: forcing function defined in parametric space
        #     force = np.array(forcing_function(quad_pts[0], quad_pts[1], nu))  
        # else:  
        #     force = np.array(forcing_function(x_g, y_g, nu))

        for a in range(n_local_hdiv):
            fe[a] += np.dot(transformed_basis[a], force) * jac_det * quad_wts * quad_jacobian
    
    return fe