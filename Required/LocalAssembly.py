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

        scale = jac_det * quad_wts * quad_jacobian
        fe[:n_local_hdiv] += (transformed_basis @ force) * scale

    return fe


def LocalStiffnessStokes(basis, deg, quad, quad_1D, elem, boundary_condition, nu=1):
    kinematic_viscosity = nu

    local_IEN_HDIV = basis.HDIV.connectivity(elem)
    n_local_hdiv = len(local_IEN_HDIV)
    local_IEN_L2 = basis.L2.connectivity(elem)
    n_local_L2 = len(local_IEN_L2)
    n_local_total = n_local_hdiv + n_local_L2

    ke = np.zeros((n_local_total, n_local_total))

    for g in range(len(quad.quad_wts)):
        quad_wts = quad.quad_wts[g]
        quad_pts = quad.quad_pts[g]
        quad_jacobian = quad.jacobian
        basis.localizePoint(quad_pts)
        jac_det = basis.jacobianDeterminant()

        transformed_basis_L2 = basis.piolaTransformedL2()
        gradient_basis = basis.piolaTransformedHDIVFirstDerivatives()

        gradient_basis_transpose = gradient_basis[:, [0, 2, 1, 3]]
        sym_gradients = (gradient_basis + gradient_basis_transpose) / 2

        # Viscous u-u block
        scale = 2 * kinematic_viscosity * jac_det * quad_wts * quad_jacobian
        ke[:n_local_hdiv, :n_local_hdiv] += scale * (sym_gradients @ sym_gradients.T)

        # Pressure coupling u-p and p-u blocks
        div_phi = gradient_basis[:, 0] + gradient_basis[:, 3]
        scale_p = jac_det * quad_wts * quad_jacobian
        block = np.outer(div_phi, transformed_basis_L2) * scale_p
        ke[:n_local_hdiv, n_local_hdiv:] -= block
        ke[n_local_hdiv:, :n_local_hdiv] += block.T

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
                        scale_up = upwind_term * weight_1D * jac_det_boundary * quad_1D.jacobian
                        ke[:n_local_hdiv, :n_local_hdiv] += scale_up * (transformed_basis @ transformed_basis.T)

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

                if upwind_term <= 0:
                    scale_up = upwind_term * weight_1D * quad_1D.jacobian * jac_det_boundary
                    fe[:n_local_hdiv] -= scale_up * (transformed_basis @ given_boundary)
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

        d_local = np.array([previous_d_coeffs[local_IEN_HDIV[i]] for i in range(n_local_hdiv)])
        u_k = d_local @ transformed_basis  # shape (2,): previous velocity at this qpt

        # ke[a,b] = sum_ij u_k[i] * G[b,i,j] * phi[a,j]  where G = gradient_basis reshaped
        G = gradient_basis.reshape(n_local_hdiv, 2, 2)
        scale = jac_det * weight * quad_jacobian
        ke[:n_local_hdiv, :n_local_hdiv] += np.einsum('i,bij,aj->ab', u_k, G, transformed_basis) * scale

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

        d_local = np.array([previous_d_coeffs[local_IEN_HDIV[i]] for i in range(n_local_hdiv)])
        u_k = d_local @ transformed_basis  # shape (2,): previous velocity at this qpt

        # gradient of previous velocity — grad_u_k[i,j] = ∂u^k_{i+1}/∂x_{j+1}
        # gradient_basis layout per basis fn: [∂φ_1/∂x, ∂φ_1/∂y, ∂φ_2/∂x, ∂φ_2/∂y]
        grad_u_k = (d_local @ gradient_basis).reshape(2, 2)

        G = gradient_basis.reshape(n_local_hdiv, 2, 2)
        scale = jac_det * weight * quad_jacobian

        # Picard term: K[a,b] = sum_ij u_k[i] * G[b,i,j] * phi[a,j]
        picard = np.einsum('i,bij,aj->ab', u_k, G, transformed_basis)
        # extra Newton term: K[a,b] = phi[a] · (grad_u_k @ phi[b])
        newton = transformed_basis @ grad_u_k @ transformed_basis.T

        ke[:n_local_hdiv, :n_local_hdiv] += (picard + newton) * scale

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

        d_local = np.array([previous_d_coeffs[local_IEN_HDIV[i]] for i in range(n_local_hdiv)])
        u_k = d_local @ transformed_basis  # shape (2,): previous velocity at this qpt
        grad_u_k = (d_local @ gradient_basis).reshape(2, 2)

        # (u^k·∇)u^k = grad_u_k @ u_k
        adv_uk = grad_u_k @ u_k

        scale = jac_det * weight * quad_jacobian
        fe[:n_local_hdiv] += (transformed_basis @ adv_uk) * scale

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

        scale = jac_det * quad_wts * quad_jacobian

        # u-u block: L2 mass matrix — identity operator on velocity, NO viscosity
        ke[:n_local_hdiv, :n_local_hdiv] += scale * (transformed_basis @ transformed_basis.T)

        # u-p block: -∫ (∇·φ_a) ψ_b dΩ  and  p-u block: transpose
        div_phi = gradient_basis[:, 0] + gradient_basis[:, 3]  # shape (n_local_hdiv,)
        block = np.outer(div_phi, transformed_basis_L2) * scale
        ke[:n_local_hdiv, n_local_hdiv:] -= block
        ke[n_local_hdiv:, :n_local_hdiv] += block.T

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

        scale = jac_det * quad_wts * quad_jacobian
        fe[:n_local_hdiv] += (transformed_basis @ force) * scale

    return fe
