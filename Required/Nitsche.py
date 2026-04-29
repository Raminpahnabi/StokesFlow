#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:21:18 2026

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
sys.path.append(str(PROJECT_ROOT / 'Required'))

import Quadrature_Operations_Solutions_boundary as gq_bc
import CommonFuncs as cf

_FACE_NAME = {
    gq_bc.BoundaryFace.BOTTOM: 'bottom',
    gq_bc.BoundaryFace.TOP:    'top',
    gq_bc.BoundaryFace.LEFT:   'left',
    gq_bc.BoundaryFace.RIGHT:  'right',
}

# ===========================================================================
#   Nitsche functions modified to act ONLY on TANGENTIAL DOFs
# ===========================================================================
def LocalForceVector_Nitsche_IGA_2D(basis, deg, quad,quad_1D, gamma, elem, forcing_function,u_exact, boundary_value_function, nu=1,
                                    skip_faces=None):  # skip_faces: outflow faces only (e.g. ['right']); tangential Nitsche applied on all other boundaries
    kinematic_viscosity = nu 

    local_IEN_HDIV = basis.HDIV.connectivity(elem)
    n_local_hdiv = len(local_IEN_HDIV)
    local_IEN_L2 = basis.L2.connectivity(elem)
    n_local_L2 = len(local_IEN_L2)
    n_local_total = n_local_hdiv + n_local_L2

    fe = np.zeros(n_local_total)

    bdries = [gq_bc.BoundaryFace.BOTTOM,gq_bc.BoundaryFace.TOP,gq_bc.BoundaryFace.LEFT,gq_bc.BoundaryFace.RIGHT]
    bounds = cf._physical_domain_bounds(basis)
    _skip = set(skip_faces or [])  

    for bdry in bdries:
        if _FACE_NAME[bdry] in _skip:  # outflow: no weak tangential Dirichlet on this face
            continue
        xi_vals = gq_bc.GetFaceQuadraturePoints(quad_1D, bdry)
        if not cf._is_boundary_face(basis, elem, bdry, quad_1D, bounds):
            continue

        basis.localizeElement(elem)
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
            scale = quad_1D.jacobian * quad_wt_1d * jac_1d

            fe[:n_local_hdiv] += scale * (
                -(sigma_n @ u_val)
                + (gamma * 2 * kinematic_viscosity / face_length) * (phi @ u_val)
            )

    return fe



def LocalStiffnessMatrix_Nitsche_IGA_2D(basis, deg, quad, quad_1D, gamma, elem, boundary_value_function=None, nu=1,
                                        skip_faces=None): 
    kinematic_viscosity = nu  

    local_IEN_HDIV = basis.HDIV.connectivity(elem)
    n_local_hdiv = len(local_IEN_HDIV)
    local_IEN_L2 = basis.L2.connectivity(elem)
    n_local_L2 = len(local_IEN_L2)
    n_local_total = n_local_hdiv + n_local_L2

    ke = np.zeros((n_local_total, n_local_total))

    bdries = [gq_bc.BoundaryFace.BOTTOM, gq_bc.BoundaryFace.TOP,
              gq_bc.BoundaryFace.LEFT, gq_bc.BoundaryFace.RIGHT]
    bounds = cf._physical_domain_bounds(basis)
    _skip = set(skip_faces or [])  

    for bdry in bdries:
        if _FACE_NAME[bdry] in _skip:  
            continue
        xi_vals = gq_bc.GetFaceQuadraturePoints(quad_1D, bdry)
        if not cf._is_boundary_face(basis, elem, bdry, quad_1D, bounds):
            continue

        basis.localizeElement(elem)
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
            scale = quad_1D.jacobian * quad_wt_1d * jac_1d

            # velocity-velocity coupling (vectorized)
            ke[:n_local_hdiv, :n_local_hdiv] += scale * (
                -(sigma_n @ phi.T)                                                # consistency
                -(phi @ sigma_n.T)                                                # symmetry
                + (gamma * 2 * kinematic_viscosity / face_length) * (phi @ phi.T)    # penalty
            )

            # pressure-velocity coupling (vectorized)
            phi_dot_n = phi @ normal_unit  # (n_local_hdiv,)
            ke[:n_local_hdiv, n_local_hdiv:] += scale * np.outer(phi_dot_n, transformed_basis_L2_bc)

    return ke


def LocalStiffnessMatrix_Nitsche_IGA_2D_L2Projection(basis, deg, quad, quad_1D, gamma, elem):
    """
    Nitsche stiffness for L2-projection (penalty-only, no viscous consistency term).
    u-u (tangential): (gamma/h) * (phi_a · t̂)(phi_b · t̂)
    Penalizes the TANGENTIAL velocity component, which is the free DOF.
    """
    local_IEN_HDIV = basis.HDIV.connectivity(elem)
    n_local_hdiv = len(local_IEN_HDIV)
    local_IEN_L2 = basis.L2.connectivity(elem)
    n_local_L2 = len(local_IEN_L2)
    n_local_total = n_local_hdiv + n_local_L2

    ke = np.zeros((n_local_total, n_local_total))

    bdries = [gq_bc.BoundaryFace.BOTTOM, gq_bc.BoundaryFace.TOP,
              gq_bc.BoundaryFace.LEFT,   gq_bc.BoundaryFace.RIGHT]
    bounds = cf._physical_domain_bounds(basis)

    for bdry in bdries:
        xi_vals = gq_bc.GetFaceQuadraturePoints(quad_1D, bdry)
        if not cf._is_boundary_face(basis, elem, bdry, quad_1D, bounds):
            continue
        basis.localizeElement(elem)
        face_length = cf.compute_face_length(basis, xi_vals, quad_1D, bdry)

        for g in range(len(xi_vals)):
            quad_pts = xi_vals[g]
            basis.localizePoint(quad_pts)

            jac    = basis.jacobian()
            jac_1d = cf.JacobianOneD(jac, bdry)

            tangent      = cf.DifferentialVector(jac, bdry)
            tangent_unit = tangent / jac_1d            

            phi = basis.piolaTransformedHDIVBasis()      # (n_local_hdiv, 2)

            quad_wt_1d = quad_1D.quad_wts[g]
            scale = quad_1D.jacobian * quad_wt_1d * jac_1d

            # Project each basis function onto the tangential direction
            phi_t = phi @ tangent_unit                   # (n_local_hdiv,) — tangential component
            ke[:n_local_hdiv, :n_local_hdiv] += (gamma / face_length) * scale * np.outer(phi_t, phi_t)

    return ke


def LocalForceVector_Nitsche_IGA_2D_L2Projection(basis, deg, quad, quad_1D, gamma, elem, boundary_value_function,use_curve_geometry):
    """
    Nitsche force for L2-projection (penalty-only, no viscous consistency term).
    Adds: (gamma/h) * (phi_a · t̂) * g_t
    where g_t = g · t̂ is the tangential component of the Dirichlet datum.
    """
    local_IEN_HDIV = basis.HDIV.connectivity(elem)
    n_local_hdiv   = len(local_IEN_HDIV)
    local_IEN_L2   = basis.L2.connectivity(elem)
    n_local_L2     = len(local_IEN_L2)
    n_local_total  = n_local_hdiv + n_local_L2

    fe = np.zeros(n_local_total)

    bdries = [gq_bc.BoundaryFace.BOTTOM, gq_bc.BoundaryFace.TOP,
              gq_bc.BoundaryFace.LEFT,   gq_bc.BoundaryFace.RIGHT]
    bounds = cf._physical_domain_bounds(basis)

    for bdry in bdries:
        xi_vals = gq_bc.GetFaceQuadraturePoints(quad_1D, bdry)
        if not cf._is_boundary_face(basis, elem, bdry, quad_1D, bounds):
            continue
        basis.localizeElement(elem)
        face_length = cf.compute_face_length(basis, xi_vals, quad_1D, bdry)

        for g in range(len(xi_vals)):
            quad_pts = xi_vals[g]
            basis.localizePoint(quad_pts)

            jac    = basis.jacobian()
            jac_1d = cf.JacobianOneD(jac, bdry)

            tangent      = cf.DifferentialVector(jac, bdry)
            tangent_unit = tangent / jac_1d              # unit tangent t̂

            phi = basis.piolaTransformedHDIVBasis()      # (n_local_hdiv, 2)

            qpt_mapped = basis.mapping()
            x_g, y_g   = qpt_mapped[0], qpt_mapped[1]

            if boundary_value_function is not None:
                if not use_curve_geometry :
                    u_boundary = np.array(boundary_value_function(x_g, y_g))
                elif use_curve_geometry :
                    u_boundary = np.array(boundary_value_function(quad_pts[0], quad_pts[1]))
            else:
                u_boundary = np.array([0.0, 0.0])        

            g_t   = np.dot(u_boundary, tangent_unit)    # scalar tangential value
            u_val = g_t * tangent_unit                   # tangential vector

            quad_wt_1d = quad_1D.quad_wts[g]
            scale      = quad_1D.jacobian * quad_wt_1d * jac_1d

            # Penalty RHS: (gamma/h) * (phi_a · t̂) * g_t
            fe[:n_local_hdiv] += (gamma / face_length) * scale * (phi @ u_val)

    return fe
