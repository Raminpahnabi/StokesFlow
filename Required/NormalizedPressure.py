#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:21:01 2026

@author: raminpahnabi
"""
import sys
import os
import numpy as np
sys.path.insert(0, '/Users/raminpahnabi/Documents/BYU/sweeps/build/src/api')
sys.path.append(os.path.join(os.getcwd(), 'HWs'))
sys.path.append(os.path.join(os.getcwd(), 'Required'))

def EvaluateAveragePressure(basis, d_coeffs, quad):
    n_l2 = basis.L2.numTotalFunctions()
    pressure_terms = d_coeffs[-n_l2:]
    average_pressure = 0
    
    for elem in basis.elements():
        basis.localizeElement(elem)
        local_IEN_L2 = basis.L2.connectivity(elem)
       
        for iqpt in range(len(quad.quad_wts)):
            weight  = quad.quad_wts[iqpt]
            qpt     = quad.quad_pts[iqpt]
            basis.localizePoint(qpt)
            jac_det = basis.jacobianDeterminant()
            
            transformed_basis_L2 = basis.piolaTransformedL2() 

            ph = sum(pressure_terms[local_IEN_L2[i]] * transformed_basis_L2[i] for i in range(len(transformed_basis_L2)))
            average_pressure += ph * weight * jac_det * quad.jacobian
            
    return average_pressure

def EvalLocalL2forcepressure(basis,deg, quad, quad_1D, elem, alpha, boundary_conditions=None):
    # Connectivity for HDIV and L2 spaces
    local_IEN_L2 = basis.L2.connectivity(elem)
    n_local_L2 = len(local_IEN_L2)

    fe = np.zeros(n_local_L2)

    for iqpt in range(len(quad.quad_wts)):
        weight  = quad.quad_wts[iqpt]
        qpt     = quad.quad_pts[iqpt]
        quad_jacobian = quad.jacobian
        basis.localizePoint(qpt)
        jac_det = basis.jacobianDeterminant()       
        
        transformed_basis_L2 = basis.piolaTransformedL2()
        qpt_mapped = basis.mapping()
        x_g, y_g = qpt_mapped[0], qpt_mapped[1]
        
        for a in range(n_local_L2):
            fe[a] +=  float(transformed_basis_L2[a] * weight * quad_jacobian * jac_det * alpha(x_g, y_g))        
    return fe

def EvalLocalL2stiffnesspressure(basis, deg, quad, quad_1D, elem, boundary_conditions=None):
    local_IEN_L2 = basis.L2.connectivity(elem)
    n_local_L2 = len(local_IEN_L2)

    ke = np.zeros((n_local_L2, n_local_L2))
    
    for iqpt in range(0,len(quad.quad_wts)):
        weight = quad.quad_wts[iqpt]
        qpt    = quad.quad_pts[iqpt]
        quad_jacobian = quad.jacobian
        basis.localizePoint(qpt)
        jac_det = basis.jacobianDeterminant()
       
        transformed_basis_L2 = basis.piolaTransformedL2()    
        
        for a in range(n_local_L2):
            for b in range(a,n_local_L2):
                ke[a, b] += transformed_basis_L2[a]*transformed_basis_L2[b] * jac_det * weight * quad_jacobian
                ke[b, a] = ke[a, b]
    return ke

def L2PressureSolver(basis, deg, quad, quad_1D, alpha, boundary_conditions=None):
    n_l2 = basis.L2.numTotalFunctions()

    K = np.zeros((n_l2, n_l2))
    F = np.zeros(n_l2)

    for elem in basis.elements():
        basis.localizeElement(elem)
        k_e = EvalLocalL2stiffnesspressure(basis, deg, quad, quad_1D, elem, boundary_conditions=None)
        f_e = EvalLocalL2forcepressure(basis,deg, quad, quad_1D, elem, alpha, boundary_conditions=None)

        local_IEN_L2 = basis.L2.connectivity(elem)
        
        for a in range(0,len(local_IEN_L2)):
            A = local_IEN_L2[a]
            F[A] += f_e[a]
     
            for b in range(0,len(local_IEN_L2)):
                B = local_IEN_L2[b]
                K[A,B] += k_e[a,b]
                # K[B,A] +=  k_e[b,a]
    d = np.linalg.solve(K,F)
    
    return d