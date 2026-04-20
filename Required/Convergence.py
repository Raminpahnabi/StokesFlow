#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:20:27 2026

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

# import splines as spline
import matplotlib.pyplot as plt
import Solver_StokesFlow as ss
import CommonFuncs as cf
import NormalizedPressure as npre

#####################################################################################################
##################################################### Convergence Rate  #############################
#####################################################################################################
def EvaluateSolution_2D_Hdiv(basis, elem, xi, eta, dtotal):
    # Evaluate velocity solution at a point
    local_IEN_hdiv = basis.HDIV.connectivity(elem)
    n_local_hdiv = len(local_IEN_hdiv)

    basis.localizeElement(elem)
    basis.localizePoint([xi, eta])

    transformed_basis = basis.piolaTransformedHDIVBasis()
   
    uh_val = np.zeros(2)
    for a in range(0, n_local_hdiv): # 12
        A = local_IEN_hdiv[a]
        dA_hdiv = dtotal[A]
        uh_val += dA_hdiv * transformed_basis[a]
        
    return uh_val

def EvaluateSolution_2D_L2(basis, elem, xi, eta, dtotal):
    local_IEN_L2 = basis.L2.connectivity(elem)
    n_local_L2 = len(local_IEN_L2)
    local_IEN_hdiv = basis.HDIV.connectivity(elem)
    n_local_hdiv = len(local_IEN_hdiv)
    
    n_hdiv_total = cf.GetNumberH1FirstComponent(basis)[0] + cf.GetNumberH1FirstComponent(basis)[1]

    basis.localizeElement(elem)
    basis.localizePoint([xi, eta])

    phi_L2 = basis.piolaTransformedL2() 
    
    p_val = 0
    for b in range(0,n_local_L2):
        B = local_IEN_L2[b] + n_hdiv_total
        dA_L2 = dtotal[B]
        p_val += dA_L2 * phi_L2[b]
        
    return p_val

def compute_all_element_areas(basis, quad):
    areas = []
    for elem in basis.elements():
        basis.localizeElement(elem)
        element_area = 0.0
        for iqpt in range(len(quad.quad_wts)):
            weight = quad.quad_wts[iqpt]
            basis.localizePoint(quad.quad_pts[iqpt])
            jac_det = basis.jacobianDeterminant()
            element_area += jac_det * weight * quad.jacobian
        areas.append(element_area)
    return areas


def compute_largest_element_area(basis, quad):
    areas = compute_all_element_areas(basis, quad)
    return max(areas)

def compute_convergence_error(basis, d_coeffs, quad, exact_solution, isHDIV=True):
    # isHDIV: True for velocity error, False for pressure error
    total_error = 0.0
    # n_hdiv = basis.HDIV.numTotalFunctions()
    # n_l2 = basis.L2.numTotalFunctions()
    

    if not isHDIV:
        # Compute average pressure from the solution
        average_pressure = 0.0
        total_area = 0.0
        
        for elem in basis.elements():
            basis.localizeElement(elem)
            for iqpt in range(len(quad.quad_wts)):
                weight = quad.quad_wts[iqpt]
                qpt = quad.quad_pts[iqpt]
                basis.localizePoint(qpt)
                jac_det = basis.jacobianDeterminant()
                
                ph = EvaluateSolution_2D_L2(basis, elem, qpt[0], qpt[1], d_coeffs)
                element_area = jac_det * weight * quad.jacobian
                average_pressure += ph * element_area
                total_area += element_area
        
        if total_area > 0:
            average_pressure = average_pressure / total_area
        else:
            average_pressure = 0.0
    else:
        average_pressure = 0.0  # Not used for velocity
       
        
    # Compute error
    for elem in basis.elements():
        basis.localizeElement(elem)
        
        if isHDIV:  # For HDIV components (velocity)
            for iqpt in range(len(quad.quad_wts)):
                weight = quad.quad_wts[iqpt]
                qpt = quad.quad_pts[iqpt]
                basis.localizePoint(qpt)
                jac_det = basis.jacobianDeterminant()
                
                qpt_mapped = basis.mapping() 
                x_g, y_g = qpt_mapped[0], qpt_mapped[1]
                
                u_exact = np.array(exact_solution(x_g, y_g))  
                uh_xy = EvaluateSolution_2D_Hdiv(basis, elem, qpt[0], qpt[1], d_coeffs)
                
                error_xy = u_exact - uh_xy
                error_squared = np.dot(error_xy, error_xy)
                    
                total_error += error_squared * jac_det * weight * quad.jacobian   
        
        else:  # for pressure terms
            for iqpt in range(len(quad.quad_wts)): 
                weight = quad.quad_wts[iqpt]
                qpt = quad.quad_pts[iqpt]
                basis.localizePoint(qpt)
                jac_det = basis.jacobianDeterminant()
                
                qpt_mapped = basis.mapping()
                x_g, y_g = qpt_mapped[0], qpt_mapped[1]
                
                p_exact = exact_solution(x_g, y_g)
                ph = EvaluateSolution_2D_L2(basis, elem, qpt[0], qpt[1], d_coeffs)
                
                # Normalize pressure by subtracting average (pressure is only determined up to a constant)
                ph_normalized = ph - average_pressure
                
                error_p = p_exact - ph_normalized
                error_squared = error_p**2
                
                total_error += error_squared * jac_det * weight * quad.jacobian

    total_error = np.sqrt(total_error)

    return total_error


def compute_pressure_convergence_error(basis, d_coeffs, quad, exact_solution_l2):
    # mean of numerical pressure (should be ~0 after NormalizePressureCoefficients)
    mean_p_h = npre.EvaluateMeanPressure(basis, d_coeffs, quad)

    # compute mean of the exact pressure so we compare zero-mean quantities on both sides.
    # Without this, the error includes a constant floor = mean(p_exact) that never converges,
    # masking good pressure convergence for high degrees (visible as ~1.74e-04 stagnation).
    total_exact = 0.0
    total_area  = 0.0
    for elem in basis.elements():
        basis.localizeElement(elem)
        for iqpt in range(len(quad.quad_wts)):
            weight  = quad.quad_wts[iqpt]
            qpt     = quad.quad_pts[iqpt]
            basis.localizePoint(qpt)
            jac_det = basis.jacobianDeterminant()
            qpt_mapped = basis.mapping()
            x_g, y_g = qpt_mapped[0], qpt_mapped[1]
            dV = jac_det * weight * quad.jacobian
            total_exact += exact_solution_l2(x_g, y_g) * dV
            total_area  += dV
    mean_p_exact = total_exact / total_area  #ns mean of exact pressure over the domain

    total_error = 0.0
    for elem in basis.elements():
        basis.localizeElement(elem)
        for iqpt in range(len(quad.quad_wts)):
            weight  = quad.quad_wts[iqpt]
            qpt     = quad.quad_pts[iqpt]
            basis.localizePoint(qpt)
            jac_det = basis.jacobianDeterminant()
            qpt_mapped = basis.mapping()
            x_g, y_g   = qpt_mapped[0], qpt_mapped[1]
            p_exact_centered = exact_solution_l2(x_g, y_g) - mean_p_exact  #ns subtract exact mean
            p_h = float(EvaluateSolution_2D_L2(basis, elem, qpt[0], qpt[1], d_coeffs))
            p_h_centered = p_h - mean_p_h                                   #ns subtract numerical mean
            total_error += (p_exact_centered - p_h_centered)**2 * jac_det * weight * quad.jacobian

    return float(np.sqrt(total_error))


def plot_error_vs_log_h(refined_basis_list, deg, quad, quad_1D, gamma, forcing_function, 
                        exact_solution, boundary_value_function, isHDIV):

    errors = []
    h_values = []
    print("\n[Convergence diagnostics]")  
    print("level | n_elem | n_hdiv_total | h | error")
    
    for ilevel, refined_basis in enumerate(refined_basis_list):  
        # Solve Stokes equations
        d_coeffs = ss.Stokes(refined_basis, deg, quad, quad_1D, gamma, forcing_function, exact_solution,
                        boundary_conditions=None, boundary_value_function=boundary_value_function)
        
        # Compute error
        total_error = compute_convergence_error(refined_basis, d_coeffs, quad, exact_solution, isHDIV)
        errors.append(total_error)

        # Compute the square root of the largest element's area to get h
        h = np.sqrt(compute_largest_element_area(refined_basis, quad))
        h_values.append(h)
        
        #per-level metadata for debugging non-monotone convergence
        n_elem = len(list(refined_basis.elements()))  
        n_hdiv_total = cf.GetNumberH1FirstComponent(refined_basis)[0] + cf.GetNumberH1FirstComponent(refined_basis)[1]  
        print(f"{ilevel:5d} | {n_elem:6d} | {n_hdiv_total:12d} | {h:.8e} | {total_error:.8e}")  

    log_h_values = np.log(h_values)
    log_errors = np.log(errors)

    # Compute convergence rate (slope of log-log plot)
    slopes = []
    for i in range(1,len(log_errors)):
        slopes.append((log_errors[i] - log_errors[i-1]) / (log_h_values[i] - log_h_values[i-1]))


    plt.figure(figsize=(8, 6))
    plt.plot(log_h_values, log_errors, marker='o', linestyle='--', color='b', label=r'$||e||_0$')
    plt.xlabel(r'$\log(h)$')
    plt.ylabel(r'$\log(||e||_0)$')
    if isHDIV:
        plt.title(r'Velocity Convergence Plot: $||e||_0$ vs $\log(h)$')
    else:
        plt.title(r'Pressure Convergence Plot: $||e||_0$ vs $\log(h)$')
    plt.legend()
    plt.grid(True)
    plt.show()

    return slopes, h_values, errors
