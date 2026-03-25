#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:06:03 2024

@author: Jordan Sheppard 
"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

# Depending on your approach, 
# You may need to import information from HW3 and/or
# HW6 to load in your unidimensional and multidimensional
# Lagrange basis function code 
from LagrangePolys_OLD import LagrangeBasisEvaluation as Lagrange1D
from MultiDimensionalLagrangePolys_OLD import MultiDimensionalBasisFunctionIdxs as LagrangeND

# input polynomial degree, p,
#       interpolation points, pts,
#       the point of evaluation, xi,
#   and the basis fnction index, a
def LagrangeBasisParamDervEvaluation(p,pts,xi,a):
    """Evaluate the derivative of the a-th 1-dimensional Lagrange
    basis function of degree p (through points in pts) at the point
    xi.
    
    Args:
        p (int): Degree of polynomial
        pts (list[float]): A list of points to interpolate 
        xi (float): The point to evaluate the Lagrange basis
            function'sderivative  at 
        a (int): The number of the polynomial to use

    Returns:
        float: The numerical value of the derivative at that point
    """
    total = 0.0
    for j in range(p+1):
        if j != a:
            frac = 1/(pts[a] - pts[j])
            new_pts = pts[:j] + pts[j+1:] 
            
            if j < a:
                total += (frac * Lagrange1D(p-1, new_pts, xi, a-1))
            elif j > a:
                total += (frac * Lagrange1D(p-1, new_pts, xi, a))
    return total
    

# you may want to create a function here that 
# converts from a single index, A, and a set of 
# polynomial degrees into a multi-index indicating
# the indices of the univariate lagrange polynomials
# 
# you will need this functionality, though you do
# not need to complete this function for full credit
def GlobalToLocalIdxs(A,degs):
    # Get multi-index from A
    n_sd = len(degs)
    idxs = np.zeros(n_sd, dtype=np.intp)
    idxs[0] = A % (degs[0] + 1)
    for i in range(1,n_sd):
        idxs[i] = A // np.prod(np.array([p + 1 for p in degs[:i]]))
    return idxs

# evaluate the partial derivative of a nD lagrange 
# basis function of index A in the "dim" dimension
# (e.g. derivative in xi is 0, in eta is 1)
def LagrangeBasisDervParamMultiD(A,degs,interp_pts,xis,dim):
    # Get multi-index from A 
    idxs = GlobalToLocalIdxs(A, degs) 

    # Evaluate 1d-derivative of the dim dimension
    deriv_val = LagrangeBasisParamDervEvaluation(degs[dim], interp_pts[dim], xis[dim], idxs[dim])
    
    # Evaluate product of other basis functions
    other_funcs = LagrangeND(
        idxs[:dim] + idxs[dim+1:],
        degs[:dim] + degs[dim+1:],
        interp_pts[:dim] + interp_pts[dim+1:],
        xis[:dim] + xis[dim+1:]
    )
    for i, (p, a, xi, pts) in enumerate(zip(degs, idxs, xis, interp_pts)):
        if i != dim:
            deriv_val *= Lagrange1D(p, pts, xi, a)
    return deriv_val 
    
        


# Plot the Lagrange polynomial basis function
# derivatives
def PlotLagrangeBasisDerivatives(p,pts,n_samples = 101):
    xis = np.linspace(min(pts),max(pts),n_samples)
    fig, ax = plt.subplots()
    for a in range(0,p+1):
        vals = []
        for xi in xis:
            vals.append(LagrangeBasisParamDervEvaluation(p, pts, xi, a))
                
        plt.plot(xis,vals)
    ax.grid(linestyle='--')
    plt.show()

# plot a basis function defined on a parent
# domain; this is similar to what was
# in a previous homework, but slightly generalized                
def PlotTwoDBasisFunctionParentDomain(A,degs,interp_pts,dim,npts=21,contours=False):
    xivals = np.linspace(interp_pts[0][0],interp_pts[0][-1],npts+1)
    etavals = np.linspace(interp_pts[1][0],interp_pts[1][-1],npts)
    
    Xi,Eta = np.meshgrid(xivals,etavals)
    Z = np.zeros(Xi.shape)
    
    for i in range(0,len(xivals)):
        for j in range(0,len(etavals)):
            xi_vals = [xivals[i],etavals[j]]
            if dim < 0:
                # if you imported your multidimensional
                # lagrange basis function code as
                # module "m_basis", uncomment the line below
                #Z[j,i] = m_basis.MultiDimensionalBasisFunction(A,degs,interp_pts,xi_vals)
                continue
            else:
                Z[j,i] = LagrangeBasisDervParamMultiD(A,degs,interp_pts,xi_vals,dim)

    if contours:
        fig, ax = plt.subplots()
        surf = ax.contourf(Xi,Eta,Z,levels=100,cmap=matplotlib.cm.jet)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        fig.colorbar(surf)
        plt.show()
    else:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(Xi, Eta, Z, cmap=matplotlib.cm.jet,
                       linewidth=0, antialiased=False)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        ax.set_zlabel(r"$\frac{\partial N_A}{\partial \xi_%d}$" % dim)
        ax.set_box_aspect(aspect=None, zoom=0.8)
        plt.show()

