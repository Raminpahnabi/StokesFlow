#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:06:03 2024

@author: kendrickshepherd
"""

import sys
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt
import CE_ME_507_MultiDimensionalBasisFunctions as m_basis

# Depending on your approach, 
# You may need to import information from HW3 and/or
# HW6 to load in your unidimensional and multidimensional
# Lagrange basis function code 

# input polynomial degree, p,
#       interpolation points, pts,
#       the point of evaluation, xi,
#   and the basis fnction index, a

def LagrangeBasisEvaluation(p, pts, xi, a):
    # ensure valid input
    if (p+1 != len(pts)):
        sys.exit("The number of input points for interpolating must be the same as one plus the polynomial degree for interpolating")
    else:
        value = 1
        for m in range(0, p + 1):
            if m != a:
                value *= (xi - pts[m]) / (pts[a] - pts[m])
        return value
    
def LagrangeBasisParamDervEvaluation(p,pts,xi,a):
    dl_a = 0

    for j in range(p+1):
        if j != a:
            product = 1
            for b in range(p + 1):
                if b != a and b != j:
                    product *= (xi - pts[b]) / (pts[a] - pts[b])

            dl_a += product / (pts[a] - pts[j])
    return dl_a

# you may want to create a function here that 
# converts from a single index, A, and a set of 
# polynomial degrees into a multi-index indicating
# the indices of the univariate lagrange polynomials
# 
# you will need this functionality, though you do
# not need to complete this function for full credit
def GlobalToLocalIdxs(A,degs):
    multi_index = []
    nsd = len(degs) 
    for i in range(nsd):
        pi_plus_1 = degs[i] + 1  
        ai = A % pi_plus_1  
        multi_index.append(ai)
        A = A // pi_plus_1  
    return multi_index

# evaluate the partial derivative of a nD lagrange 
# basis function of index A in the "dim" dimension
# (e.g. derivative in xi is 0, in eta is 1)
def LagrangeBasisDervParamMultiD(A,degs,interp_pts,xis,dim):
    local_idx = GlobalToLocalIdxs(A, degs)

    result = 1

    for i in range(len(degs)):
        if i == dim:
            result *= LagrangeBasisParamDervEvaluation(degs[i], interp_pts[i], xis[i], local_idx[i])
        else:
            result *= LagrangeBasisEvaluation(degs[i], interp_pts[i], xis[i], local_idx[i])
    return

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
    plt.title('Derivatives of 1D Lagrange Basis Functions (Degree 2)')
    plt.xlabel('ξ')
    plt.ylabel('dN(ξ)/dξ')
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
                Z[j,i] = m_basis.MultiDimensionalBasisFunction(A,degs,interp_pts,xi_vals)
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
        ax.set_zlabel(r"$\frac{\partial N_A}{\partial xi_%d}$" % dim)
        plt.show()

interp_points = [-1, 0, 1]
PlotLagrangeBasisDerivatives(2, interp_points)

