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

# Depending on your approach, 
# You may need to import information from HW3 and/or
# HW6 to load in your unidimensional and multidimensional
# Lagrange basis function code 
import HW_3
# input polynomial degree, p,
#       interpolation points, pts,
#       the point of evaluation, xi,
#   and the basis fnction index, a
def LagrangeBasisParamDervEvaluation(p,pts,xi,a):
    dN_a = 0.0  # Start with zero for accumulation
    N_a = 1.0
    for i in range(p+1):
        if i != a:
            prod = 1.0
            for j in range(p+1):
                if j != a and j != i:
                    prod *= (xi - pts[j]) / (pts[a] - pts[j])
            dN_a += prod / (pts[a] - pts[i])
    return dN_a      

# you may want to create a function here that 
# converts from a single index, A, and a set of 
# polynomial degrees into a multi-index indicating
# the indices of the univariate lagrange polynomials
# 
# you will need this functionality, though you do
# not need to complete this function for full credit
def MultiDimensionalBasisFunction(A,degs):
    basisfunction = degs[0] + 1
    a0 = A % (basisfunction)
    idxs = [a0]
    for i in range(1,len(degs)):
        idxs.append(A // (basisfunction))
        basisfunction *= degs[i] + 1
    return idxs

# evaluate the partial derivative of a nD lagrange 
# basis function of index A in the "dim" dimension
# (e.g. derivative in xi is 0, in eta is 1)
def LagrangeBasisDervParamMultiD(A,degs,interp_pts,xis,dim):
    local_idxs = MultiDimensionalBasisFunction(A, degs)
    derivative = 1.0
    for d in range(len(degs)):
        if d == dim:
            derivative *= LagrangeBasisParamDervEvaluation(degs[d], interp_pts[d], xis[d], local_idxs[d])
        else:
            derivative *= HW_3.LagrangeBasisEvaluation(degs[d], interp_pts[d], xis[d], local_idxs[d])
    return derivative

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
        ax.set_zlabel(r"$\frac{\partial N_A}{\partial xi_%d}$" % dim)
        plt.show()

# Set up the degree and interpolation points for degree 2 in both directions
p = 2  # Degree
interp_pts = list(np.linspace(-1, 1, p+1))
interp_pts2 = list(np.linspace(-1, 1, p+1))  # [-1, 1] x [-1, 1]
degs = [p, p]  # Same degree in both xi and eta directions

for A in range((p+1)**2):  # 9 basis functions total for degree 2
    #print(f"Plotting basis function {A} derivative w.r.t. xi:")
    #PlotTwoDBasisFunctionParentDomain(A, degs, interp_pts, dim=0)  # Derivative w.r.t xi

    print(f"Plotting basis function {A} derivative w.r.t. eta:")
    PlotTwoDBasisFunctionParentDomain(A, degs, [interp_pts, interp_pts2], dim=1)  # Derivative w.r.t eta
    
   
PlotLagrangeBasisDerivatives(p,interp_pts)