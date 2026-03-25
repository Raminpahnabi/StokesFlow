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
import CE_ME_507_MultiDimensionalBasisFunctions as HW6
import CE_ME_507_Lagrange_Basis_Function_Code as HW3

# input polynomial degree, p,
#       interpolation points, pts,
#       the point of evaluation, xi,
#   and the basis fnction index, a
def LagrangeBasisParamDervEvaluation(p,pts,xi,a):
    dla = 0
    for j in range(0,p+1):
        if j == a:
            continue
        else:
            if j<=a:
                a_new = a-1
            else:
                a_new = a
            partial_pts = pts[0:j] + pts[j+1:]
            part_val = HW3.LagrangeBasisEvaluation(p-1, partial_pts, xi, a_new)
            dla += (1/(pts[a]-pts[j]))*part_val
    return dla

# you may want to create a function here that 
# converts from a single index, A, and a set of 
# polynomial degrees into a multi-index indicating
# the indices of the univariate lagrange polynomials
# 
# you will need this functionality, though you do
# not need to complete this function for full credit
def GlobalToLocalIdxs(A,degs):
    basis_functions = degs[0] + 1
    a = A % basis_functions
    idxs = [a]
    for i in range(1,len(degs)):
        idxs.append(A//basis_functions)
        basis_functions *= (degs[i] + 1)
    return idxs

# evaluate the partial derivative of a nD lagrange 
# basis function of index A in the "dim" dimension
# (e.g. derivative in xi is 0, in eta is 1)
def LagrangeBasisDervParamMultiD(A,degs,interp_pts,xis,dim):
    idxs = GlobalToLocalIdxs(A, degs)
    Nap = 1
    for j in range(0,len(degs)):
        if j == dim:
            Nap *= LagrangeBasisParamDervEvaluation(degs[j], interp_pts[j], xis[j], idxs[j])
        else:
            Nap *= HW3.LagrangeBasisEvaluation(degs[j], interp_pts[j], xis[j], idxs[j])
    return Nap

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
def PlotTwoDBasisFunctionParentDomain(A,degs,interp_pts,dim,npts=21,contours=True):
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

#Problem 3
p =2
pts = list(np.linspace(-1, 1, p+1))
PlotLagrangeBasisDerivatives(p, pts)

#Problem 4
p = 2
degs = [p,p]
dim = 1
A = (p+1) * (p+1)
interp_pts = list(np.linspace(-1, 1, p+1))
for i in range(0,A):
    PlotTwoDBasisFunctionParentDomain(i, degs, [interp_pts,interp_pts], dim)