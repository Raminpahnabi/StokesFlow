# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:23:59 2024

@author: klunt
"""

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

def LagrangeBasisEvaluation(p,pts,xi,a):
    # ensure valid input
    if (p+1 != len(pts)):
        sys.exit("The number of input points for interpolating must be the same as one plus the polynomial degree for interpolating")
    product = 1
    for i in range(0, p + 1):
        if i == a:
            continue
        else:
            product *= (((xi) - pts[i]) / (pts[a] - pts[i]))
    return product

# higher-dimensional basis function with multi-index
def MultiDimensionalBasisFunctionIdxs(idxs,degs,interp_pts,xis):
    product2 = 1
    #Loop through each dimension 
    for i in range(len(idxs)):
        basis_function = LagrangeBasisEvaluation(degs[i], interp_pts[i], xis[i], idxs[i])
        product2 *= basis_function
    return product2
    
# higher-dimensional basis function with single index
def MultiDimensionalBasisFunction(A,degs,interp_pts,xis):
    #Break down single-index "A", into multi-index "a" to input into the other function
    a_indices = []
    n_basisfunctions = degs[0] + 1
    a_0 = A % n_basisfunctions
    a_indices.append(a_0)
    for i in range(1, len(degs)):
        a_indices.append(A // n_basisfunctions)
        n_basisfunctions *= (degs[i] + 1)
    return MultiDimensionalBasisFunctionIdxs(a_indices, degs, interp_pts, xis)

# input polynomial degree, p,
#       interpolation points, pts,
#       the point of evaluation, xi,
#   and the basis fnction index, a
def LagrangeBasisParamDervEvaluation(p,pts,xi,a):
    total = 0 
    for i in range(p + 1):
        if i != a:
            product = 1
            for j in range(p + 1):
                if j != i and j != a:
                    product *= ((xi - pts[j]) / (pts[a] - pts[j]))
            total += (1 / (pts[a] - pts[i])) * product
    return total

# you may want to create a function here that 
# converts from a single index, A, and a set of 
# polynomial degrees into a multi-index indicating
# the indices of the univariate lagrange polynomials
# 
# you will need this functionality, though you do
# not need to complete this function for full credit
def GlobalToLocalIdxs(A,degs):
    bfs = degs + 1
    a = A % bfs
    return a

# evaluate the partial derivative of a nD lagrange 
# basis function of index A in the "dim" dimension
# (e.g. derivative in xi is 0, in eta is 1)
def LagrangeBasisDervParamMultiD(A,degs,interp_pts,xis,dim):
    a = GlobalToLocalIdxs(A, degs[dim])
    dNa = LagrangeBasisParamDervEvaluation(degs[dim], interp_pts[dim], xis[dim], a)
    dNa *= MultiDimensionalBasisFunction(A, degs, interp_pts, xis)
    return dNa

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


p = 2
pts = [-1, 0, 1]
degs = [2, 2]
PlotLagrangeBasisDerivatives(p, pts)

degsx = 2
degsy = 2
xsplit = np.linspace(-1, 1, degsx + 1)
ysplit = np.linspace(-1, 1, degsy + 1)
degs = [degsx, degsy]
interp_pts = [xsplit, ysplit]
for A in range(9):
    PlotTwoDBasisFunctionParentDomain(A, degs, interp_pts, 0)
    PlotTwoDBasisFunctionParentDomain(A, degs, interp_pts, 1)
