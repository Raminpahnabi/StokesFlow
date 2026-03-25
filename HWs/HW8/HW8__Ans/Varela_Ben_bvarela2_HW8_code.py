import sys
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt

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

# higher-dimensional basis function with multi-index
def MultiDimensionalBasisFunctionIdxs(idxs,degs,interp_pts,xis):
    L = 1
    for i in range(0,len(degs)):
        L *= LagrangeBasisEvaluation(degs[i],interp_pts[i],xis[i],idxs[i])
    return L
    
# higher-dimensional basis function with single index
def MultiDimensionalBasisFunction(A,degs,interp_pts,xis):
    idxs = np.ones(len(degs),dtype=int)
    idxs[0] = A % (degs[0]+1)
    for i in range(1,len(degs)):
        for j in range(0,i):
            idxs[i] *= degs[j] + 1
        idxs[i] = A // idxs[i]
    idxs = idxs.tolist()
    return MultiDimensionalBasisFunctionIdxs(idxs,degs,interp_pts,xis)

def LagrangeBasisEvaluation(p,pts,xi,a):
    if (p+1 != len(pts)):
        sys.exit("The number of input points for interpolating must be the same as one plus the polynomial degree for interpolating")

    # complete this code and return an appropriate value
    # as given in Hughes Equation 3.6.1
    L = 1
    for b in range(0,p+1):
        if b != a:
            L *= (xi-pts[b])/(pts[a]-pts[b])
    return L

# input polynomial degree, p,
#       interpolation points, pts,
#       the point of evaluation, xi,
#   and the basis fnction index, a
def LagrangeBasisParamDervEvaluation(p,pts,xi,a):
    L = 0
    for b in range(0,p+1):
        if b != a:
            new_a = a if b > a else a-1
            pts_temp = [x for i,x in enumerate(pts) if i!=b]
            L += 1/(pts[a]-pts[b]) * LagrangeBasisEvaluation(p-1,pts_temp,xi,new_a)
    return L

# you may want to create a function here that 
# converts from a single index, A, and a set of 
# polynomial degrees into a multi-index indicating
# the indices of the univariate lagrange polynomials
# 
# you will need this functionality, though you do
# not need to complete this function for full credit
def GlobalToLocalIdxs(A,degs):
    bfs = degs[0]+1
    idxs = [A % (bfs)]
    for i in range(1,len(degs)):
        idxs.append(A // bfs)
        bfs *= degs[i]+1
    return idxs

# evaluate the partial derivative of a nD lagrange 
# basis function of index A in the "dim" dimension
# (e.g. derivative in xi is 0, in eta is 1)
def LagrangeBasisDervParamMultiD(A,degs,interp_pts,xis,dim):
    idxs = GlobalToLocalIdxs(A,degs)
    Ld = 1.0
    for i in range(0,len(degs)):
        if i == dim:
            Ld *= LagrangeBasisParamDervEvaluation(degs[i],interp_pts[i],xis[i],idxs[i])
        else:
            Ld *= LagrangeBasisEvaluation(degs[i],interp_pts[i],xis[i],idxs[i])
    return Ld

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
    etavals = np.linspace(interp_pts[1][0],interp_pts[1][-1],npts+1)
    
    Xi,Eta = np.meshgrid(xivals,etavals)
    Z = np.zeros(Xi.shape)
    
    for i in range(0,len(xivals)):
        for j in range(0,len(etavals)):
            xi_vals = [xivals[i],etavals[j]]
            if dim < 0:
                Z[j,i] = MultiDimensionalBasisFunction(A,degs,interp_pts,xi_vals)
                continue
            else:
                Z[j,i] = LagrangeBasisDervParamMultiD(A,degs,interp_pts,xi_vals,dim)

    if contours:
        fig, ax = plt.subplots()
        surf = ax.contourf(Xi,Eta,Z,levels=100,cmap=matplotlib.colormaps['jet'])
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        fig.colorbar(surf)
        plt.title(r"$\frac{\partial N_%d}{\partial \xi_%d}$" % (A, dim))
        plt.show()
    else:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(Xi, Eta, Z, cmap=matplotlib.colormaps['jet'],
                        linewidth=0, antialiased=False)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        ax.set_zlabel(r"$\frac{\partial N_A}{\partial xi_%d}$" % dim)
        plt.show()
