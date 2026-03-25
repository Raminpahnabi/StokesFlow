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
from LagrangeBasisFunctions import LagrangeBasisEvaluation
from MultiDimensionalBasisFunctions import GlobalToLocalIdxs

# Depending on your approach, 
# You may need to import information from HW3 and/or
# HW6 to load in your unidimensional and multidimensional
# Lagrange basis function code 

# input polynomial degree, p,
#       interpolation points, pts,
#       the point of evaluation, xi,
#   and the basis function index, a
def LagrangeBasisParamDervEvaluation(p,pts,xi,a):
    funcderiv = 0
    for j in range(len(pts)):
        if j != a:
            funcj = 1 / (pts[a] - pts[j])
            temppts = pts.copy()
            del temppts[j]
            ptemp = p-1
            if j>a:
                atemp = a
            if j<a:
                atemp = a-1
            funcj *= LagrangeBasisEvaluation(ptemp,temppts,xi,atemp)
            funcderiv += funcj
    return funcderiv

# evaluate the partial derivative of a nD lagrange 
# basis function of index A in the "dim" dimension
# (e.g. derivative in xi is 0, in eta is 1)
def LagrangeBasisDervParamMultiD(A,degs,interp_pts,xis,dim):
    npd = len(degs)
    idxs = GlobalToLocalIdxs(A,degs)

    Derv = 1
    j = dim
    for i in range(npd):
        if i == j:
            Derv *= LagrangeBasisParamDervEvaluation(degs[i],interp_pts[i],xis[i],idxs[i])
        else:
            Derv *= LagrangeBasisEvaluation(degs[i],interp_pts[i],xis[i],idxs[i])

    return Derv

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


def Homework8():
    p=2
    x_interval = list(np.linspace(-1,1,p+1))
    PlotLagrangeBasisDerivatives(p,x_interval)


    y_interval = x_interval
    intervals = [x_interval,y_interval]
    degs=[p,p]
    for A in range((degs[0]+1)*(degs[1]+1)):
        PlotTwoDBasisFunctionParentDomain(A,degs,intervals,len(degs))

Homework8()