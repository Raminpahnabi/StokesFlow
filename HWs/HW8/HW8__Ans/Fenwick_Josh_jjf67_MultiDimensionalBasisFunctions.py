#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:45:50 2024

@author: kendrickshepherd
"""

import sys
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt
from LagrangeBasisFunctions import LagrangeBasisEvaluation

# higher-dimensional basis function with multi-index
def MultiDimensionalBasisFunctionIdxs(idxs,degs,interp_pts,xis):
    lag_func = 1
    for i in range(len(idxs)):
            lag_func *= LagrangeBasisEvaluation(degs[i],interp_pts[i],xis[i],idxs[i])
    return lag_func
    
# converts from a single index, A, and a set of 
# polynomial degrees into a multi-index indicating
# the indices of the univariate lagrange polynomials
def GlobalToLocalIdxs(A,degs):
    idxs = []
    for i in range(len(degs)):
        if i == 0:
            idx = A % (degs[i] + 1)
        else:
            denominator = 1
            for j in range(i):
                denominator *= (degs[j] + 1)
            idx = A // denominator
        idxs.append(idx)
    return idxs

# higher-dimensional basis function with single index
def MultiDimensionalBasisFunction(A,degs,interp_pts,xis):
    idxs = GlobalToLocalIdxs(A,degs)

    lag_func = MultiDimensionalBasisFunctionIdxs(idxs,degs,interp_pts,xis)

    return lag_func
    
# plot of 2D basis functions with A a single index
def PlotTwoDimensionalParentBasisFunction(A,degs,npts = 101,contours = True):
    interp_pts = [np.linspace(-1,1,degs[i]+1) for i in range(0,len(degs))]
    xivals = np.linspace(-1,1,npts)
    etavals = np.linspace(-1,1,npts)
    
    Xi,Eta = np.meshgrid(xivals,etavals)
    Z = np.zeros(Xi.shape)
    
    for i in range(0,len(xivals)):
        for j in range(0,len(etavals)):
            Z[i,j] = MultiDimensionalBasisFunction(A, degs, interp_pts, [xivals[i],etavals[j]])
    
    # contour plot
    if contours:
        fig, ax = plt.subplots()
        surf = ax.contourf(Eta,Xi,Z,levels=100,cmap=matplotlib.cm.jet)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        fig.colorbar(surf)
        plt.show()
    # 3D surface plot
    else:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(Eta, Xi, Z, cmap=matplotlib.cm.jet,
                       linewidth=0, antialiased=False)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        ax.set_zlabel(r"$N(\xi,\eta)$")
        plt.show()


# PlotTwoDimensionalParentBasisFunction(0,[1,1],contours=False)
# PlotTwoDimensionalParentBasisFunction(3,[2,1],contours=False)
# PlotTwoDimensionalParentBasisFunction(5,[1,2],contours=False)
# PlotTwoDimensionalParentBasisFunction(7,[3,3],contours=False)