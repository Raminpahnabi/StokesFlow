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

# IMPORT (or copy) your code from HW3 here
# which evaluated a Lagrane polynomial basis function
import lagrangePoly as lp

# higher-dimensional basis function with multi-index
def MultiDimensionalBasisFunctionIdxs(idxs,degs,interp_pts,xis):
    L = 1
    for i in range(0,len(degs)):
        L *= lp.LagrangeBasisEvaluation(degs[i],interp_pts[i],xis[i],idxs[i])
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
        surf = ax.contourf(Eta,Xi,Z,levels=100,cmap=matplotlib.colormaps['jet'])
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        fig.colorbar(surf)
        plt.show()
    # # 3D surface plot
    # else:
    #     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #     surf = ax.plot_surface(Eta, Xi, Z, cmap=matplotlib.colormaps['jet'],linewidth=0, antialiased=False)
    #     ax.set_xlabel(r"$\xi$")
    #     ax.set_ylabel(r"$\eta$")
    #     ax.set_zlabel(r"$N(\xi,\eta)$")
    #     plt.show()
    
    