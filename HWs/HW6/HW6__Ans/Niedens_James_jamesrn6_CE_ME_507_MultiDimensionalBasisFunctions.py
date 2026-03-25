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
import CE_ME_507_Lagrange_Basis_Function_Code as HW3

# higher-dimensional basis function with multi-index
def MultiDimensionalBasisFunctionIdxs(idxs,degs,interp_pts,xis):
    l_a = 1
    for b in range(len(idxs)):
        l_a *= HW3.LagrangeBasisEvaluation(degs[b], interp_pts[b], xis[b], idxs[b])
    return l_a
    
# higher-dimensional basis function with single index
def MultiDimensionalBasisFunction(A,degs,interp_pts,xis):
    basis_functions = degs[0] + 1
    a = A % basis_functions
    idxs = [a]
    for i in range(1,len(degs)):
        idxs.append(A//basis_functions)
        basis_functions *= (degs[i] + 1)
    return MultiDimensionalBasisFunctionIdxs(idxs, degs, interp_pts, xis)
    
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
        
# for i in range(0,4):
#     PlotTwoDimensionalParentBasisFunction(i, [1,1])
    
# for i in range(0,6):
#     PlotTwoDimensionalParentBasisFunction(i, [2,1])

# for i in range(0,9):
#     PlotTwoDimensionalParentBasisFunction(i, [2,2])

for i in range(0,16):
    PlotTwoDimensionalParentBasisFunction(i, [3,3])
    