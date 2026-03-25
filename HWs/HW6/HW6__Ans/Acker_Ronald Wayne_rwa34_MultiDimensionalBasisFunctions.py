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
def LagrangeBasisEvaluation(p,pts,xi,a):
    # ensure valid input
    if (p+1 != len(pts)):
        sys.exit("The number of input points for interpolating must be the same as one plus the polynomial degree for interpolating")
    # complete this code and return an appropriate value
    # as given in Hughes Equation 3.6.1
    L_a = 1.0
    for b in range(p + 1):
        if b != a:
            # Multiply by the Lagrange polynomial term for the a-th basis function
            L_a *= (xi - pts[b]) / (pts[a] - pts[b])
            
    return L_a
# which evaluated a Lagrane polynomial basis function
# import <UNCOMMENT THIS LINE AND INPUT YOUR FILENAME HERE>

# higher-dimensional basis function with multi-index
def MultiDimensionalBasisFunctionIdxs(idxs,degs,interp_pts,xis):
    # ensure valid input
    N_a = 1.0
    for i in range(len(xis)):
        p = degs[i]                    # Degree in the i-th dimension
        points = interp_pts[i]          # Interpolation points in the i-th dimension
        xi = xis[i]                     # Evaluation coordinate in the i-th dimension
        a = idxs[i]                     # Index for the basis function in the i-th dimension
        
        # Multiply by the 1D Lagrange basis for this dimension
        N_a *= LagrangeBasisEvaluation(p, points, xi, a)
    
    return N_a
    
# higher-dimensional basis function with single index
def MultiDimensionalBasisFunction(A,degs,interp_pts,xis):
    basisfunction = degs[0] + 1
    a0 = A % (basisfunction)
    idxs = [a0]
    for i in range(1,len(degs)):
        idxs.append(A // (basisfunction))
        basisfunction *= degs[i] + 1
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
        
# List of polynomial degrees to plot


# Generate plots for each set of polynomial degrees
degs = [1,1]
for i in range(4):
    PlotTwoDimensionalParentBasisFunction(i,degs,npts = 101,contours = True)
degs = [2,1]
for i in range(6):
    PlotTwoDimensionalParentBasisFunction(i,degs,npts = 101,contours = True)
degs = [2,2]
for i in range(9):
    PlotTwoDimensionalParentBasisFunction(i,degs,npts = 101,contours = True)
degs = [3,3]
for i in range(16):
    PlotTwoDimensionalParentBasisFunction(i,degs,npts = 101,contours = True)