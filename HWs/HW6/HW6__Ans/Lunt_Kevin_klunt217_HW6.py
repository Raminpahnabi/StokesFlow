# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 21:31:14 2024

@author: klunt
"""

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
        
for i in range(4):
    PlotTwoDimensionalParentBasisFunction(i, (1, 1), contours=False)     
for i in range(6):
    PlotTwoDimensionalParentBasisFunction(i, (2, 1), contours=False)
for i in range(9):
    PlotTwoDimensionalParentBasisFunction(i, (2, 2), contours=False)
for i in range(16):
    PlotTwoDimensionalParentBasisFunction(i, (3, 3), contours=False)      