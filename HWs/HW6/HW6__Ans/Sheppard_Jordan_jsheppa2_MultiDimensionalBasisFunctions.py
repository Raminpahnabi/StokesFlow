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
# which evaluated a Lagrange polynomial basis function
from LagrangePolys import LagrangeBasisEvaluation

# higher-dimensional basis function with multi-index
def MultiDimensionalBasisFunctionIdxs(idxs,degs,interp_pts,xis):
    individual_vals = np.array([LagrangeBasisEvaluation(deg, pts, xi, a) for deg, a, pts, xi in zip(degs, idxs, interp_pts, xis)])
    return np.prod(individual_vals)

    
# higher-dimensional basis function with single index
def MultiDimensionalBasisFunction(A,degs,interp_pts,xis):
    # Get multi-index from A
    n_sd = len(degs)
    idxs = np.zeros(n_sd, dtype=np.intp)
    idxs[0] = A % (degs[0] + 1)
    for i in range(1,n_sd):
        idxs[i] = A // np.prod(np.array([p + 1 for p in degs[:i]]))
    
    # Use multi-indexed version 
    return MultiDimensionalBasisFunctionIdxs(idxs, degs, interp_pts, xis)


    
# plot of 2D basis functions with A a single index
def PlotTwoDimensionalParentBasisFunction(A,degs, savename=None, npts = 101,contours = True):
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
        ax.set_box_aspect(aspect=None, zoom=0.8)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        fig.colorbar(surf)
    # 3D surface plot
    else:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(Eta, Xi, Z, cmap=matplotlib.cm.jet,
                       linewidth=0, antialiased=False)
        ax.set_box_aspect(aspect=None, zoom=0.9)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        ax.set_zlabel(r"$N(\xi,\eta)$")
    plt.suptitle(f"2D Lagrange Basis Function \nDegs = {degs}, A = {A}")
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    plt.show()
        

if __name__ == "__main__":
    degs = np.array([2, 2])
    num_funcs = np.prod(degs + np.ones_like(degs))
    for A in range(num_funcs):
        PlotTwoDimensionalParentBasisFunction(A, degs, contours=False)
