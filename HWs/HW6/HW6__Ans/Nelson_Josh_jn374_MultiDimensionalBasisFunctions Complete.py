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
# import <UNCOMMENT THIS LINE AND INPUT YOUR FILENAME HERE>

def LagrangeBasisEvaluation(p,pts,xi,a):
    # ensure valid input
    output = 1
    if (p+1 != len(pts)):
        sys.exit("The number of input points for interpolating must be the same as one plus the polynomial degree for interpolating")

    for i in range(0,p+1):

        if i != a:
            output *=  (xi-pts[i]) / (pts[a]-pts[i])   
            
    return output

# higher-dimensional basis function with multi-index
def MultiDimensionalBasisFunctionIdxs(idxs,degs,interp_pts,xis):
    MultDimBasis = 1
    for i in range(len(idxs)):
        MultDimBasis *= LagrangeBasisEvaluation(degs[i], interp_pts[i], xis[i], idxs[i])
    return MultDimBasis
    


# higher-dimensional basis function with single index
def MultiDimensionalBasisFunction(A,degs,interp_pts,xis):
    multi_index = []
    horizShape = degs[0] + 1
    a = A % horizShape
    index = [a]
    for i in range(1,len(degs)):
       index.append(A // horizShape)
       horizShape *= (degs[i] + 1) 
    
    
    return MultiDimensionalBasisFunctionIdxs(index, degs, interp_pts, xis)   
# plot of 2D basis functions with A a single index
def PlotTwoDimensionalParentBasisFunction(A,degs,npts = 101,contours = False):
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
        
    return

def printPlots(degs):
    
   shape = (degs[0]+1) * (degs[1] +1)
   for i in range(shape):
       
       PlotTwoDimensionalParentBasisFunction(i, degs) 
    
    
   return



printPlots([3,3])

# PlotTwoDimensionalParentBasisFunction(3,[1,1])
# PlotTwoDimensionalParentBasisFunction(3,[2,1])
# PlotTwoDimensionalParentBasisFunction(3,[2,2])
# PlotTwoDimensionalParentBasisFunction(3,[3,3])
    
    

        