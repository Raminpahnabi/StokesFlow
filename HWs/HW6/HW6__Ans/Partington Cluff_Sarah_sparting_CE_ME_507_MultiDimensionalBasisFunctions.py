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
# Given a set of points, pts, to interpolate
# and a polynomial degree, this function will
# evaluate the a-th basis function at the 
# location xi
def LagrangeBasisEvaluation(p,pts,xi,a):
    # ensure valid input
    if (p+1 != len(pts)):
        sys.exit("The number of input points for interpolating must be the same as one plus the polynomial degree for interpolating")
    numerator = 1
    denominator = 1
    
    for i in range(p+1):
        if i == a:
            continue
        numerator *= (xi - pts[i])   
        denominator *= (pts[a] - pts[i])
    solution = numerator / denominator
    return solution

    # complete this code and return an appropriate value
    # as given in Hughes Equation 3.6.1

# Plot the Lagrange polynomial basis functions
# COMMENT THIS CODE
def PlotLagrangeBasisFunctions(p,pts,n_samples = 101): # Define the function and the variables it accepts
    xis = np.linspace(min(pts),max(pts),n_samples) # The variable 'xis' contains equally spaced points ranging from the minimum value in pts and the max vlaue in pts, with a numeber of points equal to n_samples 
    fig, ax = plt.subplots() # Define how the plots should be set up
    for a in range(0,p+1): # Loop through indices ranging from 0 to p+1
        vals = [] # Create an empty array
        for xi in xis: # Look through each xi in xis
            vals.append(LagrangeBasisEvaluation(p, pts, xi, a)) # Append the solution of the basis function to the empty values array
                
        plt.plot(xis,vals) # Plot the evenlys spaced poins on the x-axis and the solutions to the basis functions corresponding to those points on the y-axis
        plt.show()
    ax.grid(linestyle='--') # Define the style of the grid on the plot
        
# Interpolate two-dimensional data
def InterpolateFunction(p,pts2D,n_samples = 101):
    
    pts = [x[0] for x in pts2D]
    coeffs = [x[1] for x in pts2D]

    xis = np.linspace(min(pts),max(pts),n_samples)
    ys = np.zeros(len(xis))

    for i in range(len(xis)):
        xi = xis[i]
        for j in range(0,p+1):
            ys[i] += (coeffs[j] * LagrangeBasisEvaluation(p,pts,xi,j))
        
    fig, ax = plt.subplots()    
    plt.plot(xis,ys)
    plt.scatter(pts, coeffs,color='r')
    plt.show()

# higher-dimensional basis function with multi-index
def MultiDimensionalBasisFunctionIdxs(idxs,degs,interp_pts,xis):
    Na = 1
    for i in range(len(degs)):
        Na *= LagrangeBasisEvaluation(degs[i],interp_pts[i],xis[i],idxs[i])
    return Na
    
# higher-dimensional basis function with single index
def MultiDimensionalBasisFunction(A,degs,interp_pts,xis):
    idxs = []
    idxs.append(A % (degs[0] + 1))

    for i in range(len(degs)-1):

        temporary = 1

        for j in range(len(degs)-1):
            temporary *= degs[j] + 1

        idxs.append(A // temporary) 

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
        

 # Four plots:
 
# degs = [1,1]
# degs = [2,1]
# degs = [2,2]
degs = [3,3]
 
 

n = (degs[0]+1)
m = (degs[1]+1)

for i in range(n*m):
    PlotTwoDimensionalParentBasisFunction(i,degs,npts = 101,contours = False)
    plt.show()   
    