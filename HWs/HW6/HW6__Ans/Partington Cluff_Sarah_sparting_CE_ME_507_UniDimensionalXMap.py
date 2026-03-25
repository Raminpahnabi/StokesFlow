#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:18:02 2024

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

# x(\xi) = \sum_{a=0}^p x_a * N_a(\xi)
def XMap(deg,spatial_pts,interp_pts,xi):
    
    xe = []
    Na = 0

    for a in range(deg+1):

        Na += spatial_pts[a]*LagrangeBasisEvaluation(deg,interp_pts,xi,a)
        
    xe.append(Na)

    return xe

def PlotXMap(deg,spatial_pts,interp_pts, npts=101, contours=True):
    # parametric points to evaluate in Lagrange basis function
    xi_vals = np.linspace(interp_pts[0],interp_pts[-1],npts)
    
    # evaluate and plot as a line
    if not contours:
        xs = []
        for i in range(0,len(xi_vals)):
            xi = xi_vals[i]
            xs.append(XMap(deg,spatial_pts,interp_pts,xi))
        
        plt.plot(xi_vals,xs)
        plt.show()

    # evaluate as a contour plot
    else:
        Xi,Xi2 = np.meshgrid(xi_vals,[-0.2,0.2])
        Z = np.zeros(Xi.shape)
        for i in range(0,len(xi_vals)):
            xi = xi_vals[i]
            x = XMap(deg,spatial_pts,interp_pts,xi)
            for j in range(0,2):
                Z[j,i] = x

        fig, ax = plt.subplots()
        surf = ax.contourf(Z,Xi2,Xi,levels=100,cmap=matplotlib.cm.binary)
        ax.set_xlabel(r"$x$")
        ax.yaxis.set_ticklabels([])
        fig.colorbar(surf)
        plt.show()

# deg = 1
# spatial_pts = [0,1]
# interp_pts = [-1,1]

# deg = 2
# spatial_pts = [0.5,1,1.5]
# interp_pts = [-1,0,1]

# deg = 2
# spatial_pts = [0.5,0.7,1.5]
# interp_pts = [-1,-0.6,1]

deg = 2
spatial_pts = [0.5,0.7,1.5]
interp_pts = [-1,0,1]

PlotXMap(deg,spatial_pts,interp_pts, npts=101, contours=False)