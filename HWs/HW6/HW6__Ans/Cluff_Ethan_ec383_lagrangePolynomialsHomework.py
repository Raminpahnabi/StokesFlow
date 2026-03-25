#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 06:48:53 2024

@author: kendrickshepherd
"""

import sys
import numpy as np
from matplotlib import pyplot as plt

# Given a set of points, pts, to interpolate
# and a polynomial degree, this function will
# evaluate the a-th basis function at the 
# location xi
def LagrangeBasisEvaluation(p,pts,xi,a):
    # ensure valid input
    if (p+1 != len(pts)):
        sys.exit("The number of input points for interpolating must be the same as one plus the polynomial degree for interpolating")

    # complete this code and return an appropriate value
    # as given in Hughes Equation 3.6.1
    numer = 1
    denom = 1

    for i in range(p+1):
        if i != a:
            numer *= (xi - pts[i])
            denom *= (pts[a] - pts[i])

    result = numer / denom

    return result


# Plot the Lagrange polynomial basis functions
# COMMENT THIS CODE
def PlotLagrangeBasisFunctions(p,pts,n_samples = 101):
    # Create a linear interpolation between the minimum and maximum value
    xis = np.linspace(min(pts),max(pts),n_samples)

    # Create a figure to plot the resulting polynomial on
    fig, ax = plt.subplots()

    # Iterate through the polynomial degrees
    for a in range(0,p+1):
        vals = []

        # Evaluate the Lagrange basis function at the point with the given x-coordinate
        for xi in xis:
            vals.append(LagrangeBasisEvaluation(p, pts, xi, a))

        # Plot the resulting Lagrange basis function and show the result 
        plt.plot(xis,vals)
        plt.xlabel('Point')
        plt.ylabel('Basis Function Value')
        plt.title('Lagrange Basis Function')
        plt.show()

    # Figure formatting
    ax.grid(linestyle='--')
        
# Interpolate two-dimensional data
def InterpolateFunction(p,pts2D,n_samples = 101):
    # Insert appropriate text here, as described
    # in the homework prompt
    
    # Extract the points and coefficients
    pts = [x[0] for x in pts2D]
    coeffs = [x[1] for x in pts2D]

    # Create containers for the x and y values of the polynomial graph
    xis = np.linspace(min(pts), max(pts), n_samples)
    ys = np.zeros(np.shape(xis))

    # Generate the polynomial graph
    for i in range(len(xis)):
        xi = xis[i]
        for a in range(0, p+1):
            # Get the basis function evaluation and scale it by the y-coordinate
            val = LagrangeBasisEvaluation(p, pts, xi, a)
            val *= coeffs[a]
            ys[i] += val

    # Plot stuff
    fig, ax = plt.subplots()    
    plt.plot(xis,ys)
    plt.scatter(pts, coeffs,color='r')
    plt.xlabel('Zeta')
    plt.ylabel('Value')
    plt.title('Lagrange Polynomial Interpolation of Points')
    plt.show()

# p=3

# mypts = np.linspace(-1, 1, 4)
# PlotLagrangeBasisFunctions(p,mypts)

# my2Dpts = [[0,1],[4,6],[6,2],[7,11]]
# InterpolateFunction(p,my2Dpts)