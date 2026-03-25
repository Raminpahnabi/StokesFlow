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
    if (a > p+1):
        sys.exit("The requested basis function's index is greater than the number of nodes")


    terms = []
    xa = pts[a]
    for i in range(0, len(pts)):
        for j in range(0, len(pts)):
            if (i != j):
                if (pts[i] == pts[j]):
                    sys.exit("The input nodes are not unique")

        xb = pts[i]
        if (i != a):
            term = (xi - xb) / (xa - xb)
            terms.append(term)
    product = np.prod(terms)
    return product


# Given a set of points, pts, and number of samples to
# evaluate, n_samples, plot the Lagrange polynomial basis
# functions of polynomial degree p.
def PlotLagrangeBasisFunctions(p,pts,n_samples = 101):
    # Create an array of n_samples number of even spaced numbers
    #  from the minimum to maximum values in pts. These are all
    #  input values to be plotted. 
    xis = np.linspace(min(pts),max(pts),n_samples)

    # Set up the figure
    fig, ax = plt.subplots()

    # Caculate and plot all basis functions
    for a in range(0,p+1):
        # vals stores input,output values to be plotted for the
        #  a-th basis function
        vals = []

        # For every input value, the ouput is calculated using
        #  LagrangeBasisEvaluation and stored in vals
        for xi in xis:
            vals.append(LagrangeBasisEvaluation(p, pts, xi, a))
                
        # The a-th basis function is plotted
        plt.plot(xis,vals)
    ax.grid(linestyle='--')
    plt.show()
        
# Interpolate two-dimensional data
def InterpolateFunction(p,pts2D,n_samples = 101):
    pts = []
    coeffs = []
    for i in range(0, len(pts2D)):
        pts.append(pts2D[i][0])
        coeffs.append(pts2D[i][1])
    xis = np.linspace(min(pts),max(pts),n_samples)
    ys = []

    for i in range(0,len(xis)):
        xi = xis[i]
        vals = []
        for a in range(0,p+1):
            val = LagrangeBasisEvaluation(p, pts, xi, a)
            vals.append(val*coeffs[a])
        ys.append(np.sum(vals))

    fig, ax = plt.subplots()    
    plt.plot(xis,ys)
    plt.scatter(pts, coeffs,color='r')
    plt.show()

def Homework2():
    p1 = 1
    pts1 = [-1,1]
    p2 = 3
    pts2 = [-1,- 1/3 , 1/3 ,1]
    p3 = 3
    pts3 = [-1,0, 1/2 ,1]

    PlotLagrangeBasisFunctions(p1,pts1)
    PlotLagrangeBasisFunctions(p2,pts2)
    PlotLagrangeBasisFunctions(p3,pts3)


    p = 3
    my2Dpts = [[0,1],[4,6],[6,2],[7,11]]
    InterpolateFunction(p,my2Dpts)