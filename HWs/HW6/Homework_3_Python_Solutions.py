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
        
    numerator = 1
    denominator = 1
    for i in range(0,p+1):
        # skip the point where i equals a
        if i == a:
            continue
        else:
            numerator *= (xi-pts[i])
            denominator *= (pts[a] - pts[i])
    
    return numerator / denominator

# Plot the Lagrange polynomial basis functions
def PlotLagrangeBasisFunctions(p,pts,n_samples = 101):
    xis = np.linspace(min(pts),max(pts),n_samples)
    fig, ax = plt.subplots()
    for a in range(0,p+1):
        vals = []
        for xi in xis:
            vals.append(LagrangeBasisEvaluation(p, pts, xi, a))
                
        plt.plot(xis,vals)
    ax.grid(linestyle='--')
        
# Interpolate two-dimensional data
def InterpolateFunction(p,pts2D,n_samples = 101):
    pts = [x[0] for x in pts2D]
    coeffs = [x[1] for x in pts2D]
    xis = np.linspace(min(pts),max(pts),n_samples)
    ys = np.zeros(n_samples)
    for i in range(0,len(ys)):
        xi = xis[i]
        for a in range(0,p+1):
            ys[i] += coeffs[a] * LagrangeBasisEvaluation(p, pts, xi, a)
    
    fig, ax = plt.subplots()    
    plt.plot(xis,ys)
    plt.scatter(pts, coeffs,color='r')

def HomeworkProblem1():
    mypts = np.linspace(-1,1,4)
    myaltpts = [-1,0,.5,1]
    p = 3
    PlotLagrangeBasisFunctions(p,mypts)

def HomeworkProblem3():
    p = 3
    my2Dpts = [[0,1],[4,6],[6,2],[7,11]]
    InterpolateFunction(p,my2Dpts)