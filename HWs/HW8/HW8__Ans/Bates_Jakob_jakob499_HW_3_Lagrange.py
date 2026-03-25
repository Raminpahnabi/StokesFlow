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
    if (p+1) != len(pts):
        sys.exit("The number of input points for interpolating must be the same as one plus the polynomial degree for interpolating")
    
    Npts = len(pts)
    if isinstance(pts,np.ndarray):
       pts.tolist()
       
    if (Npts != len(set(pts))):
        sys.exit("There are duplicate points")
    for i in range(Npts-1):
        if pts[i+1] - pts[i] < 0:
            sys.exit("The points are not ordered from least to greatest")
    
    l_a = 1
    n_en = len(pts)
    for b in range(n_en):
        if b != a:
            l_a = l_a*(xi-pts[b])/(pts[a]-pts[b])
    return l_a

# Plot the Lagrange polynomial basis functions
# COMMENT THIS CODE
def PlotLagrangeBasisFunctions(p,pts,n_samples = 101):
    xis = np.linspace(min(pts),max(pts),n_samples)
    fig, ax = plt.subplots()
    for a in range(0,p+1):
        vals = []
        for xi in xis:
            vals.append(LagrangeBasisEvaluation(p, pts, xi, a))
                
        plt.plot(xis,vals, label=a+1)
    ax.grid(linestyle='--')
    plt.legend()
        
# Interpolate two-dimensional data
def InterpolateFunction(p,pts2D,n_samples = 101):
    pts2D = np.array(pts2D)
    pts = pts2D[:,0]
    coeffs = pts2D[:,1]
    xis = np.linspace(np.min(pts),np.max(pts),n_samples)
    ys = np.zeros(n_samples)
    for i in range(n_samples):
        xi = xis[i]
        for a in range(p+1):
            ys[i] = ys[i] + coeffs[a]*LagrangeBasisEvaluation(p, pts, xi, a)
    
    fig, ax = plt.subplots()    
    plt.plot(xis,ys)
    plt.scatter(pts, coeffs,color='r')
    
# mypts = np.linspace(-1,1,4)
# # mypts = [-1,0,1/2,1]
# myaltpts = [-1,0,.5,1]
# p = 3
# PlotLagrangeBasisFunctions(p,mypts)

# my2Dpts = [[0,1],[4,6],[6,2],[7,11]]
# InterpolateFunction(p,my2Dpts)