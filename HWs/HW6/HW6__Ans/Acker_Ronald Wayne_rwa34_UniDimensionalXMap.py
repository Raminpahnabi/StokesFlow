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
# import <UNCOMMENT THIS LINE AND INPUT YOUR FILENAME HERE>

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

# x(\xi) = \sum_{a=0}^p x_a * N_a(\xi)
# Mapping from parent domain to spatial domain
def XMap(p, spatial_pts, interp_pts, xi):
    x_e = 0.0  # Initialize to 0 instead of 1
    for a in range(p + 1):  # Loop over all the basis functions
        N_a = LagrangeBasisEvaluation(p, interp_pts, xi, a)  # Evaluate the a-th Lagrange basis function
        x_e += spatial_pts[a] * N_a  # Multiply the spatial point with the basis function and accumulate
    return x_e

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
        
        
cases = [
    (1, [0, 1], [-1, 1]),                # p = 1, {xa} = {0, 1}, {ξa} = {-1, 1}
    (2, [0.5, 1, 1.5], [-1, 0, 1]),      # p = 2, {xa} = {0.5, 1, 1.5}, {ξa} = {-1, 0, 1}
    (2, [0.5, 0.7, 1.5], [-1, -0.6, 1]), # p = 2, {xa} = {0.5, 0.7, 1.5}, {ξa} = {-1, -0.6, 1}
    (2, [0.5, 0.7, 1.5], [-1, 0, 1])     # p = 2, {xa} = {0.5, 0.7, 1.5}, {ξa} = {-1, 0, 1}
]
plt.figure(figsize=(10, 8))
for case in cases:
    p, spatial_pts, parent_pts = case
    PlotXMap(p, spatial_pts, parent_pts)

plt.show()