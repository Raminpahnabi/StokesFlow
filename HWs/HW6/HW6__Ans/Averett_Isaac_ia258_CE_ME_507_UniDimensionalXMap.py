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

def LagrangeBasisEvaluation(p, pts, xi, a):
    # ensure valid input
    if (p+1 != len(pts)):
        sys.exit("The number of input points for interpolating must be the same as one plus the polynomial degree for interpolating")
    else:
        value = 1
        for m in range(0, p + 1):
            if m != a:
                value *= (xi - pts[m]) / (pts[a] - pts[m])
        return value

# x(\xi) = \sum_{a=0}^p x_a * N_a(\xi)
def XMap(deg,spatial_pts,interp_pts,xi):
    # Initialize the result for x^e(xi)
    x_e_xi = 0.0

    # Sum over all basis functions and spatial points
    for a in range(deg + 1):
        N_a_xi = LagrangeBasisEvaluation(deg, interp_pts, xi, a)  # N_a(xi)
        x_e_xi += spatial_pts[a] * N_a_xi  # Sum x_a * N_a(xi)

    return x_e_xi

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

# Example test case 1:
deg = 1
spatial_pts = [0, 1]
interp_pts = [-1, 1]

PlotXMap(deg, spatial_pts, interp_pts)
PlotXMap(deg, spatial_pts, interp_pts,contours=False)

# Example test case 2:
deg = 2
spatial_pts = [0.5, 1, 1.5]
interp_pts = [-1, 0, 1]

PlotXMap(deg, spatial_pts, interp_pts)
PlotXMap(deg, spatial_pts, interp_pts,contours=False)

# Example test case 3:
deg = 2
spatial_pts = [0.5, .7, 1.5]
interp_pts = [-1, -.6, 1]

PlotXMap(deg, spatial_pts, interp_pts)
PlotXMap(deg, spatial_pts, interp_pts,contours=False)

# Example test case 4:
deg = 2
spatial_pts = [0.5, .7, 1.5]
interp_pts = [-1, 0, 1]

PlotXMap(deg, spatial_pts, interp_pts)
PlotXMap(deg, spatial_pts, interp_pts,contours=False)

