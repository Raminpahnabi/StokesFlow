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

from LagrangeBasisFunctions import LagrangeBasisEvaluation

# x(\xi) = \sum_{a=0}^p x_a * N_a(\xi)
def XMap(deg,spatial_pts,interp_pts,xi):
    p = deg
    x = 0
    for i in range(p+1):
        x_a = spatial_pts[i]
        a = i
        N_a = LagrangeBasisEvaluation(p,interp_pts,xi,a)
        x += x_a * N_a
    return x

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

#Plot 1
p = 1
x = [0,1]
eps = [-1,1]
PlotXMap(p,x,eps,contours=False)

#Plot 2
p = 2
x = [0.5,1,1.5]
eps = [-1,0,1]
PlotXMap(p,x,eps,contours=False)

#Plot 3
p = 2
x = [0.5,0.7,1.5]
eps = [-1,-0.6,1]
PlotXMap(p,x,eps,contours=False)

#Plot 4
p = 2
x = [0.5,0.7,1.5]
eps = [-1,0,1]
PlotXMap(p,x,eps,contours=False)