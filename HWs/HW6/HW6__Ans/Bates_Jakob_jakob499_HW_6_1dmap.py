# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:49:43 2024

@author: jk-ba
"""

import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt
from HW_3_Lagrange import LagrangeBasisEvaluation

# x(\xi) = \sum_{a=0}^p x_a * N_a(\xi)
def XMap(deg,spatial_pts,interp_pts,xi):
    x = 0
    nbf = deg+1
    for i in range(nbf):
        x += spatial_pts[i]*LagrangeBasisEvaluation(deg, interp_pts, xi, i)
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
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$x(\xi)$')
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
        
p = 1
xas = [0,1]
xis = [-1,1]
PlotXMap(deg=p,spatial_pts=xas,interp_pts=xis,contours=False)
p = 2
xas = [0.5,1,1.5]
xis = [-1,0,1]
PlotXMap(deg=p,spatial_pts=xas,interp_pts=xis,contours=False)
p = 2
xas = [0.5,0.7,1.5]
xis = [-1,-0.6,1]
PlotXMap(deg=p,spatial_pts=xas,interp_pts=xis,contours=False)
p = 2
xas = [0.5,0.7,1.5]
xis = [-1,0,1]
PlotXMap(deg=p,spatial_pts=xas,interp_pts=xis,contours=False)