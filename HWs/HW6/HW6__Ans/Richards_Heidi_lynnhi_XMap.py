# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 00:01:25 2024

@author: heidi
"""

import sys
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt

# IMPORT (or copy) your code from HW3 here
# which evaluated a Lagrane polynomial basis function
# import <UNCOMMENT THIS LINE AND INPUT YOUR FILENAME HERE>

#HW 3
def LagrangeBasisEvaluation(p,pts,xi,a):
    # ensure valid input
    if (p+1 != len(pts)):
        sys.exit("The number of input points for interpolating must be the same as one plus the polynomial degree for interpolating")
    if(a > len(pts)):
        sys.exit("a must be less than or equal to the number of input points")
    list2=[]
    for i in range(0, len(pts)):
        xa= pts[a]
        xb= pts[i]
        if(i!=a):
            number = (xi-xb)/(xa-xb)
            list2.append(number)
    answer = np.prod(list2)
    return answer

    
#End HW 3 

# x(\xi) = \sum_{a=0}^p x_a * N_a(\xi)
def XMap(deg,spatial_pts,interp_pts,xi):
    x=0
    for a in range(0,deg+1):
        x+= LagrangeBasisEvaluation(deg, interp_pts, xi, a)*spatial_pts[a]
        

    # complete this function
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
        
PlotXMap(2, [.5, .7, 1.5], [-1,0,1], npts=101, contours=True)