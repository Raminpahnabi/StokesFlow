# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:05:28 2024

@author: 4466n
"""

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

def LagrangeBasisEvaluation(p,pts,xi,a):
    # p is the number order of the polynomial (Polynomial degree)
    # pts is the array of points
    # xi is point to be evaluated
    # a is the a-th basis function...the node
    # ensure valid input
    if (p+1 != len(pts)):
        sys.exit("The number of input points for interpolating must be the same as one plus the polynomial degree for interpolating")
    # complete this code and return an appropriate value
    # as given in Hughes Equation 3.6.1
    l=1
    for i in range(p+1):
        if not i==a:
            l*=(xi-pts[i])/(pts[a]-pts[i])
    return(l)
    
# Plot the Lagrange polynomial basis functions
# COMMENT THIS CODE
def PlotLagrangeBasisFunctions(p,pts,n_samples = 101):
    xis = np.linspace(min(pts),max(pts),n_samples)  #This is creates an "n" long array from our minimum point to our maximum point.  We will fill this to plot our function later
    fig, ax = plt.subplots() #creates a plot
    for a in range(0,p+1): #a for loop to begin our calculation up until our last point
        vals = [] #creates an array called vals to store our values in
       
        for xi in xis: #loop that checks each point in our function
            vals.append(LagrangeBasisEvaluation(p, pts, xi, a)) #solves the our function an an array
                
        plt.plot(xis,vals)  #plots our function
    ax.grid(linestyle='--') #adds dashed axis gridlines
    
# Interpolate two-dimensional data
def InterpolateFunction(p,pts2D,n_samples = 101):
    # Insert appropriate text here, as described
    # in the homework prompt

    pts=[]
    coeffs=[]
    
    for i in range(0,p+1):
        pts.append(pts2D[i][0])
        coeffs.append(pts2D[i][1])
        
    xis=np.linspace(min(pts),max(pts),n_samples)
    ys=np.zeros(len(xis))
    
    for i in range(0,len(xis)):
        xi=xis[i]
        for j in range(0,p+1):
            ys[i]+=(coeffs[j]*LagrangeBasisEvaluation(p, pts, xi, j))
            
    fig, ax = plt.subplots()    
    plt.plot(xis,ys)
    plt.scatter(pts, coeffs,color='r')
    
#mypts = np.linspace(-1,1,4)
#myaltpts = [-1,0,.5,1]
#p = 3
#PlotLagrangeBasisFunctions(p,mypts)
#my2Dpts = [[0,1],[4,6],[6,2],[7,11]]
#InterpolateFunction(p,my2Dpts)

# x(\xi) = \sum_{a=0}^p x_a * N_a(\xi)
def XMap(deg,spatial_pts,interp_pts,xi):
    xe=0
    for i in range(0,deg+1):
        xe+=LagrangeBasisEvaluation(deg,interp_pts,xi, i)*spatial_pts[i]
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
        
        
PlotXMap(1, [-1,0], [0,1],contours=False)
PlotXMap(2,[0.5,1,1.5],[-1,0,1],contours=False)
PlotXMap(2,[0.5,0.7,1.5],[-1,-0.6,1],contours=False)
PlotXMap(2,[0.5,0.7,1.5],[-1,0,1],contours=False)
