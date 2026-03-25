# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:21:18 2024

@author: 4466n
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:45:50 2024

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
    if (p+1 != len(pts)):
        sys.exit("The number of input points for interpolating must be the same as one plus the polynomial degree for interpolating")
    l=1
    for i in range(0,p+1):
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

# higher-dimensional basis function with multi-index
def MultiDimensionalBasisFunctionIdxs(idxs,degs,interp_pts,xis):
    solution =1
    for i in range(0,len(idxs)):
        solution*=LagrangeBasisEvaluation(degs[i], interp_pts[i], xis[i], idxs[i])
    return solution
    
# higher-dimensional basis function with single index
def MultiDimensionalBasisFunction(A,degs,interp_pts,xis):
    bfs=degs[0]+1
    a=A%(bfs)
    idxs=[a]
    for i in range(1,len(degs)):
        idxs.append(A//(bfs))
        bfs*=(degs[i]+1)
    return MultiDimensionalBasisFunctionIdxs(idxs, degs, interp_pts, xis)

# plot of 2D basis functions with A a single index
def PlotTwoDimensionalParentBasisFunction(A,degs,npts = 101,contours = True):
    interp_pts = [np.linspace(-1,1,degs[i]+1) for i in range(0,len(degs))]
    xivals = np.linspace(-1,1,npts)
    etavals = np.linspace(-1,1,npts)
    
    Xi,Eta = np.meshgrid(xivals,etavals)
    Z = np.zeros(Xi.shape)
    
    for i in range(0,len(xivals)):
        for j in range(0,len(etavals)):
            Z[i,j] = MultiDimensionalBasisFunction(A, degs, interp_pts, [xivals[i],etavals[j]])
    
    # contour plot
    if contours:
        fig, ax = plt.subplots()
        surf = ax.contourf(Eta,Xi,Z,levels=100,cmap=matplotlib.cm.jet)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        fig.colorbar(surf)
        plt.show()
    # 3D surface plot
    else:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(Eta, Xi, Z, cmap=matplotlib.cm.jet,
                       linewidth=0, antialiased=False)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        ax.set_zlabel(r"$N(\xi,\eta)$")
        plt.show()
        
#for i in range(0,4):
#    PlotTwoDimensionalParentBasisFunction(i, [1,1])

#for i in range(0,6):
#     PlotTwoDimensionalParentBasisFunction(i, [2,1])

#for i in range(0,9):
#     PlotTwoDimensionalParentBasisFunction(i, [2,2])

for i in range(0,16):
     PlotTwoDimensionalParentBasisFunction(i, [3,3])