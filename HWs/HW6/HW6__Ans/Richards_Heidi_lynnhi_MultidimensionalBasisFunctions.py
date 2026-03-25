# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:45:04 2024

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

# higher-dimensional basis function with multi-index
def MultiDimensionalBasisFunctionIdxs(idxs,degs,interp_pts,xis):
    N_a=1
    for i in range(0,len(idxs)):
        N_a *= LagrangeBasisEvaluation(degs[i], interp_pts[i], xis[i], idxs[i])
    return N_a
    
# higher-dimensional basis function with single index
def MultiDimensionalBasisFunction(A,degs,interp_pts,xis):
    basis_func = degs[0]+1
    a_0 = A % (basis_func)
    a = [a_0]
    # a.append(a_0)
    
    for n in range(1, len(degs)):
        #basis_func = A//(basis_func)
        a.append(A//basis_func)
        basis_func *= (degs[n]+1) #Basis function is the index of the basis function
    
    return MultiDimensionalBasisFunctionIdxs(a, degs,interp_pts,xis)
    
    
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
        

        

for i in range(4):
    PlotTwoDimensionalParentBasisFunction(i, [1,1],contours=True)
  