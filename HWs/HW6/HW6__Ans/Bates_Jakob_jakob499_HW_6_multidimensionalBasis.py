# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 09:40:57 2024

@author: jk-ba
"""

import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt

from HW_3_Lagrange import LagrangeBasisEvaluation

# higher-dimensional basis function with multi-index
def MultiDimensionalBasisFunctionIdxs(idxs,degs,interp_pts,xis):
    N = 1
    for i in range(len(idxs)):
        N *= LagrangeBasisEvaluation(degs[i], interp_pts[i], xis[i],idxs[i])
    return N
    
# higher-dimensional basis function with single index
def MultiDimensionalBasisFunction(A,degs,interp_pts,xis):
    nsd = len(degs)
    idxs = []
    for i in range(nsd):
        if i == 0:
            idxs.append(A % (degs[i] + 1))
        else:
            temp = 1
            for j in range(nsd-1):
                temp *= (degs[j] + 1)
            idxs.append(A//temp)
    # print(idxs)
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

        
ps = [1]
    
for A in range(math.prod((p+1) for p in ps)):
    PlotTwoDimensionalParentBasisFunction(A,degs=ps,npts=101,contours=True)
