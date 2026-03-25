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

sys.path.append('../HW3/')

import Homework_3_Python_Solutions as HW3


def MultiDimensionalBasisFunctionIdxs(idxs,degs,interp_pts,xis):
    val = 1
    for i in range(0,len(idxs)):
        val *= HW3.LagrangeBasisEvaluation(degs[i],interp_pts[i],xis[i],idxs[i])
    return val

def MultiDimensionalBasisFunction(A,degs,interp_pts,xis):
    bfs = degs[0]+1 
    horiz = A % (bfs) # First line
    idxs = [horiz] # First value
    for i in range(1,len(degs)):
        idxs.append(A // bfs) # Doing the second line and ...
        bfs *= (degs[i]+1)
    return MultiDimensionalBasisFunctionIdxs(idxs,degs,interp_pts,xis)

def MultiDimensionalParentBasisFunction(A,degs,xis):
    interp_pts = []
    for i in range(0,len(degs)):
        interp_pts.append(np.linspace(-1,1,degs[i]+1))
    return MultiDimensionalBasisFunction(A, degs, interp_pts, xis)

def PlotTwoDimensionalParentBasisFunction(A,degs,npts = 101,contours = True):
    interp_pts = [np.linspace(-1,1,degs[i]+1) for i in range(0,len(degs))]
    xivals = np.linspace(-1,1,npts)
    etavals = np.linspace(-1,1,npts)
    
    Xi,Eta = np.meshgrid(xivals,etavals)
    Z = np.zeros(Xi.shape)
    
    for i in range(0,len(xivals)):
        for j in range(0,len(etavals)):
            Z[i,j] = MultiDimensionalBasisFunction(A, degs, interp_pts, [xivals[i],etavals[j]])
    
    if contours:
        fig, ax = plt.subplots()
        surf = ax.contourf(Eta,Xi,Z,levels=100,cmap=matplotlib.cm.jet)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        fig.colorbar(surf)
        plt.show()
    else:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(Eta, Xi, Z, cmap=matplotlib.cm.jet,
                       linewidth=0, antialiased=False)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        ax.set_zlabel(r"$N(\xi,\eta)$")
        plt.show()
        
# degs = [3,3]
# nbfs = math.prod([p+1 for p in degs])

# for i in range(0,nbfs):
#     PlotTwoDimensionalParentBasisFunction(i, degs, contours=True)
    
# print(MultiDimensionalBasisFunctionIdxs([2,2], degs, [[-1,0,1],[-1,0,1]], [-0.5,-0.5]))