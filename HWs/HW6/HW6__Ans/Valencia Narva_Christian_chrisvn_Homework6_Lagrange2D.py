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

# IMPORT (or copy) your code from HW3 here
# which evaluated a Lagrane polynomial basis function
# import <UNCOMMENT THIS LINE AND INPUT YOUR FILENAME HERE>

def LagrangeBasisEvaluation(p,pts,xi,a):
    # ensure valid input
    if (p+1 != len(pts)):
        sys.exit("The number of input points for interpolating must be the same as one plus the polynomial degree for interpolating")

    a_basis=1
    zeta_a=pts[a]
    for zeta_b in pts:
        if zeta_b==zeta_a:
            pass # a term is omitted
        else:
            a_basis=a_basis*(xi-zeta_b)/(zeta_a-zeta_b) #Multiplication to get the a basis
       
    
    return a_basis ##a_th basis function at the location xi


# higher-dimensional basis function with multi-index
def MultiDimensionalBasisFunctionIdxs(idxs,degs,interp_pts,xis):
    basis_start=1
    
    for i in range(len(idxs)):
        basis_start=basis_start*LagrangeBasisEvaluation(degs[i],interp_pts[i],xis[i],idxs[i])
        
    return basis_start
    
# higher-dimensional basis function with single index
def MultiDimensionalBasisFunction(A,degs,interp_pts,xis):

    idxs=np.zeros(len(degs))
    for i in range(len(degs)):
        
        if i==0:            
            idxs[i]=A%(degs[i]+1)
            deg_base=(degs[i]+1)
        else:
            idxs[i]=A//deg_base
            deg_base=deg_base*(degs[i]+1)

    idxs=idxs.astype(int)
    
    basis_start=MultiDimensionalBasisFunctionIdxs(idxs,degs,interp_pts,xis)

    
    return basis_start
    
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
        



def startGraphing(pol_deg):
    A=1
    for a in pol_deg:
        A=A*(a+1)
    print(f"A is {A-1}")
    for i in range(A):
        print(f"Graph for A = {i}")
        PlotTwoDimensionalParentBasisFunction(i,pol_deg)


############################ PROBLEM 1 ############################
print("GRAPH FOR POLYNOMIALS OF DEGREE[1,1]")
pol1=[1,1]
startGraphing(pol1)
print("GRAPH FOR POLYNOMIALS OF DEGREE[2,1]")
pol2=[2,1]
startGraphing(pol2)
print("GRAPH FOR POLYNOMIALS OF DEGREE[2,2]")
pol3=[2,2]
startGraphing(pol3)
print("GRAPH FOR POLYNOMIALS OF DEGREE[3,3]")
pol4=[3,3]
startGraphing(pol4)
####################################################################