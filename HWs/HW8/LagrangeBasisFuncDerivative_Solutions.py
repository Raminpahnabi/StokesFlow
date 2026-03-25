#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:06:03 2024

@author: kendrickshepherd
"""

import sys
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt

sys.path.append('../HW3/')
sys.path.append('../HW6/')

import Homework_3_Python_Solutions as u_basis
import MultiDimensionalBasisFunctions_Solutions as m_basis

def LagrangeBasisParamDervEvaluation(p,pts,xi,a):
    summation = 0
    for j in range(0,p+1):
        if j == a:
            continue
        else:
            part_pts = np.array(list(pts[0:j])+list(pts[j+1:]))
            new_a = a if j > a else a-1
            part_lagrange = u_basis.LagrangeBasisEvaluation(p-1, part_pts, xi, new_a)
            summation += 1/(pts[a]-pts[j]) * part_lagrange
    
    return summation

def GlobalToLocalIdxs(A,degs):
    bfs = degs[0]+1
    horiz = A % (bfs)
    idxs = [horiz]
    for i in range(1,len(degs)):
        idxs.append(A // bfs)
        bfs *= (degs[i]+1)

    return idxs


# evaluate the partial derivative of a nD lagrange 
# basis function of index A in the "dim" dimension
# (e.g. in xi is 0, in eta is 1)
def LagrangeBasisDervParamMultiD(A,degs,interp_pts,xis,dim):
    idxs = GlobalToLocalIdxs(A,degs)
    
    val = 1
    for i in range(0,len(degs)):
        if i == dim:
            val *= LagrangeBasisParamDervEvaluation(degs[i], interp_pts[i], xis[i], idxs[i])
        else:
            val *= u_basis.LagrangeBasisEvaluation(degs[i], interp_pts[i], xis[i], idxs[i])

    return val

# Plot the Lagrange polynomial basis function
# derivatives
def PlotLagrangeBasisDerivatives(p,pts,n_samples = 101):
    xis = np.linspace(min(pts),max(pts),n_samples)
    fig, ax = plt.subplots()
    for a in range(0,p+1):
        vals = []
        for xi in xis:
            vals.append(LagrangeBasisParamDervEvaluation(p, pts, xi, a))
                
        plt.plot(xis,vals)
    ax.grid(linestyle='--')

# plot a basis function defined on a parent
# domain; this is similar to what was
# in a previous homework, but slightly generalized                
def PlotTwoDBasisFunctionParentDomain(A,degs,interp_pts,dim,npts=21,contours=False):
    xivals = np.linspace(interp_pts[0][0],interp_pts[0][-1],npts+1)
    etavals = np.linspace(interp_pts[1][0],interp_pts[1][-1],npts)
    
    Xi,Eta = np.meshgrid(xivals,etavals)
    Z = np.zeros(Xi.shape)
    
    for i in range(0,len(xivals)):
        for j in range(0,len(etavals)):
            xi_vals = [xivals[i],etavals[j]]
            if dim < 0:
                Z[j,i] = m_basis.MultiDimensionalBasisFunction(A,degs,interp_pts,xi_vals)
            else:
                Z[j,i] = LagrangeBasisDervParamMultiD(A,degs,interp_pts,xi_vals,dim)

    if contours:
        fig, ax = plt.subplots()
        surf = ax.contourf(Xi,Eta,Z,levels=100,cmap=matplotlib.cm.jet)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        fig.colorbar(surf)
        plt.show()
    else:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(Xi, Eta, Z, cmap=matplotlib.cm.jet,
                       linewidth=0, antialiased=False)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        ax.set_zlabel(r"$\frac{\partial N_A}{\partial xi_%d}$" % dim)
        plt.show()



# p = 2
# dim = 1
# mycontours = False
# interp_pts = list(np.linspace(-1, 1, p+1))
# interp_pts2 = list(np.linspace(-1, 2, p+1))
# # PlotLagrangeBasisDerivatives(p, interp_pts)

# for A in range(0,(p+1)**2):
#     PlotTwoDBasisFunctionParentDomain(A,[p,p],[interp_pts,interp_pts2],dim,contours=mycontours)