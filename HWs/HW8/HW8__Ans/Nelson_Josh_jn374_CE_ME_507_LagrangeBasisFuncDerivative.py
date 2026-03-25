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

import MultiDimensionalBasisFunctions as Multi
import UniLangrange as Uni

# input polynomial degree, p,
#       interpolation points, pts,
#       the point of evaluation, xi,
#   and the basis fnction index, a

# def LagrangeBasisParamDervEvaluation(p,pts,xi,a):
#     uniBasis = 1
#     deri = 0
#     newcpt = []
#     for j in range(0,p+1):
#         if j == a:
#             continue
#         else:
#         # for b in range(0,p+1):
#             newcpt = [:a] + [a+1:]
#             newcpt = [:j] + [j+1:]
#             # if b != a and b != j:
#             uniBasis *=  (xi-pts[newcpt]) / (pts[a]-pts[newcpt])   
                
#             # if j != a:
#             deri += (1/ (pts[a]-pts[j])) * uniBasis
#         # output = deri * uniBasis      
#     return deri

def LagrangeBasisParamDervEvaluation(p,pts,xi,a):
    deri = 0
    for j in range(0,p+1):
        if j == a:
            continue
        else: 
            if j > a:
                ax = a
            elif j <= a:
                ax = a-1

            newcpt = pts[0:j] + pts[j+1:]
            uni = Uni.LagrangeBasisEvaluation(p-1, newcpt, xi, ax)
            
            deri += (1/ (pts[a]-pts[j])) * uni
    
    return deri

# you may want to create a function here that 
# converts from a single index, A, and a set of 
# polynomial degrees into a multi-index indicating
# the indices of the univariate lagrange polynomials
# 
# you will need this functionality, though you do
# not need to complete this function for full credit
def GlobalToLocalIdxs(A,degs):
    multi_index = []
    horizShape = degs[0] + 1
    a = A % horizShape
    index = [a]
    for i in range(1,len(degs)):
        index.append(A // horizShape)
        horizShape *= (degs[i] + 1) 
    
    return index

# evaluate the partial derivative of a nD lagrange 
# basis function of index A in the "dim" dimension
# (e.g. derivative in xi is 0, in eta is 1)
def LagrangeBasisDervParamMultiD(A,degs,interp_pts,xis,dim):
    multi_index = GlobalToLocalIdxs(A,degs)
    
    derivative_value = 1
    for i in range(len(degs)):
        a_i = degs[i]
        
        if i == dim:
            derivative_value *= LagrangeBasisParamDervEvaluation(degs[i], interp_pts[i], xis[i], multi_index[i])
        else:
            derivative_value *= Uni.LagrangeBasisEvaluation(degs[i], interp_pts[i], xis[i], multi_index[i])
    
    
    return derivative_value

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
    return

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
                # if you imported your multidimensional
                # lagrange basis function code as
                # module "m_basis", uncomment the line below
                Z[j,i] = Multi.MultiDimensionalBasisFunction(A,degs,interp_pts,xi_vals)
                continue
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

    return

# PlotLagrangeBasisDerivatives(2, [-1,0,1]
for i in range(0,9):
    PlotTwoDBasisFunctionParentDomain(i, [2,2], [[-1,0,1],[-1,0,1]], 1)