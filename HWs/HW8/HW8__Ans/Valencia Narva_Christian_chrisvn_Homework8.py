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

# Depending on your approach, 
# You may need to import information from HW3 and/or
# HW6 to load in your unidimensional and multidimensional
# Lagrange basis function code 
import Homework6_Lagrange2D as m_basis


# input polynomial degree, p,
#       interpolation points, pts,
#       the point of evaluation, xi,
#   and the basis fnction index, a
def LagrangeBasisParamDervEvaluation(p,pts,xi,a):

    sum=0
    xi_a=pts[a]
    for j in range(len(pts)):
        basis=1
        xi_j=pts[j]

        if xi_a==xi_j:
            pass
        else:
            for b in range(len(pts)):
                xi_b=pts[b]
                if xi_a==xi_b or xi_j==xi_b:
                    pass
                else:
                    basis=basis*(xi-xi_b)/(xi_a-xi_b)
    
            sum=sum+1/(xi_a-xi_j)*basis

    return sum

# you may want to create a function here that 
# converts from a single index, A, and a set of 
# polynomial degrees into a multi-index indicating
# the indices of the univariate lagrange polynomials
# 
# you will need this functionality, though you do
# not need to complete this function for full credit
def GlobalToLocalIdxs(A,degs):
    idxs=np.zeros(len(degs))
    for i in range(len(degs)):
        
        if i==0:            
            idxs[i]=A%(degs[i]+1)
            deg_base=(degs[i]+1)
        else:
            idxs[i]=A//deg_base
            deg_base=deg_base*(degs[i]+1)

    idxs=idxs.astype(int)
    return idxs

# evaluate the partial derivative of a nD lagrange 
# basis function of index A in the "dim" dimension
# (e.g. derivative in xi is 0, in eta is 1)
def LagrangeBasisDervParamMultiD(A,degs,interp_pts,xis,dim):

    idxs=GlobalToLocalIdxs(A,degs)

    partial_derviative=LagrangeBasisParamDervEvaluation(degs[dim],interp_pts[dim],xis[dim],idxs[dim])

    product=1
    for i in range(len(idxs)):
        
        if i==dim:
            pass
        else:
            product=product*m_basis.LagrangeBasisEvaluation(degs[i],interp_pts[i],xis[i],idxs[i])

    partial_derviative=partial_derviative*product

    return partial_derviative

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
                # if you imported your multidimensional
                # lagrange basis function code as
                # module "m_basis", uncomment the line below
                #Z[j,i] = m_basis.MultiDimensionalBasisFunction(A,degs,interp_pts,xi_vals)
                continue
            else:
                Z[j,i] = LagrangeBasisDervParamMultiD(A,degs,interp_pts,xi_vals,dim) #<------------ i,j?
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







## Problem 3
PlotLagrangeBasisDerivatives(2,[-1,0,1],n_samples = 101)

# print(LagrangeBasisDervParamMultiD(0,[2,2],[[-1,0,1],[-1,0,1]],[0,-1],1))
xi_deg=2
eta_deg=2
xi_pts=np.linspace(-1,1,xi_deg+1)
eta_pts=np.linspace(-1,1,eta_deg+1)


#Problem 4
# A=0, dim=0
PlotTwoDBasisFunctionParentDomain(0,[xi_deg,eta_deg],[xi_pts,eta_pts],0,contours=False)
# A=1, dim=0
PlotTwoDBasisFunctionParentDomain(1,[xi_deg,eta_deg],[xi_pts,eta_pts],0,contours=False)
# A=2, dim=0
PlotTwoDBasisFunctionParentDomain(2,[xi_deg,eta_deg],[xi_pts,eta_pts],0,contours=False)
# A=3, dim=0
PlotTwoDBasisFunctionParentDomain(3,[xi_deg,eta_deg],[xi_pts,eta_pts],0,contours=False)
# A=4, dim=0
PlotTwoDBasisFunctionParentDomain(4,[xi_deg,eta_deg],[xi_pts,eta_pts],0,contours=False)
# A=5, dim=0
PlotTwoDBasisFunctionParentDomain(5,[xi_deg,eta_deg],[xi_pts,eta_pts],0,contours=False)
# A=6, dim=0
PlotTwoDBasisFunctionParentDomain(6,[xi_deg,eta_deg],[xi_pts,eta_pts],0,contours=False)
# A=7, dim=0
PlotTwoDBasisFunctionParentDomain(7,[xi_deg,eta_deg],[xi_pts,eta_pts],0,contours=False)
# A=8, dim=0
PlotTwoDBasisFunctionParentDomain(8,[xi_deg,eta_deg],[xi_pts,eta_pts],0,contours=False)

# A=0, dim=1
PlotTwoDBasisFunctionParentDomain(0,[xi_deg,eta_deg],[xi_pts,eta_pts],1,contours=False)
# A=1, dim=1
PlotTwoDBasisFunctionParentDomain(1,[xi_deg,eta_deg],[xi_pts,eta_pts],1,contours=False)
# A=2, dim=1
PlotTwoDBasisFunctionParentDomain(2,[xi_deg,eta_deg],[xi_pts,eta_pts],1,contours=False)
# A=3, dim=1
PlotTwoDBasisFunctionParentDomain(3,[xi_deg,eta_deg],[xi_pts,eta_pts],1,contours=False)
# A=4, dim=1
PlotTwoDBasisFunctionParentDomain(4,[xi_deg,eta_deg],[xi_pts,eta_pts],1,contours=False)
# A=5, dim=1
PlotTwoDBasisFunctionParentDomain(5,[xi_deg,eta_deg],[xi_pts,eta_pts],1,contours=False)
# A=6, dim=1
PlotTwoDBasisFunctionParentDomain(6,[xi_deg,eta_deg],[xi_pts,eta_pts],1,contours=False)
# A=7, dim=1
PlotTwoDBasisFunctionParentDomain(7,[xi_deg,eta_deg],[xi_pts,eta_pts],1,contours=False)
# A=8, dim=1
PlotTwoDBasisFunctionParentDomain(8,[xi_deg,eta_deg],[xi_pts,eta_pts],1,contours=False)