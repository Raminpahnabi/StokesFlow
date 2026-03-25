# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 09:24:19 2024

@author: hayde
"""
import sys
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt
import MEEN507_HW3_LagrangePolynomials as HW3
import MEEN507_HW6_Mulitdimension as HW6

#%%
# Depending on your approach, 
# You may need to import information from HW3 and/or
# HW6 to load in your unidimensional and multidimensional
# Lagrange basis function code 

# input polynomial degree, p,
#       interpolation points, pts,
#       the point of evaluation, xi,
#   and the basis fnction index, a
def LagrangeBasisParamDervEvaluation(p,pts,xi,a):
    derivative = 0.0 # intitialize sum for the derivative
    
    for i in range(0, p + 1): # itrrate through points but not a
        if i == a:
            continue
        else:
            points = pts[0:i] + pts[i + 1:] # make a new list with out j
            newA = a
            if i <= a:
                newA = a - 1
            term = 1 / (pts[a] - pts[i])  # compute the division term 
            # Compute the product
            product = HW3.LagrangeBasisEvaluation(p - 1, points, xi, newA)
            derivative += term * product # calculate the derivative (provided equation)
    
    return derivative

#%%

# you may want to create a function here that 
# converts from a single index, A, and a set of 
# polynomial degrees into a multi-index indicating
# the indices of the univariate lagrange polynomials
# 
# you will need this functionality, though you do
# not need to complete this function for full credit
def GlobalToLocalIdxs(A, degs):
    bfs = degs[0] + 1
    horiz = A % bfs
    idxs = [horiz]
    
    for i in range(1, len(degs)):
        idxs.append(A//bfs) #integer devide
        bfs *= (degs[i] + 1)
    return idxs

#%% 

# evaluate the partial derivative of a nD lagrange 
# basis function of index A in the "dim" dimension
# (e.g. derivative in xi is 0, in eta is 1)
def LagrangeBasisDervParamMultiD(A,degs,interp_pts,xis,dim):
    idxs = GlobalToLocalIdxs(A, degs)
    product = 1
    for i in range(len(degs)):
        
        p_i = degs[i]
        pts_i = interp_pts[i]
        
        if i == dim:
            product *= LagrangeBasisParamDervEvaluation(p_i, pts_i, xis[i], idxs[i])
        else:
            product *= HW3.LagrangeBasisEvaluation(p_i, pts_i, xis[i], idxs[i])
            
    return product

#%%

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


#%%

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
        
#%%

#PROBLEM 3 and 4
p = 2
degs = [p, p]
list1 = list(np.linspace(-1, 1, p + 1))
list2 = list(np.linspace(-1, 2, p + 1))
PlotLagrangeBasisDerivatives(p, list1)

for A in range((p+1)*(p+1)):
    plt.title("A = " + A)
    PlotTwoDBasisFunctionParentDomain(A, degs, [list1, list2], 1)