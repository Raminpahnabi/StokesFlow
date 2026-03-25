# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:11:44 2024

@author: heidi
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

# input polynomial degree, p,
#       interpolation points, pts,
#       the point of evaluation, xi,
#   and the basis fnction index, a
def LagrangeBasisParamDervEvaluation(p,pts,xi,a):
    sum = 0
    for j in range(0, p+1):       
     
        if j== a: 
            continue
        else:
            basis_func = 1
            # derivative = 0
            for b in range(0, p+1):
                if b!=a and b!=j:
                    xi_a= pts[a]
                    xi_b= pts[b]
                    basis_func *= ((xi-xi_b)/(xi_a-xi_b))

            sum += (1/(pts[a]-pts[j]))*basis_func
                
    return sum

# you may want to create a function here that 
# converts from a single index, A, and a set of 
# polynomial degrees into a multi-index indicating
# the indices of the univariate lagrange polynomials
# 
# you will need this functionality, though you do
# not need to complete this function for full credit
def GlobalToLocalIdxs(A,degs):
    basis_func = degs[0]+1
    a_0 = A % (basis_func)
    a = [a_0]
    # a.append(a_0)
    
    for n in range(1, len(degs)):
        #basis_func = A//(basis_func)
        a.append(A//basis_func)
        basis_func *= (degs[n]+1) #Basis function is the index of the basis function
    return a

# evaluate the partial derivative of a nD lagrange 
# basis function of index A in the "dim" dimension
# (e.g. derivative in xi is 0, in eta is 1)
def LagrangeBasisDervParamMultiD(A,degs,interp_pts,xis,dim):
    a = GlobalToLocalIdxs(A, degs)
    deriv =1
    for i in range(0,len(a)):
        if i!=dim:
            deriv *= LagrangeBasisEvaluation(degs[i], interp_pts[i], xis[i], a[i])
        else:
            deriv *= LagrangeBasisParamDervEvaluation(degs[dim],interp_pts[dim],xis[dim],a[dim])
    
    
    return deriv


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

# p=2
# PlotLagrangeBasisDerivatives(p,np.linspace(-1,1,p+1),n_samples = 101)

x=np.linspace(-1,1,3)
y=np.linspace(-1,1,3)

# for A in range(0,9):
#     PlotTwoDBasisFunctionParentDomain(A, [2,2], [x,y], 1)
