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
        bfs*=(degs[0]+1)
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

# input polynomial degree, p,
#       interpolation points, pts,
#       the point of evaluation, xi,
#       and the basis fnction index, a
def LagrangeBasisParamDervEvaluation(p,pts,xi,a):
    dl=0
    for j in range(p+1):
        if j!=a:
            l=1
            for b in range(p+1):
                if b!=a and b!=j:
                    l*=(xi-pts[b])/(pts[a]-pts[b])
            dl+=(1/(pts[a]-pts[j]))*l
    return dl

# you may want to create a function here that converts from a single index, A, and a set of polynomial degrees into a multi-index indicating
# the indices of the univariate lagrange polynomials you will need this functionality, though you do not need to complete this function for full credit
def GlobalToLocalIdxs(A,degs):
    bfs=degs+1
    a=A%(bfs)
    return a

# evaluate the partial derivative of a nD lagrange basis function of index A in the "dim" dimension (e.g. derivative in xi is 0, in eta is 1)
def LagrangeBasisDervParamMultiD(A,degs,interp_pts,xis,dim):
    a=GlobalToLocalIdxs(A,degs[dim])
    dNa=LagrangeBasisParamDervEvaluation(degs[dim],interp_pts[dim],xis[dim],a)
    dNa*=MultiDimensionalBasisFunction(A,degs,interp_pts,xis)
    return dNa

# Plot the Lagrange polynomial basis functionderivatives
def PlotLagrangeBasisDerivatives(p,pts,n_samples = 101):
    xis = np.linspace(min(pts),max(pts),n_samples)
    fig, ax = plt.subplots()
    for a in range(0,p+1):
        vals = []
        for xi in xis:
            vals.append(LagrangeBasisParamDervEvaluation(p, pts, xi, a))
        plt.plot(xis,vals)
    ax.grid(linestyle='--')

# plot a basis function defined on a parent domain; this is similar to what was in a previous homework, but slightly generalized                
def PlotTwoDBasisFunctionParentDomain(A,degs,interp_pts,dim,npts=21,contours=False):
    xivals = np.linspace(interp_pts[0][0],interp_pts[0][-1],npts+1)
    etavals = np.linspace(interp_pts[1][0],interp_pts[1][-1],npts)
    
    Xi,Eta = np.meshgrid(xivals,etavals)
    Z = np.zeros(Xi.shape)
    
    for i in range(0,len(xivals)):
        for j in range(0,len(etavals)):
            xi_vals = [xivals[i],etavals[j]]
            if dim < 0:
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
        plt.title("2D Basis Function dim=%d A=%s" % (dim,A))
        plt.show()

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
        plt.title("2D Basis Function dim=%d A=%s" % (dim,A)) 
        plt.show()

degsx = 2
degsy = 2
xspit = np.linspace(-1,1,degsx+1)  
yspit = np.linspace(-1,1,degsy+1)
degs = [degsx,degsy]
interp_pt = [xspit,yspit]


for A in range(9):
    PlotTwoDBasisFunctionParentDomain(A, degs, interp_pt, 0)
    PlotTwoDBasisFunctionParentDomain(A, degs, interp_pt, 1) 
    #PlotTwoDBasisFunctionParentDomain(A, degs, interp_pts, dim)
