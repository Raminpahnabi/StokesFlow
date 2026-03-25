#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:06:03 2024

@author: kendrickshepherd
"""
import os
import sys
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt

# sys.path.append('HW6/')
# sys.path.append('HW8/')
sys.path.append(os.path.join(os.getcwd(), 'HW6/'))
sys.path.append(os.path.join(os.getcwd(), 'HW8/'))

# import HW6_MultiDimensionalBasisFunctions_Solutions as basis
import CE_ME_507_MultiDimensionalBasisFunctions_Solutions as basis
# import HW8_LagrangeBasisFuncDerivative_Solutions as derv
import LagrangeBasisFuncDerivative_Solutions as derv



# This is a class that describes a Lagrange basis
# in two dimensions
class LagrangeBasis2D:
    
    # initializor
    def __init__(self,degx,degy,interp_pts_x,interp_pts_y):
        self.degs = [degx,degy]
        self.interp_pts = [interp_pts_x,interp_pts_y]
        
    # the number of basis functions is the 
    # product of basis functions in the x (xi)
    # and y (eta) directions
    def NBasisFuncs(self):
        nbf = 1
        for deg in self.degs:
            nbf *= (deg+1)
        return nbf

    # basis function evaluation code from 
    # previous homework assignment
    def EvalBasisFunction(self,A,xis):
        return basis.MultiDimensionalBasisFunction(A,self.degs,self.interp_pts,xis)
        
    # derivative of basis function code
    # from previous homework
    def EvalBasisDerivative(self,A,xis,dim):
        return derv.LagrangeBasisDervParamMultiD(A,self.degs,self.interp_pts,xis,dim)

    # Evaluate a sum of basis functions times 
    # coefficients on the parent domain
    def EvaluateFunctionParentDomain(self, d_coeffs, xi_vals):
        val = 0
        for a in range(0,self.NBasisFuncs()):
            val += d_coeffs[a] * self.EvalBasisFunction(a,xi_vals)
        return val
    
    # Evaluate the spatial mapping from xi and eta
    # into x and y coordinates
    def EvaluateSpatialMapping(self, x_pts, xi_vals):
        dim = len(x_pts[0])
        pt = np.zeros(dim)
        for a in range(0,self.NBasisFuncs()):
            Na = self.EvalBasisFunction(a,xi_vals)
            x_pt = x_pts[a]
            for j in range(0,dim):
                pt[j] += x_pt[j] * Na
                
        return pt
    
    # Evaluate the Deformation Gradient (i.e.
    # the Jacobian matrix)
    def EvaluateDeformationGradient(self, x_pts, xi_vals):
        n_spat_dim = len(x_pts[0])
        n_param_dim = 2
        def_grad = np.zeros((n_param_dim,n_spat_dim))
        for j in range(0,n_param_dim):
            for A in range(0,self.NBasisFuncs()):
                Na_derv = self.EvalBasisDerivative(A,xi_vals,j)
                for i in range(0,n_spat_dim):
                    def_grad[i,j] += x_pts[A][i] * Na_derv
        
        return def_grad
    
    # Evaluate the jacobian (or the determinant
    # of the deformation gradient)
    def EvaluateJacobian(self, x_pts, xi_vals):
        def_grad = self.EvaluateDeformationGradient(x_pts, xi_vals)
        n_param_dim = def_grad.shape[1]
        n_spat_dim = def_grad.shape[0]
        if n_param_dim != n_spat_dim:
            sys.exit("Cannot yet evalutate the Jacobian when differing spatial and parametric dimensions")
        else:
            return np.linalg.det(def_grad)

    # Evaluate the parametric gradient of a basis
    # function
    def EvaluateBasisParametricGradient(self,A, xi_vals):
        pgrad = np.zeros(2)
        for i in range(0,2):
            pgrad[i] = self.EvalBasisDerivative(A, xi_vals, i)
            
        return pgrad

    # Evaluate the parametric gradient of a basis
    # function
    def EvaluateBasisSpatialGradient(self,A, x_pts, xi_vals):
        def_grad = self.EvaluateDeformationGradient(x_pts, xi_vals)
        pgrad = self.EvaluateBasisParametricGradient(A, xi_vals)
        s_grad = np.linalg.solve(def_grad.transpose(),pgrad)
        return s_grad

    # Grid plotting functionality that is used
    # in all other plotting functions
    def PlotGridData(self,X,Y,Z,npts=21,contours=False,xlabel=r"$x$",ylabel=r"$y$",zlabel=r"$z$", show_plot = True):
        if contours:
            fig, ax = plt.subplots()
            surf = ax.contourf(X,Y,Z,levels=100,cmap=matplotlib.cm.jet)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.colorbar(surf)
        else:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(X, Y, Z, cmap=matplotlib.cm.jet,
                           linewidth=0, antialiased=False)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
        if show_plot:
            plt.show()
        
        return fig,ax

    # plot the mapping from parent domain to 
    # spatial domain            
    def PlotSpatialMapping(self,x_pts,npts=21,contours=False):
        dim = len(x_pts[0])
        
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)

        Xi,Eta = np.meshgrid(xivals,etavals)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)
        Z = np.zeros(Xi.shape)
        
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                X[j,i] = pt[0]
                Y[j,i] = pt[1]
                if dim == 3:
                    Z[i,j] = pt[2] 
        
        self.PlotGridData(X,Y,Z,contours=contours,)

    # plot a basis function defined on a parent
    # domain; this is similar to what was
    # in a previous homework, but slightly generalized                
    def PlotBasisFunctionParentDomain(self,A,npts=21,contours=False):
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        Z = np.zeros(Xi.shape)
        
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                Z[j,i] = self.EvalBasisFunction(A, xi_vals)

        self.PlotGridData(Xi,Eta,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$N(\xi,\eta)$")

    # plot a basis function defined on a spatial
    # domain
    def PlotBasisFunctionSpatialDomain(self,A,x_pts,npts=21,contours=False,on_parent_domain=True):
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        Z = np.zeros(Xi.shape)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)

        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                X[j,i] = pt[0]
                Y[j,i] = pt[1]
                Z[j,i] = self.EvalBasisFunction(A, xi_vals)
        
        self.PlotGridData(X,Y,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$N(\xi,\eta)$")


    # plot a solution field defined on a parent
    # domain
    def PlotParentSolutionField(self,d_coeffs,npts=21,contours = False):
        
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        Z = np.zeros(Xi.shape)
    
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                Z[j,i] = self.EvaluateFunctionParentDomain(d_coeffs,[xivals[i],etavals[j]])
    
        self.PlotGridData(Xi,Eta,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$u_h^e(\xi,\eta)$")

    # define a solution field mapped into the
    # spatial domain for an element
    def PlotSpatialSolutionField(self,d_coeffs,x_pts,npts=21,contours = False):
        
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)
        Z = np.zeros(Xi.shape)
    
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                X[j,i] = pt[0]
                Y[j,i] = pt[1]
                Z[j,i] = self.EvaluateFunctionParentDomain(d_coeffs,[xivals[i],etavals[j]])
    
        self.PlotGridData(X,Y,Z,contours=contours,zlabel=r"$u_h^e(x,y)$")


    # plot Jacobians defined on the spatial 
    # or parent domain
    def PlotJacobian(self,x_pts,npts=21,contours = False, parent_domain = False):
        
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)
        Z = np.zeros(Xi.shape)
    
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                if not parent_domain:
                    pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                    X[j,i] = pt[0]
                    Y[j,i] = pt[1]
                Z[j,i] = self.EvaluateJacobian(x_pts,xi_vals)
    
        if parent_domain:
            self.PlotGridData(Xi,Eta,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$J^e(\xi,\eta)$")
        else:
            self.PlotGridData(X,Y,Z,contours=contours,zlabel=r"$J^e(x,y)$")

    def PlotBasisFunctionGradient(self,A,x_pts,npts=21, parent_domain = True, parent_gradient = True):
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)
        Z = np.zeros(Xi.shape)
        U = np.zeros(Xi.shape)
        V = np.zeros(Xi.shape)
    
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                if not parent_domain:
                    pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                    X[j,i] = pt[0]
                    Y[j,i] = pt[1]
                if parent_gradient:
                    grad = self.EvaluateBasisParametricGradient(A, xi_vals)
                else:
                    grad = self.EvaluateBasisSpatialGradient(A, x_pts, xi_vals)
                U[j,i] = grad[0]
                V[j,i] = grad[1]
                Z[j,i] = self.EvalBasisFunction(A, xi_vals)

        if parent_domain:
            fig,ax = self.PlotGridData(Xi,Eta,Z,contours=True,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$J^e(\xi,\eta)$",show_plot = False)
            ax.quiver(Xi,Eta,U,V)
        else:
            fig,ax = self.PlotGridData(X,Y,Z,contours=True,zlabel=r"$J^e(x,y)$",show_plot = False)
            ax.quiver(X,Y,U,V)
        plt.show()

# # transform into a quarter circle sort of
# def Prompt1():
#     # degx = 2
#     # degy = 2
#     # interval_x = [-1, 0, 1]#np.linspace(-1,1,degx+1)
#     # interval_y = [-1, 0, 1]#np.linspace(-1,1,degy+1)
#     # lg = LagrangeBasis2D(degx,degy,interval_x,interval_y)
#     lg = LagrangeBasis2D(2,2,[-1,0,1],[-1,0,1])
#     cpts = [[0,0],[0,1],[1,1],[-1,0],[-1,2],[1,2],[-2,0],[-2,3],[1,3]]
#     lg.PlotBasisFunctionGradient(0, cpts, parent_domain=True, parent_gradient = True)
#     lg.PlotBasisFunctionGradient(0, cpts, parent_domain=True, parent_gradient = False)
#     lg.PlotBasisFunctionGradient(0, cpts, parent_domain=False, parent_gradient = True)
#     lg.PlotBasisFunctionGradient(0, cpts, parent_domain=False, parent_gradient = False)

# Prompt1()


# x_pts = [[0,0],[0,1],[1,1],[-1,0],[-1,2],[1,2],[-2,0],[-2,3],[1,3]]
# lb = LagrangeBasis2D(2, 2, [-1, 0 , 1], [-1, 0, 1])
# A = 1

# lb.PlotBasisFunctionGradient(A, x_pts, parent_domain=True, parent_gradient=True)
# lb.PlotBasisFunctionGradient(A, x_pts, parent_domain=True, parent_gradient=False)
# lb.PlotBasisFunctionGradient(A, x_pts, parent_domain=False, parent_gradient=True)
# lb.PlotBasisFunctionGradient(A, x_pts, parent_domain=False, parent_gradient=False)

# degx = 2
# degy = 2
# interval_x = np.linspace(-1,1,degx+1)
# interval_y = np.linspace(-1,1,degy+1)

# lg = LagrangeBasis2D(degx,degy,interval_x,interval_y)
# dcoeffs = [-1,2,3,5,6,7,2,1,3]
# lg.PlotBasisFunctionParentDomain(1,contours=False)
# lg.PlotParentSolutionField(dcoeffs,npts=100,contours=False)

# cpts = [[0,0],[0,1],[1,1],[-1,0],[-1,2],[1,2],[-2,0],[-2,3],[1,3]]
# lg.PlotSpatialMapping(cpts,21,contours=False)
# lg.PlotBasisFunctionSpatialDomain(4, cpts,contours=False)
# lg.PlotSpatialSolutionField(dcoeffs,cpts,contours=False)

# lg.PlotJacobian(cpts,parent_domain=True,contours=True)
# lg.PlotJacobian(cpts,parent_domain=False,contours=True)

