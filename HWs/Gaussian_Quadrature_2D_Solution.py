#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 16:51:03 2023

@author: kendrickshepherd
"""

import math
import numpy as np
import scipy
from scipy import sparse
from scipy import linalg

import sys
import itertools

# Beta term from Trefethen, Bau Equation 37.6
def BetaTerm(n):
    if n <= 0:
        return 0
    else:
        return 0.5*math.pow((1-math.pow(2*n,-2)),-0.5)

# Theorem 37.4 from Trefethen, Bau
def ComputeQuadraturePtsWts(n):
    # Compute the Jacobi Matrix, T_n
    # given explicitly in Equation 37.6
    diag = np.zeros(n)
    off_diag = np.zeros(n-1)
    for i in range(0,n-1):
        off_diag[i] = BetaTerm(i+1)
        
    # Divide and conquer algorithm for tridiagonal
    # matrices
    # w is eigenvalues
    # v is matrix with columns corresponding eigenvectors
    [w,v] = scipy.linalg.eigh_tridiagonal(diag,off_diag,check_finite=False)
    
    # nodes of quadrature given as eigenvalues
    nodes = w
    # weights given as two times the square of the first 
    # index of each eigenvector
    weights = 2*(v[0,:]**2)
    
    return [nodes,weights]

#FROM HW17
class GaussQuadrature1D:
    
    def __init__(self,n_quad, start_pt = -1, end_pt = 1):
        self.n_quad = n_quad
        [self.quad_pts,self.quad_wts] = ComputeQuadraturePtsWts(self.n_quad)
        self.jacobian = 1
        self.start = start_pt
        self.end = end_pt
        
        if start_pt != -1 or end_pt != 1:
           self.__TransformToInterval__(start_pt,end_pt)
     
    def __TransformToInterval__(self,start,end):
        # complete this function
        new_quad_pts = np.zeros((self.n_quad))
        # loop through each quadrature point
        for i in range(self.n_quad):
            new_quad_pts[i] = ((end - start) / 2) * self.quad_pts[i] + (start + end) / 2
        # update the list of quadrature points
        self.quad_pts = new_quad_pts
        # update the new jacobian
        self.jacobian = (end - start) / 2
        
        
class GaussQuadrature:

    def __init__(self, n_quad,interval):
        self.n_quad = n_quad
        [self.quad_pts, self.quad_wts] = ComputeQuadraturePtsWts(self.n_quad)
        self.quad_pts,self.jacobian=self.__transformtointerval__(interval,self.quad_pts)
        
    def __transformtointerval__(self,interval,quad_pts):
      new_pts = []
      start= interval[0]
      end =interval[1]
      for pt in quad_pts:
          new_pts.append((end-start)/2 * pt + (start+end)/2)
      
      jacobian = (end-start)/2
      return new_pts,jacobian


# class GaussQuadratureQuadrilateral:
    
#     def __init__(self,n_quad,start = -1,end = 1,ndim = 1):
#         self.n_quad = n_quad
#         self.jacobian = 1
#         [self.quad_pts,self.quad_wts] = ComputeQuadraturePtsWts(self.n_quad)
#         self.start = start
#         self.end = end
#         self.ndim = ndim
#         if start != -1 or end != 1:
#             self.__TransformToInterval__(start,end)
            
#         temp_wts = self.quad_wts
#         temp_pts = self.quad_pts
        
#         if ndim == 2:
#             # pts_extrusion = np.broadcast_to(temp_pts.reshape(n_quad,1),(4,2)).copy()
#             param_pts = []
#             param_wts = []
#             for j in range(0,n_quad):
#                 for i in range(0,n_quad):
#                     param_pts.append([temp_pts[i],temp_pts[j]])
#                     param_wts.append(temp_wts[i]*temp_wts[j])
            
        
#             # for i in range(0,ndim-1):
#             #     temp_wts = np.kron(temp_wts,self.quad_wts)
#                 # temp_pts = np.kron(temp_pts,self.quad_pts)
#             self.n_quad *= self.n_quad
#             self.quad_pts = param_pts
#             self.quad_wts = param_wts
#             self.jacobian = self.jacobian**2

  

class GaussQuadratureQuadrilateral:
    
    def __init__(self,n_quad,start = -1,end = 1):
        self.n_quad = n_quad
        self.jacobian = 1
        [self.quad_pts,self.quad_wts] = ComputeQuadraturePtsWts(self.n_quad)
        self.start = start
        self.end = end
        if start != -1 or end != 1:
            self.__TransformToInterval__(start,end)

        new_pts = []
        new_wts = []
        for i in range(n_quad):
            new_pts.append([(self.quad_pts[i], self.quad_pts[j]) for j in range(n_quad)])
        for i in range(n_quad):
            for j in range(n_quad):
                new_wts.append(self.quad_wts[i] * self.quad_wts[j])
            
        self.pts = new_pts
        self.wts = new_wts
        print(self.pts, self.wts, self.jacobian)
        
    def __TransformToInterval__(self,start,end):
        # complete this function
        new_quad_pts = np.zeros((self.n_quad))
        # loop through each quadrature point
        for i in range(self.n_quad):
            new_quad_pts[i] = ((end - start) / 2) * self.quad_pts[i] + (start + end) / 2
        # update the list of quadrature points
        self.quad_pts = new_quad_pts
        # update the new jacobian
        self.jacobian = (end - start) / 2
        
        
class GaussQuadrature2D:

    def __init__(self, n_quad_x, n_quad_y, x_interval=[-1,1],y_interval=[-1,1]):
        self.n_quad_x = n_quad_x
        self.n_quad_y = n_quad_y
        [self.quad_pts_x, self.quad_wts_x] = ComputeQuadraturePtsWts(self.n_quad_x)
        [self.quad_pts_y, self.quad_wts_y] = ComputeQuadraturePtsWts(self.n_quad_y)
        
        self.quad_pts_x,x_jacobian=self.__transformtointerval__(x_interval,self.quad_pts_x)
        self.quad_pts_y,y_jacobian=self.__transformtointerval__(y_interval,self.quad_pts_y)
        temp_pts = []
        temp_wts = []

        for i in range(self.n_quad_y):
            for j in range(self.n_quad_x):
                xi = (self.quad_pts_x[j])
                eta = (self.quad_pts_y[i])
                weight_x = self.quad_wts_x[j]
                weight_y = self.quad_wts_y[i]
                temp_pts.append([xi, eta])
                temp_wts.append(weight_x * weight_y)

        self.quad_pts = temp_pts
        self.quad_wts = temp_wts
        
        self.jacobian= x_jacobian*y_jacobian

    def __transformtointerval__(self,interval,quad_pts):
      new_pts = []
      start= interval[0]
      end =interval[1]
      for pt in quad_pts:
          new_pts.append((end-start)/2 * pt + (start+end)/2)
      
      jacobian = (end-start)/2
      return new_pts,jacobian