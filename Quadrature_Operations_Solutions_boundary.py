#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:31:53 2024

@author: kendrickshepherd
"""

import os
import sys
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt

# sys.path.append("../HW10/")
# sys.path.append('../HW16/')
sys.path.append(os.path.join(os.getcwd(), './HWs'))

import Gaussian_Quadrature_2D_Solution as gq
import MultidimensionalSpatialParametricGradient_Solutions as bf

from enum import Enum

class BoundaryFace(Enum):
    BOTTOM = 0
    TOP = 1
    LEFT = 2
    RIGHT = 3
    
# Convert a 
def GetFaceQuadraturePoints(quad_1d,bdry_face):
    if bdry_face == BoundaryFace.BOTTOM:
    # if bdry_face == 0:
        return [[pt,quad_1d.start] for pt in quad_1d.quad_pts]
    elif bdry_face == BoundaryFace.TOP:
    # elif bdry_face == 1:
        return [[pt,quad_1d.end] for pt in quad_1d.quad_pts]
    elif bdry_face == BoundaryFace.LEFT:
    # elif bdry_face == 2:
        return [[quad_1d.start,pt] for pt in quad_1d.quad_pts]
    elif bdry_face == BoundaryFace.RIGHT:
    # elif bdry_face == 3:
        return [[quad_1d.end,pt] for pt in quad_1d.quad_pts]
    else:
        sys.exit("An invalid boundary face has been selected", bdry_face)
        
# Determine which column of a boundary face should be extracted from
# the deformation gradient (Jacobian matrix)
def __BdryFaceToVaryingCoordinate__(bdry_face):
    if bdry_face==BoundaryFace.BOTTOM or bdry_face==BoundaryFace.TOP:
        """Returns 0 if xi varies, 1 if eta varies."""
        return 0 # \xi varies while \eta is fixed
    elif bdry_face==BoundaryFace.LEFT or bdry_face==BoundaryFace.RIGHT:
        return 1 # \eta varies while \xi is fixed
    
# Extract the appropriate differential vector from a boundary face
# given points on the face, control points that define the mapping,
# a basis function object, and the face of interest
def DifferentialVector(xi_vals,x_pts,lagrange_basis,bdry_face):
    def_grad = lagrange_basis.EvaluateDeformationGradient(x_pts, xi_vals)
    varying_coord = __BdryFaceToVaryingCoordinate__(bdry_face)
    return def_grad[:,varying_coord] 
"""extract the column that matches the varying direction for further calculations"""
# Compute the Jacobian of a curve given points on the face,
# control points that define the parent to spatial mapping,
# a basis function object, and the face of interest
def JacobianOneD(xi_vals,x_pts,lagrange_basis,bdry_face):
    diff_vect = DifferentialVector(xi_vals,x_pts,lagrange_basis,bdry_face)
    return np.sqrt(diff_vect[0]**2 + diff_vect[1]**2)









# Gaussian quadrature on a prescribed 2d function
def Problem1():
    # original integrand is (x^2 + x*y + y)
    # transformed to -1,1 is below, with jacobian of 1/2
    #   to represent the change in length of y from length
    #   of 1 to length of 1/2
    func = lambda u,v: (u+1)**2 + (u+1)*(v+3)/2 + (v+3)/2
    jac = 0.5
    
    start = -1
    end = 1
    n_quad = 3
    n_dim = 2
    # quad = gq.GaussQuadratureQuadrilateral(n_quad,start,end,n_dim)
    quad = gq.GaussQuadratureQuadrilateral(n_quad,start,end)
    
    
    integral = 0
    for i in range(0,quad.n_quad):
        loc = quad.pts[i]
        # integral += quad.quad_wts[i] * func(loc[0],loc[1]) * quad.jacobian
        integral += quad.wts[i] * func(loc[0],loc[1]) * quad.jacobian
    
    integral *= jac
    print("The requested integral for Problem 1 is ", integral)
    
# Evaluate the length of all sides of an input cell
def Problem2():
    degx = 2
    degy = 2
    interval_x = np.linspace(-1,1,degx+1)
    interval_y = np.linspace(-1,1,degy+1)
    lg = bf.LagrangeBasis2D(degx,degy,interval_x,interval_y)
    cpts = [[0,0],[0,1],[1,1],[-1,0],[-1,2],[1,2],[-2,0],[-2,3],[1,3]]
    
    lg.PlotSpatialMapping(cpts,contours=True)
    
    n_quad = 3
    gq_1D = gq.GaussQuadrature1D(n_quad,start_pt=-1,end_pt=1)
    
    bdries = [BoundaryFace.BOTTOM,BoundaryFace.TOP,BoundaryFace.LEFT,BoundaryFace.RIGHT]
    
    lens = []
    for bdry in bdries:
        xi_vals = GetFaceQuadraturePoints(gq_1D, bdry)
        side_len = 0
        for i in range(0,len(xi_vals)):
            side_len += gq_1D.quad_wts[i] * JacobianOneD(xi_vals[i], cpts, lg, bdry)
            # side_len += gq_1D.wts[i] * JacobianOneD(xi_vals[i], cpts, lg, bdry)
        lens.append(side_len)
    
    for i in range(0,len(lens)):
        print(bdries[i],"length:", lens[i])
        
    
# Problem1()
# Problem2()