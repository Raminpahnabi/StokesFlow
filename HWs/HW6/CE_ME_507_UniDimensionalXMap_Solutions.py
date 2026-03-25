#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:18:02 2024

@author: kendrickshepherd
"""

import sys
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt

sys.path.append('../HW3/')

import Homework_3_Python_Solutions as HW3

def XMap(deg,spatial_pts,interp_pts,xi):
    xval = 0
    for a in range(0,deg+1):
        xval += spatial_pts[a] * HW3.LagrangeBasisEvaluation(deg,interp_pts,xi,a)
    return xval

def PlotXMap(deg,spatial_pts,interp_pts, npts=101, contours=True):
    xi_vals = np.linspace(interp_pts[0],interp_pts[-1],npts)
    
    if not contours:
        xs = []
        for i in range(0,len(xi_vals)):
            xi = xi_vals[i]
            xs.append(XMap(deg,spatial_pts,interp_pts,xi))
        
        plt.plot(xi_vals,xs)
        plt.show()

    else:
        Xi,Xi2 = np.meshgrid(xi_vals,[-0.2,0.2])
        Z = np.zeros(Xi.shape)
        for i in range(0,len(xi_vals)):
            xi = xi_vals[i]
            x = XMap(deg,spatial_pts,interp_pts,xi)
            for j in range(0,2):
                Z[j,i] = x

        fig, ax = plt.subplots()
        surf = ax.contourf(Z,Xi2,Xi,levels=100,cmap=matplotlib.cm.binary)
        ax.set_xlabel(r"$x$")
        ax.yaxis.set_ticklabels([])
        fig.colorbar(surf)
        plt.show()

    
PlotXMap(2,[0.5,0.7,1.5],[-1,0,1],contours=False)
# PlotXMap(2,[-1,-0.6,1],[0.5,0.7,1.5],contours=False)
