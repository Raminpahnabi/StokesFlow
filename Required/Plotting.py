#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:41:09 2026

@author: raminpahnabi
"""

import sys
import os
import numpy as np
sys.path.insert(0, '/Users/raminpahnabi/Documents/BYU/sweeps/build/src/api')
sys.path.append(os.path.join(os.getcwd(), '../HWs'))
sys.path.append(os.path.join(os.getcwd(), 'Required'))

import matplotlib.pyplot as plt
import CommonFuncs as cf

KINEMATIC_VISCOSITY = 1

###########################################################################
##############################     Plotting     ###########################
###########################################################################
def PlotSolution(basis, dtotal, quad, quad_1D,gamma, f_exact, elem_n, u_exact, p_exact):
    n_hdiv_1_comp = cf.GetNumberH1FirstComponent(basis)[0]
    n_hdiv_2_comp = cf.GetNumberH1FirstComponent(basis)[1]
    n_hdiv_total = n_hdiv_1_comp + n_hdiv_2_comp
    # n_hdiv_total = basis.HDIV.numTotalFunctions()
    
    # vcoeffs_1 = dtotal[:n_hdiv_1_comp]
    # vcoeffs_2 = dtotal[n_hdiv_1_comp:n_hdiv_total]
    # pcoeffs = dtotal[n_hdiv_total:]
    
    X, Y, U, V, P = [], [], [], [], []
    
    for elem in basis.elements():
        basis.localizeElement(elem)
        local_IEN_hdiv = basis.HDIV.connectivity(elem)
        local_IEN_L2   = basis.L2.connectivity(elem)
        
        for g in range(len(quad.quad_pts)):
            xi, eta = quad.quad_pts[g]
            
            basis.localizePoint([xi, eta])

            transformed_basis = basis.piolaTransformedHDIVBasis()
            uh_val = np.zeros(2)
            for a in range(0,len(local_IEN_hdiv)):
                A = local_IEN_hdiv[a]
                dA_hdiv = dtotal[A]
                uh_val += dA_hdiv * transformed_basis[a]
                
            phi_L2 = basis.piolaTransformedL2()          
            p_val = 0
            for b in range(0,len(local_IEN_L2)):
                B = local_IEN_L2[b] + n_hdiv_total
                dA_L2 = dtotal[B]
                p_val += dA_L2 * phi_L2[b]
                
            qpt_mapped = basis.mapping()
            x_g, y_g = qpt_mapped[0], qpt_mapped[1]
            p_val = float(p_val)

            
            X.append(x_g)
            Y.append(y_g)
            U.append(uh_val[0])
            V.append(uh_val[1])
            P.append(p_val)
        
    X = np.array(X)
    Y = np.array(Y)
    U = np.array(U)
    V = np.array(V)
    P = np.array(P)
    
    # U_exact = Y #np.sin(np.pi * Y)
    # V_exact = X #np.sin(np.pi * X)
    # P_exact = 0
    
    U_exact = u_exact(X,Y)[0]
    V_exact = u_exact(X,Y)[1]
    
    P_exact = p_exact(X,Y) 

    
    error_hdiv = np.sqrt(np.mean((U - U_exact)**2 + (V - V_exact)**2))
    print("L2 velocity error:", error_hdiv)
    
    error_l2 = np.sqrt(np.mean((P - P_exact)**2))
    print("L2 pressure error:", error_l2)

    
    # Vector plot comparison 
    plt.figure(figsize=(7, 6))
    plt.quiver(X, Y, U_exact, V_exact, color='red', alpha=0.4, label='Exact')
    plt.quiver(X, Y, U, V, color='blue', alpha=0.6, label='Computed')
    # plt.legend()
    plt.title("Velocity Field: Computed (blue) vs Exact (red)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
    
    # Pressure field 
    from matplotlib  import cm
    
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(X, Y,c=P, marker = 'o', cmap = cm.jet) #, P)#, levels=30, cmap='coolwarm')
    plt.colorbar(sc, label="Pressure")
    plt.title("Pressure field")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()