#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 13:40:03 2025

@author: raminpahnabi
"""

import sys
import os
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from sweeps_path import ensure_sweeps_api_on_path

ensure_sweeps_api_on_path()
sys.path.append(str(PROJECT_ROOT / 'HWs'))
sys.path.append(str(PROJECT_ROOT / 'Required'))

import LocalAssembly as la
import Nitsche as ni
import CommonFuncs as cf
import BoundaryConditions as bc

################################################################################
##############     L2Projection_div_free     ###################################
################################################################################
def L2Projection(basis, deg, gaussian, quad_1D, gamma, f, u_exact, boundary_conditions, boundary_value_function, ifID, nu,use_curve_geometry):  #ns added nu so the Stokes solver (used as NS initial guess) uses the correct viscosity throughout
    
    if ifID:
        
        if bc.is_hierarchical(basis):  # LR basis: GetNumberH1FirstComponent uses knotVectors() → unavailable
            n_hdiv = basis.HDIV.numTotalFunctions()  
        else:  
            n_hdiv_1_comp = cf.GetNumberH1FirstComponent(basis)[0]
            n_hdiv_2_comp = cf.GetNumberH1FirstComponent(basis)[1]
            n_hdiv = n_hdiv_1_comp + n_hdiv_2_comp
        n_l2 = basis.L2.numTotalFunctions()
        n_total_funcs = n_hdiv + n_l2


        if bc.is_hierarchical(basis):  # LR basis: build boundary_dofs from find_boundary_elements output
            boundary_conditions_lr = cf.find_boundary_elements(basis)  
            boundary_dofs = bc._lr_to_boundary_dofs(boundary_conditions_lr)  
        else:  
            boundary_dofs = bc.GetBoundaryDOFs(basis, deg) 
            
        
        # n_hdiv_1_comp = cf.GetNumberH1FirstComponent(basis)[0]
        # n_hdiv_2_comp = cf.GetNumberH1FirstComponent(basis)[1]
        # n_hdiv = n_hdiv_1_comp + n_hdiv_2_comp
        # n_l2 = basis.L2.numTotalFunctions()
    
        # boundary_dofs = bc.GetBoundaryDOFs(basis)
        prescribed = bc.ComputePrescribedNormalDOFValues(basis, boundary_dofs, boundary_value_function, quad_1D)
        
        ID = cf.ID_array_l2projection(basis.HDIV, basis.L2, boundary_dofs)
        
        n = max(ID)+1

        K = lil_matrix((n, n))
        F = np.zeros(n)
        
        
        for e in basis.elements():
            basis.localizeElement(e)

            ke = la.LocalStiffnessL2Projection(basis, deg, gaussian, quad_1D, e)
            fe = la.LocalForceStokesL2Projection(basis, deg, gaussian, quad_1D, gamma, e, f, nu,use_curve_geometry)

            local_IEN_HDIV = basis.HDIV.connectivity(e)
            n_local_hdiv = len(local_IEN_HDIV)
            local_IEN_L2 = basis.L2.connectivity(e)
            n_local_L2 = len(local_IEN_L2)

            for a in range(0,n_local_hdiv):
                A = local_IEN_HDIV[a]
                P = ID[A]
                
                if P == -1:
                    continue
                F[P] += fe[a]

                for b in range(0, n_local_hdiv):
                    B = local_IEN_HDIV[b]
                    Q = ID[B]

                    if Q == -1:
                        u_B = prescribed.get(B, 0.0)
                        F[P] -= ke[a, b] * u_B

                        continue

                    K[P, Q] += ke[a, b]        
                    
                for b in range(0, n_local_L2):
                    B = n_hdiv + local_IEN_L2[b]
                    Q = ID[B]
    
                    if Q == -1:
                        u_B = prescribed.get(B, 0.0)
                        F[P] -= ke[a, b + n_local_hdiv] * u_B
                        continue
                    
                    K[P, Q] += ke[a, b + n_local_hdiv]
                    K[Q, P] += ke[b + n_local_hdiv, a]
                    
        
        d_reduced = spsolve(K.tocsr(), F)
        d_total = cf.ExtractTotalD_l2projection(ID, d_reduced, prescribed, n_hdiv, n_l2)
    
    return d_total

