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

os.environ["SWEEPS_API_PATH"] = "/Users/raminpahnabi/Documents/BYU/sweeps/build/src/api"
import splines as spline

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

#################################################################################################
################################     Stokes_Flow_div_free     ###################################
#################################################################################################
def Stokes(basis, deg, gaussian, quad_1D, gamma, f, u_exact, boundary_conditions, boundary_value_function, ifID, nu,
           outflow_faces=None):  # ['right']: traction-free outlet (no strong u·n, no Nitsche on those faces only)

    if ifID:

        # n_hdiv_1_comp = cf.GetNumberH1FirstComponent(basis)[0]
        # n_hdiv_2_comp = cf.GetNumberH1FirstComponent(basis)[1]
        # n_hdiv = n_hdiv_1_comp + n_hdiv_2_comp
        n_hdiv = basis.HDIV.numTotalFunctions()
        n_l2 = basis.L2.numTotalFunctions()
        n_total_funcs = n_hdiv + n_l2

        _out = list(outflow_faces or [])

        boundary_conditions = cf.find_boundary_elements(basis)  # used by local assembly (works for TP and LR)

        if bc.is_hierarchical(basis):  
            boundary_dofs = bc._lr_to_boundary_dofs(boundary_conditions)  # convert LR format to standard dict
        else:  # tensor-product basis
            boundary_dofs = bc.GetBoundaryDOFs(basis, deg)  

        # all_tangential = set(boundary_dofs['all_tangential'])
        # all_normal = set(boundary_dofs['all_normal'])

        # Compute prescribed values for normal DOFs
        prescribed = bc.ComputePrescribedNormalDOFValues(basis, boundary_dofs, boundary_value_function, quad_1D,
                                                         skip_faces=_out)  
        # Build ID array (marks normal DOFs as -1)
        ID = cf.ID_array(basis.HDIV, basis.L2, boundary_dofs, free_faces=_out)  
        
        n = max(ID)+1

        K = lil_matrix((n, n))
        F = np.zeros(n)
        
        
        
        for e in basis.elements():
            basis.localizeElement(e)

            # print("HDIV connectivity elem 0:", basis.HDIV.connectivity(e))
            # break

            ke = la.LocalStiffnessStokes(basis, deg, gaussian, quad_1D, e, boundary_conditions, nu=nu)
            fe = la.LocalForceStokes(basis, deg, gaussian, quad_1D, gamma, e, f,nu)
            ke_Nitsche = ni.LocalStiffnessMatrix_Nitsche_IGA_2D(basis, deg, gaussian, quad_1D, gamma, e, nu=nu,
                                                                 skip_faces=_out)  
            fe_Nitsche = ni.LocalForceVector_Nitsche_IGA_2D(basis, deg, gaussian, quad_1D, gamma, e, f, u_exact, boundary_value_function, nu=nu,
                                                             skip_faces=_out)  


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
                F[P] += fe_Nitsche[a]


                for b in range(0, n_local_hdiv):
                    B = local_IEN_HDIV[b]
                    Q = ID[B]

                    if Q == -1:
                        u_B = prescribed.get(B, 0.0)
                        F[P] -= ke[a, b] * u_B
                        F[P] -= ke_Nitsche[a, b] * u_B

                        continue

                    K[P, Q] += ke[a, b]
                    K[P, Q] += ke_Nitsche[a, b]
        
                    
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

        d_total = cf.ExtractTotalD(ID, d_reduced, prescribed, n_hdiv, n_l2)
    
    elif not ifID:
        
        n_hdiv_1_comp = cf.GetNumberH1FirstComponent(basis)[0]
        n_hdiv_2_comp = cf.GetNumberH1FirstComponent(basis)[1]
        n_hdiv = n_hdiv_1_comp + n_hdiv_2_comp
        # n_hdiv = basis.HDIV.numTotalFunctions()
        n_l2 = basis.L2.numTotalFunctions()
        n_total_funcs = n_hdiv + n_l2

        K = lil_matrix((n_total_funcs, n_total_funcs))
        F = np.zeros(n_total_funcs)

        for e in basis.elements():
            basis.localizeElement(e)

            # print("HDIV connectivity elem 0:", basis.HDIV.connectivity(e))
            # break

            ke = la.LocalStiffnessStokes(basis, deg, gaussian, quad_1D, e, boundary_conditions, nu=nu)  
            fe = la.LocalForceStokes(basis, deg, gaussian, quad_1D, gamma, e, f)
            ke_Nitsche = ni.LocalStiffnessMatrix_Nitsche_IGA_2D(basis, deg, gaussian, quad_1D, gamma, e, nu=nu) 
            fe_Nitsche = ni.LocalForceVector_Nitsche_IGA_2D(basis, deg, gaussian, quad_1D, gamma, e, f, u_exact, boundary_value_function, nu=nu)

            local_IEN_HDIV = basis.HDIV.connectivity(e)
            n_local_hdiv = len(local_IEN_HDIV)
            local_IEN_L2 = basis.L2.connectivity(e)
            n_local_L2 = len(local_IEN_L2)

            for a in range(0,n_local_hdiv):
                A = local_IEN_HDIV[a]
                F[A] += fe[a]
                F[A] += fe_Nitsche[a]
         
                for b in range (0,n_local_hdiv):
                    B = local_IEN_HDIV[b]
                    K[A,B] += ke[a,b]
                    K[A,B] += ke_Nitsche[a,b]
                    
                for b in range(0,n_local_L2):
                    B = n_hdiv + local_IEN_L2[b]

                    K[A,B] += ke[a, b + n_local_hdiv]
                    K[B,A] += ke[b + n_local_hdiv, a]
                    
                    K[A,B] += ke_Nitsche[a, b + n_local_hdiv]
                    K[B,A] += ke_Nitsche[b + n_local_hdiv, a]
            
        d_total = spsolve(K.tocsr(), F)
    
    return d_total

