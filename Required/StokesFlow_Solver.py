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
sys.path.insert(0, '/Users/raminpahnabi/Documents/BYU/sweeps/build/src/api')
# sys.path.append('HWs')
sys.path.append(os.path.join(os.getcwd(), '../HWs'))
sys.path.append(os.path.join(os.getcwd(), 'Required'))

import LocalAssembly as la
import Nitsche as ni
import CommonFuncs as cf
import BoundaryConditions as bc

#################################################################################################
################################     Stokes_Flow_div_free     ###################################
#################################################################################################
def Stokes(basis, deg, gaussian, quad_1D, gamma, f, u_exact, boundary_conditions,boundary_value_function,ifID=True):
    
    if ifID:
        
        n_hdiv_1_comp = cf.GetNumberH1FirstComponent(basis)[0]
        n_hdiv_2_comp = cf.GetNumberH1FirstComponent(basis)[1]
        n_hdiv = n_hdiv_1_comp + n_hdiv_2_comp
        # n_hdiv = basis.HDIV.numTotalFunctions()
        n_l2 = basis.L2.numTotalFunctions()
        n_total_funcs = n_hdiv + n_l2
    
        
        boundary_dofs = bc.GetBoundaryDOFs(basis)
        all_tangential = set(boundary_dofs['all_tangential'])
        all_normal = set(boundary_dofs['all_normal'])
        
        # Compute prescribed values for normal DOFs
        prescribed = bc.ComputePrescribedNormalDOFValues(basis, boundary_dofs, boundary_value_function, quad_1D)
        # Build ID array (marks normal DOFs as -1)
        ID = cf.ID_array(basis.HDIV, basis.L2, boundary_dofs, prescribed)
        
        n = max(ID)+1

        K = lil_matrix((n, n))
        F = np.zeros(n)
        
        
        
        for e_idx, e in enumerate(basis.elements()):  # NEWCODE: enumerate for parametric boundary detection
            basis.localizeElement(e)

            # print("HDIV connectivity elem 0:", basis.HDIV.connectivity(e))
            # break

            ke = la.LocalStiffnessStokes(basis, deg, gaussian, quad_1D, e, boundary_conditions)
            fe = la.LocalForceStokes(basis, deg, gaussian, quad_1D, gamma, e,f)
            ke_Nitsche = ni.LocalStiffnessMatrix_Nitsche_IGA_2D(basis, deg, gaussian, quad_1D, gamma, e)  # NEWCODE
            fe_Nitsche = ni.LocalForceVector_Nitsche_IGA_2D(basis, deg, gaussian, quad_1D, gamma, e, f, u_exact,boundary_value_function)  # NEWCODE
                            

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

        for e_idx, e in enumerate(basis.elements()):  # NEWCODE: enumerate for parametric boundary detection
            basis.localizeElement(e)

            # print("HDIV connectivity elem 0:", basis.HDIV.connectivity(e))
            # break

            ke = la.LocalStiffnessStokes(basis, deg, gaussian, quad_1D, e, boundary_conditions)
            fe = la.LocalForceStokes(basis, deg, gaussian, quad_1D, gamma, e,f)
            ke_Nitsche = ni.LocalStiffnessMatrix_Nitsche_IGA_2D(basis, deg, gaussian, quad_1D, gamma, e, e_idx)  # NEWCODE
            fe_Nitsche = ni.LocalForceVector_Nitsche_IGA_2D(basis, deg, gaussian, quad_1D, gamma, e, e_idx, f, u_exact,boundary_value_function)  # NEWCODE

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


