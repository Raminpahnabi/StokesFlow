#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:05:37 2026

@author: raminpahnabi
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

import LocalAssembly as la
import Nitsche as ni
import CommonFuncs as cf
import BoundaryConditions as bc
import Solver_StokesFlow as ss
import Inputfile as inp

nu = inp.KINEMATIC_VISCOSITY

#######################################################################
##################     NavierStokes_Flow_div_free     #################
#######################################################################
def NavierStokes(basis, deg, gaussian, quad_1D, gamma, f, f_ns, u_exact, boundary_conditions, boundary_value_function,
                 ifID=True, curve_domain=False, geo_map_func=None, d_initial=None, nu=nu,
                 outflow_faces=None):  # e.g. ['right']: natural outlet; all other faces get strong u·n + Nitsche tangential

    if bc.is_hierarchical(basis):  # LR basis: GetNumberH1FirstComponent uses knotVectors() → unavailable
        n_hdiv = basis.HDIV.numTotalFunctions()  
    else:  
        n_hdiv_1_comp = cf.GetNumberH1FirstComponent(basis)[0]
        n_hdiv_2_comp = cf.GetNumberH1FirstComponent(basis)[1]
        n_hdiv = n_hdiv_1_comp + n_hdiv_2_comp
    n_l2 = basis.L2.numTotalFunctions()
    n_total_funcs = n_hdiv + n_l2

    _out = list(outflow_faces or [])

    if bc.is_hierarchical(basis):  # LR basis: build boundary_dofs from find_boundary_elements output
        boundary_conditions_lr = cf.find_boundary_elements(basis)  
        boundary_dofs = bc._lr_to_boundary_dofs(boundary_conditions_lr)  
    else:  
        boundary_dofs = bc.GetBoundaryDOFs(basis, deg)  

    all_tangential = set(boundary_dofs['all_tangential'])
    all_normal = set(boundary_dofs['all_normal'])

    prescribed = bc.ComputePrescribedNormalDOFValues(basis, boundary_dofs, boundary_value_function, quad_1D,
                                                     skip_faces=_out)  

    ID = cf.ID_array(basis.HDIV, basis.L2, boundary_dofs, free_faces=_out)  
    
    n = max(ID)+1
    
    f_ns_nu = f_ns  
    f_stokes_nu = f 

    if d_initial is not None:
        d_prev = d_initial
    else:
        d_prev = ss.Stokes(basis, deg, gaussian, quad_1D, gamma, f_stokes_nu, u_exact,
                           boundary_conditions, boundary_value_function, ifID=ifID, nu=nu,
                           outflow_faces=_out)  
    
    max_iter = 40  
    tol = 1e-8  

    for k in range(max_iter):  
        K = lil_matrix((n, n))  
        F = np.zeros(n)  

        for e in basis.elements():  
            basis.localizeElement(e) 
            ke = la.LocalStiffnessStokes(basis, deg, gaussian, quad_1D, e, boundary_conditions, nu=nu)  
            fe = la.LocalForceStokes(basis, deg, gaussian, quad_1D, gamma, e, f_ns_nu, nu=nu)  
            ke_Nitsche = ni.LocalStiffnessMatrix_Nitsche_IGA_2D(basis, deg, gaussian, quad_1D, gamma, e, nu=nu,
                                                                 skip_faces=_out)  
            fe_Nitsche = ni.LocalForceVector_Nitsche_IGA_2D(basis, deg, gaussian, quad_1D, gamma, e, f_ns_nu, u_exact, boundary_value_function, nu=nu,
                                                             skip_faces=_out)  
            
            # ke_adv = la.LocalAdvectionPicard(basis, deg, gaussian, quad_1D, e, d_prev, boundary_conditions)
            ke_adv = la.LocalAdvectionNewton(basis, deg, gaussian, quad_1D, e, d_prev, boundary_conditions)
            fe_adv = la.LocalForceNS_Newton(basis, deg, gaussian, quad_1D, e, d_prev)
            
            ke_boundary = la.EvalLocalStiffnessStokes_boundary(basis, deg, gaussian, quad_1D, e, boundary_conditions, nu=nu)
            fe_boundary = la.EvalLocalforceStokes_boundary(basis, deg, gaussian, quad_1D, e, f_ns_nu, boundary_conditions, nu=nu)
            
            local_IEN_HDIV = basis.HDIV.connectivity(e)  
            n_local_hdiv = len(local_IEN_HDIV)  
            local_IEN_L2 = basis.L2.connectivity(e)  
            n_local_L2 = len(local_IEN_L2)  
        
            ke_total2 = ke + ke_adv + ke_boundary

            for a in range(n_local_hdiv):
                A = local_IEN_HDIV[a]
                P = ID[A]
                if P == -1:
                    continue
                F[P] += fe[a] + fe_Nitsche[a] + fe_adv[a] + fe_boundary[a]

                for b in range(n_local_hdiv):
                    B = local_IEN_HDIV[b]
                    Q = ID[B]
                    if Q == -1:
                        u_B = prescribed.get(B, 0.0)
                        F[P] -= ke_total2[a, b] * u_B       # ke + ke_adv contribution
                        F[P] -= ke_Nitsche[a, b] * u_B      # Nitsche contribution (matches Stokes)
                        continue
                    K[P, Q] += ke_total2[a, b]              # ke + ke_adv
                    K[P, Q] += ke_Nitsche[a, b]             # Nitsche velocity-velocity

                for b in range(n_local_L2):
                    B = n_hdiv + local_IEN_L2[b]
                    Q = ID[B]
                    if Q == -1:
                        u_B = prescribed.get(B, 0.0)
                        F[P] -= ke_total2[a, b + n_local_hdiv] * u_B
                        continue
                    K[P, Q] += ke_total2[a, b + n_local_hdiv]   # pressure-velocity (ke only, same as Stokes)
                    K[Q, P] += ke_total2[b + n_local_hdiv, a]   # symmetric

        d_reduced = spsolve(K.tocsr(), F)  
        
        d_picard = cf.ExtractTotalD(ID, d_reduced, prescribed, n_hdiv, n_l2) 
        d_next = d_picard
         
        rel_change = np.linalg.norm(d_next[:n_hdiv] - d_prev[:n_hdiv]) / max(np.linalg.norm(d_next[:n_hdiv]), 1e-14)  
        print(f"Picard iter {k+1:02d}: rel_change = {rel_change:.3e}")  
        d_prev = d_next  
        if rel_change < tol:  
            print(f"Picard converged in {k+1} iterations.")  
            break  

    return d_prev  
