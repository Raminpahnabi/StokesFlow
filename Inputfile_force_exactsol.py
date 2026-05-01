#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:28:10 2026

@author: raminpahnabi
"""

import Inputfile as inp


use_curve_geometry = inp.USE_CURVED_GEOMETRY
L2Projection       = inp.is_L2Projection
Stokes             = inp.is_Stokes
NavierStokes       = inp.is_NavierStokes
JetNavierStokes    = inp.is_JetNavierStokes
option             = inp.option_number

if L2Projection:
    if use_curve_geometry   == False:
        if option == 0:
            forcing_function        = inp.forcing_function_l2projection_0
            exact_solution          = inp.exact_solution_0
            exact_solution_l2       = inp.exact_solution_l2_0
            boundary_value_function = inp.boundary_value_function_0
        
        elif option == 1:
            forcing_function        = inp.forcing_function_l2projection_1
            exact_solution          = inp.exact_solution_1
            exact_solution_l2       = inp.exact_solution_l2_1
            boundary_value_function = inp.boundary_value_function_1
    
    elif use_curve_geometry == True: 
        if option == 0:
            forcing_function        = inp.forcing_function_l2projection_0
            exact_solution          = inp.exact_solution_0
            exact_solution_l2       = inp.exact_solution_l2_0
            boundary_value_function = inp.boundary_value_function_0
        elif option == 1:
            forcing_function        = inp.forcing_function_l2projection_1
            exact_solution          = inp.exact_solution_1
            exact_solution_l2       = inp.exact_solution_l2_1
            boundary_value_function = inp.boundary_value_function_1


elif Stokes:
    if use_curve_geometry   == False:
        if option == 1:
            forcing_function        = inp.forcing_function_s_1
            exact_solution          = inp.exact_solution_1
            exact_solution_l2       = inp.exact_solution_l2_1
            boundary_value_function = inp.boundary_value_function_1
        
        elif option == 2:
            forcing_function        = inp.forcing_function_cavity_2
            exact_solution          = inp.exact_solution_cavity_2
            exact_solution_l2       = inp.exact_solution_l2_cavity_2
            boundary_value_function = inp.boundary_value_function_cavity_2
            
    elif use_curve_geometry   == True:
        if option == 0:
            forcing_function        = inp.forcing_function_l2projection_0
            exact_solution          = inp.exact_solution_0
            exact_solution_l2       = inp.exact_solution_l2_0
            boundary_value_function = inp.boundary_value_function_0
        elif option == 1:
            forcing_function        = inp.forcing_function_s_1_curve
            exact_solution          = inp.exact_solution_1_curve
            exact_solution_l2       = inp.exact_solution_l2_1_curve
            boundary_value_function = inp.boundary_value_function_1_curve


elif NavierStokes:
    if use_curve_geometry   == False:
        if option == 1:
            forcing_function        = inp.forcing_function_s_1
            f_ns                    = inp.forcing_function_ns_1
            exact_solution          = inp.exact_solution_1
            exact_solution_l2       = inp.exact_solution_l2_1
            boundary_value_function = inp.boundary_value_function_1
        
        elif option == 2:
            forcing_function        = inp.forcing_function_cavity_2
            f_ns                    = inp.forcing_function_cavity_2
            exact_solution          = inp.exact_solution_cavity_2
            exact_solution_l2       = inp.exact_solution_l2_cavity_2
            boundary_value_function = inp.boundary_value_function_cavity_2
