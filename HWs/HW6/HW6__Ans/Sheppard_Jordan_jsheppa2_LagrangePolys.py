"""
Homework 03 - Lagrange Basis Functions

@author: Jordan Sheppard
"""
import sys
import numpy as np
from matplotlib import pyplot as plt

# Given a set of points, pts, to interpolate
# and a polynomial degree, this function will
# evaluate the a-th basis function at the 
# location xi
def LagrangeBasisEvaluation(p,pts,xi,a):
    # Ensure correct number of points
    if (p+1 != len(pts)):
        sys.exit("The number of input points for interpolating must be the same as one plus the polynomial degree for interpolating")
    
    # Ensure each point in pts is unique
    if len(set(pts)) != len(pts):
        raise ValueError("The input points in pts should all be unique")

    # Ensure that pts is ordered from least to greatest 
    sorted = True 
    pt = pts[0]
    for other_pt in pts[1:]:
        if other_pt <= pt:
            sorted = False 
            break
        pt = other_pt 
    if not sorted:
        raise ValueError("The points in pts should be sorted from least to greatest")

    # Make sure pts is an array for easy manipulation 
    if not isinstance(pts, np.ndarray):
        pts = np.array(pts)
    
    # Evaluate a-th Lagrange basis function of order p at point xi 
    # as given in Hughes Equation 3.6.1
    # (uses numpy fancy indexing and array broadcasting rather than for-loops
    # added efficiency)
    pt_indexes = np.concatenate(
        (np.arange(0, a), np.arange(a+1, p+1))
    )   # Gives all indexes [0, 1, ..., a-1, a+1, ..., p]
    numerator = np.prod(xi - pts[pt_indexes])
    denominator = np.prod(pts[a] - pts[pt_indexes])
    return numerator/denominator 


# Plot the Lagrange polynomial basis functions
# COMMENT THIS CODE
def PlotLagrangeBasisFunctions(p,pts,n_samples = 101):
    xis = np.linspace(min(pts),max(pts),n_samples)  # Get n_samples evenly-spaced points on the inverval [min(pts), max(pts)] to use for plotting
    fig, ax = plt.subplots()                        # Create a figure/axis object for plotting
    
    # For each a in [0, 1, ..., p], evaluate the a'th a-th 
    # Lagrange basis function of order p at each point xi in xis
    for a in range(0,p+1):                          
        vals = []           # Stores the outputs of the a'th lagrange basis function for all our points in xis
        for xi in xis:
            vals.append(LagrangeBasisEvaluation(p, pts, xi, a))  # Calls our function to evaluate the a'th Lagrange basis function at the point xi
                
        plt.plot(xis,vals)  # Plot the resulting "curve", with the x-values being the equally-spaced points from before, and the y-values being the value of the a'th Lagrange basis function
    ax.grid(linestyle='--') # Add a grid to the plot to make it easier to visualize where the plot gets which values
    
    # CODE I ADDED TO MAKE PLOTS LOOK NICE AND ACTUALLY SHOW UP
    plt.title(f'Lagrange Basis Functions\np = {p}, pts={list(pts)}')
    plt.show()              # Added this so we can actually see the plot



# Interpolate two-dimensional data
def InterpolateFunction(p,pts2D,n_samples = 101):
    # Turn pts2D into an array for easier slicing 
    if not isinstance(pts2D, np.ndarray):
        pts2D = np.array(pts2D)

    # Convert x-coords of pts2D into a list called pts 
    pts = pts2D[:,0]

    # Convert y-coords of pts2D into a list called coefs 
    coeffs = pts2D[:,1]

    # Create an evenly-distributed list of points between 
    # the minimum value of pts and the maximum value of pts 
    # with n_samples points
    xis = np.linspace(np.min(pts), np.max(pts), n_samples)

    # Create an array of zeros of the same dimension as xis
    ys = np.zeros_like(xis)

    # Create a loop that goes through every element in the list xis 
    for i, xi in enumerate(xis):
        # Create a nested for-loop inside the above for-loop that evaluates
        # every Lagrange basis function at the point xi
        # and multiplies the value of this basis function evaluated at xi
        # by the correspoinding coefficient from coeffs 
        for a in range(p+1):
            raw_y = LagrangeBasisEvaluation(p, pts, xi, a)
            y = raw_y * coeffs[a]
            ys[i] += y          # Add this quantity to the ys value corresponding to index xi
        
    # Plot the resulting function 
    fig, ax = plt.subplots()    
    plt.plot(xis,ys)
    plt.scatter(pts, coeffs,color='r')
    plt.title("Lagrange Interpolating Polynomial")  # Added for clarity of what's going on
    plt.show()      # Added to actually show plot
    


if __name__ == "__main__":
    ### -------------- CODE FOR RUNNING PROBLEMS --------------- ###

    # # Problem 2a 
    # p = 1 
    # pts = np.array([-1., 1.])
    # PlotLagrangeBasisFunctions(p, pts)

    # # Problem 2b
    # p = 3 
    # pts = np.array([-1, -1/3, 1/3, 1])
    # PlotLagrangeBasisFunctions(p, pts)

    # # Problem 2c
    # p = 3 
    # pts = np.array([-1, 0, 1/2, 1])
    # PlotLagrangeBasisFunctions(p, pts)

    # Problem 3 
    p = 3
    my2Dpts = [[0,1],[4,6],[6,2],[7,11]]
    InterpolateFunction(p, my2Dpts)

    # # Old Code 
    # mypts = np.linspace(-1,1,4)
    # myaltpts = [-1,0,.5,1]
    # p = 3
    # PlotLagrangeBasisFunctions(p,mypts)