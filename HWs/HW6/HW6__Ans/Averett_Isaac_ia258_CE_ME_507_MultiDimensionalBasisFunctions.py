import sys
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt

# Lagrange polynomial basis function
def LagrangeBasisEvaluation(p, pts, xi, a):
    # ensure valid input
    if (p + 1 != len(pts)):
        sys.exit("The number of input points for interpolating must be the same as one plus the polynomial degree for interpolating")
    else:
        value = 1
        for m in range(0, p + 1):
            if m != a:
                value *= (xi - pts[m]) / (pts[a] - pts[m])
        return value

# Convert single index A into a multi-index (a0, a1, ...)
def convert_single_index_to_multi_index(A, degs):
    multi_index = []
    nsd = len(degs)  # Number of spatial dimensions
    for i in range(nsd):
        pi_plus_1 = degs[i] + 1  # p0 + 1, p1 + 1, etc.
        ai = A % pi_plus_1  # Compute a_i as A mod (p_i + 1)
        multi_index.append(ai)
        A = A // pi_plus_1  # Update A for the next index computation
    return multi_index

# Higher-dimensional basis function with multi-index
def MultiDimensionalBasisFunctionIdxs(idxs, degs, interp_pts, xis):
    result = 1.0
    for i in range(len(degs)):
        result *= LagrangeBasisEvaluation(degs[i], interp_pts[i], xis[i], idxs[i])
    return result

# Higher-dimensional basis function with single index
def MultiDimensionalBasisFunction(A, degs, interp_pts, xis):
    multi_idx = []
    for i in range(len(degs)):
        multi_idx.append(A % degs[i])
        A = A // degs[i]
    
    val = 1
    for i in range(0, len(multi_idx)):
        val *= LagrangeBasisEvaluation(degs[i], interp_pts[i], xis[i], multi_idx[i])
    return val

# Plot of 2D basis functions with A as a single index
def PlotTwoDimensionalParentBasisFunction(A, degs, npts=101, contours=True):
    interp_pts = [np.linspace(-1, 1, degs[i] + 1) for i in range(0, len(degs))]
    xivals = np.linspace(-1, 1, npts)
    etavals = np.linspace(-1, 1, npts)
    
    Xi, Eta = np.meshgrid(xivals, etavals)
    Z = np.zeros(Xi.shape)
    
    for i in range(0, len(xivals)):
        for j in range(0, len(etavals)):
            Z[i, j] = MultiDimensionalBasisFunction(A, degs, interp_pts, [xivals[i], etavals[j]])
    
    # Contour plot
    if contours:
        fig, ax = plt.subplots()
        surf = ax.contourf(Eta, Xi, Z, levels=100, cmap=matplotlib.cm.jet)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        fig.colorbar(surf)
        plt.show()
    # 3D surface plot
    else:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(Eta, Xi, Z, cmap=matplotlib.cm.jet,
                               linewidth=0, antialiased=False)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        ax.set_zlabel(r"$N(\xi,\eta)$")
        plt.show()

# Plot for degree [1,1]
PlotTwoDimensionalParentBasisFunction(A=2, degs=[2, 1])

# Plot for degree [2,1]
PlotTwoDimensionalParentBasisFunction(A=2, degs=[2, 1])

# Plot for degree [2,2]
PlotTwoDimensionalParentBasisFunction(A=2, degs=[2, 2])

# Plot for degree [3,3]
PlotTwoDimensionalParentBasisFunction(A=2, degs=[3, 3])
