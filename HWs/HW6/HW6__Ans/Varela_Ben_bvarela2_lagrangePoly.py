import sys

# Given a set of points, pts, to interpolate
# and a polynomial degree, this function will
# evaluate the a-th basis function at the 
# location xi
def LagrangeBasisEvaluation(p,pts,xi,a):
    pts = sorted(set(pts))
    # ensure valid input
    if (p+1 != len(pts)):
        sys.exit("The number of input points for interpolating must be the same as one plus the polynomial degree for interpolating")

    # complete this code and return an appropriate value
    # as given in Hughes Equation 3.6.1
    L = 1
    for b in range(0,p+1):
        if b != a:
            L *= (xi-pts[b])/(pts[a]-pts[b])
    return L