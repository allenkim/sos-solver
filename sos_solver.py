import cvxpy as cvx
import numpy as np
from itertools import combinations_with_replacement
from sympy import *

def homogeneous_monomial_helper(syms, d):
    if d < 0:
        return []
    elif d == 0:
        return np.matrix('1')
    monomials = []
    for tup in combinations_with_replacement(syms,d):
        monomials.append(np.prod(tup))
    return np.matrix(monomials).T

def construct_monomial(syms, d, homogeneous=False):
    #z = np.matrix([[x*x],[y*y],[x*y]])
    # If homogenous, only need to generate monomials with exact degree d
    if homogeneous:
        return homogeneous_monomial_helper(syms,d)
    # Else, we need to generate monomials up to degree d
    else:
        all_monos = []
        for deg in range(d+1):
            all_monos.append(homogeneous_monomial_helper(syms,deg))
        return np.vstack(all_monos)
    return []

def construct_constraints(z):
    #A = np.matrix('1 0 0 0 0 0 0 0 0; 0 0 0 0 1 0 0 0 0; 0 0 0 0 0 0 0 2 0; 0 0 0 0 0 0 2 0 0; 0 0 0 2 0 0 0 0 1')
    #b = np.matrix('2; 5; 0; 2; -1')
    pass

def solve_sdp(A,b,degree):
    Q = cvx.Semidef(degree)
    q = cvx.vec(Q)

    constraints = [A*q==b]
    obj = cvx.Minimize(0)

    prob = cvx.Problem(obj, constraints)
    prob.solve()
    return (prob.status, Q.value)

def sos_to_sdp(poly):
    pass

def sdp_to_sos(Q,z):
    L = np.linalg.cholesky(Q).T
    g = L*z
    g_squared = np.square(g)
    return np.sum(g_squared)

def check_sos(poly):
    z = construct_monomial([x,y],2,poly.is_homogeneous)

def main():
    x = Symbol('x')
    y = Symbol('y')

    poly = 2*x**4 + 5*y**4 - x**2*y**2 + 2*x**3*y
    check_sos(poly)
    
if __name__=='__main__':
    main()

