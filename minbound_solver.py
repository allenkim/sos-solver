import cvxpy as cvx
import numpy as np
import time
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

def construct_constraints_min(syms,poly,z):
    d = len(z)
    qs = symbols('q0:{}'.format(d*d))
    Q = []
    for i in range(d):
        row = []
        for j in range(d):
            row.append(qs[i*d+j])
        Q.append(row)
    Q = np.matrix(Q)
    q_poly = Poly(expand(np.asscalar(z.T * Q * z)),syms)
    monom_coeffs = q_poly.monoms()
    eqns = []
    for monom_coeff in monom_coeffs:
        expr = 1
        for var, exp in zip(syms, monom_coeff):
            expr *= var**exp
        q_expr = q_poly.coeff_monomial(expr)
        q_val = poly.coeff_monomial(expr)
        eqns.append(q_expr-q_val)
    A,b = linear_eq_to_matrix(eqns, qs)
    A = np.matrix(A).astype(float)
    b = np.matrix(b).astype(float)
    return (A,b)


def sos_to_sdp_min(syms,poly,z):
    deg = len(z)
    Q = cvx.Semidef(deg)
    q = cvx.vec(Q)
    gamma = cvx.Variable()

    A, b = construct_constraints_min(syms,poly,z)
    const_val = np.asscalar(b[-1]) - gamma
    A = A[:-1]
    b = b[:-1]
    constraints = [A*q==b, q[0] == const_val]
    obj = cvx.Maximize(gamma)

    prob = cvx.Problem(obj, constraints)
    opt_val = prob.solve(solver=cvx.SCS)

    Q = np.matrix(Q.value)
    return (prob.status, opt_val, Q)

def check_bound(poly):
    if isinstance(poly,int):
        return poly
    poly = Poly(poly)
    deg = poly.total_degree()
    if deg % 2 == 1:
        print("Infeasible: degree odd")
        return None
    syms = poly.gens
    z = construct_monomial(syms,deg//2,False)
    status, opt_val, Q = sos_to_sdp_min(syms,poly,z)
    if status == cvx.INFEASIBLE:
        print("Infeasible")
        return None
    return opt_val

def print_min_test(poly):
    print(poly)
    start = time.time()
    sos = check_bound(poly)
    end = time.time()
    print(end-start)
    if sos:
        print("Min Val: {}".format(sos))
 

def main():
    syms = symbols('x y')
    x, y = syms
    
    f1 = x+y+1
    f2 = 19-14*x+3*x**2-14*y+6*x*y+3*y**2
    f3 = 2*x-3*y
    f4 = 18-32*x+12*x**2+48*y-36*x*y+27*y**2
    poly = (1+f1**2*f2)*(30+f3**2*f4)
    # poly = 4*x**2 - 2.1*x**4 + (1/3)*x**6 + x*y - 4*y**2 + 4*y**4
    print_min_test(poly)
    
if __name__=='__main__':
    main()

