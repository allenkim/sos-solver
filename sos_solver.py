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

def construct_constraints(syms,poly,z):
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

def is_diagonal(A):
    return np.count_nonzero(A - np.diag(np.diagonal(A))) == 0

def sos_to_sdp(syms,poly,z):
    deg = len(z)
    Q = cvx.Semidef(deg)
    q = cvx.vec(Q)

    A, b = construct_constraints(syms,poly,z)
    constraints = [A*q==b]
    obj = cvx.Minimize(0)

    prob = cvx.Problem(obj, constraints)
    prob.solve()

    Q = np.matrix(Q.value)
    return (prob.status, Q)

def sdp_to_sos(Q,z):
    try:
        L = np.linalg.cholesky(Q).T
    except:
        Q += np.eye(len(z))*1e-7
        L = np.linalg.cholesky(Q).T
    print(L)
    g = L*z
    g_squared = np.square(g)
    return np.sum(g_squared)

def check_sos(poly):
    if isinstance(poly,int):
        if poly >= 0:
            return poly
        else:
            print("Infeasible")
            return None
    poly = Poly(poly).exclude()
    deg = poly.total_degree()
    if deg % 2 == 1:
        print("Infeasible: degree odd")
        return None
    syms = poly.gens
    z = construct_monomial(syms,deg//2,poly.is_homogeneous)
    status, Q = sos_to_sdp(syms,poly,z)
    print(Q)
    if status == cvx.INFEASIBLE:
        print("Infeasible")
        return None
    sos = sdp_to_sos(Q,z)
    return sos

def drop_epsilon_coeff(poly):
    new_poly = 0
    syms = poly.gens
    for exps, coeff in poly.terms():
        if abs(coeff) > 1e-6:
            expr = 1
            for var, exp in zip(syms,exps):
                expr *= var**exp
            new_poly += coeff * expr
    return new_poly

def print_sos_test(poly):
    print(poly)
    sos = check_sos(poly)
    if sos:
        print("SOS: {}".format(sos))
        print("SOS Expanded: {}".format(expand(sos)))
        """
        sos = drop_epsilon_coeff(Poly(sos))
        print("SOS Expanded (cleaned): {}".format(expand(sos)))
        """
 
def main():
    syms = symbols('x y')
    x, y = syms
    
    # poly = 1 # edge case SOS
    # poly = x*y # not SOS
    # poly = x**4 + y**4 # simple SOS
    # poly = x**2 + 2*x*y + y**2 # SOS
    # poly = x**4 + x**2 + 2*x*y + y**2 # SOS
    # poly = 4*x**4*y**6 + x**2 - x*y**2 + y**2 # SOS
    # poly = x**4*y**2 + x**2*y**4 - 3*x**2*y**2 + 1 # not SOS but PSD
    poly = 2*x**4 + 5*y**4 - x**2*y**2 + 2*x**3*y # is SOS
    print_sos_test(poly)
    
if __name__=='__main__':
    main()

