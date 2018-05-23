import cvxpy as cvx
from cvxpy.expressions.expression import Expression
import cvxpy.atoms as atom
from cvxpy.atoms.affine.reshape import reshape
import numpy as np
from itertools import combinations_with_replacement

from numpy.core.multiarray import ndarray
from sympy import *


def homogeneous_monomial_helper(syms, d):
    if d < 0:
        return []
    elif d == 0:
        return np.matrix('1')
    monomials = []
    for tup in combinations_with_replacement(syms, d):
        monomials.append(np.prod(tup))
    return np.matrix(monomials).T


def construct_monomial(syms, d, homogeneous=False):
    # If homogenous, only need to generate monomials with exact degree d
    if homogeneous:
        return homogeneous_monomial_helper(syms, d)
    # Else, we need to generate monomials up to degree d
    else:
        all_monos = []
        for deg in range(d + 1):
            all_monos.append(homogeneous_monomial_helper(syms, deg))
        return np.vstack(all_monos)
    return []


def construct_constraints(syms, poly, z):
    d = len(z)
    qs = symbols('q0:{}'.format(d * d))
    Q = []
    for i in range(d):
        row = []
        for j in range(d):
            row.append(qs[i * d + j])
        Q.append(row)
    Q = np.matrix(Q)
    q_poly = Poly(expand(np.asscalar(z.T * Q * z)), syms)
    monom_coeffs = q_poly.monoms()
    eqns = []
    for monom_coeff in monom_coeffs:
        expr = 1
        for var, exp in zip(syms, monom_coeff):
            expr *= var ** exp
        q_expr = q_poly.coeff_monomial(expr)
        q_val = poly.coeff_monomial(expr)
        eqns.append(q_expr - q_val)
    A, b = linear_eq_to_matrix(eqns, qs)
    A = np.matrix(A).astype(float)
    b = np.matrix(b).astype(float)
    return (A, b)


def is_diagonal(A):
    return np.count_nonzero(A - np.diag(np.diagonal(A))) == 0

"""
def sos_to_sdp(syms, poly, z):
    deg = len(z)
    Q = cvx.Semidef(deg)
    q = cvx.vec(Q)

    A, b = construct_constraints(syms, poly, z)
    constraints = [A * q == b]
    obj = cvx.Minimize(0)

    prob = cvx.Problem(obj, constraints)
    prob.solve()

    Q = np.matrix(Q.value)
    return (prob.status, Q)

"""
#Abiyaz Chowdhury (Barrier method)
def sos_to_sdp(syms, poly, z):
    deg = len(z)
    A, b = construct_constraints(syms, poly, z)
    obj = cvx.Minimize(0)
    q = cvx.Variable(deg*deg,1)
    X = Expression.cast_to_const(q)

    #Find a solution to the equality constraints for setting up the starting strongly feasible point
    constraints = [-A * q == b]
    prob = cvx.Problem(obj, constraints)
    prob.solve()
    Q0 = np.reshape(np.array(q.value),(deg,deg))
    s0 = np.max(np.linalg.eigvals(Q0)) + 1

    #initialize starting points
    Q = cvx.Variable(deg,deg)
    q = cvx.vec(Q)
    s = cvx.Variable()
    Q.value = Q0
    s.value = s0
    m = b.shape[0]
    eps = 0.0001
    t = 0.1
    mu = 1.5
    iteration = 1
    while (s.value > 0) and (m/t > eps):
        obj = cvx.Minimize(s-(1/t)*atom.log_det(s*np.eye(deg)-Q))
        constraints = [-A * q == b, Q == Q.T, q == cvx.vec(Q)]
        prob = cvx.Problem(obj,constraints)
        prob.solve()
        print(Q.value)
        print("Iteration: {} Value of s: {}".format(iteration,s.value))
        t *= mu
        iteration += 1

    return (prob.status, -np.matrix(Q.value))


def sdp_to_sos(Q, z):
    try:
        L = np.linalg.cholesky(Q).T
    except:
        Q += np.eye(len(z)) * 1e-7
        L = np.linalg.cholesky(Q).T
    g = L * z
    g_squared = np.square(g)
    return np.sum(g_squared)


def check_sos(poly):
    if isinstance(poly, int):
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
    z = construct_monomial(syms, deg // 2, poly.is_homogeneous)
    status, Q = sos_to_sdp(syms, poly, z)
    if status == cvx.INFEASIBLE:
        print("Infeasible")
        return None
    sos = sdp_to_sos(Q, z)
    return sos


def drop_epsilon_coeff(poly):
    new_poly = 0
    syms = poly.gens
    for exps, coeff in poly.terms():
        if abs(coeff) > 1e-6:
            expr = 1
            for var, exp in zip(syms, exps):
                expr *= var ** exp
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

    #poly = 1 # edge case SOS
    #poly = x*y # not SOS
    #poly = x**4 + y**4 # simple SOS
    poly = x**2 + 2*x*y + y**2 # SOS (why error?)
    #poly = x**4 + x**2 + 2*x*y + y**2 # SOS (why error?)
    #poly = 4*x**4*y**6 + x**2 - x*y**2 + y**2 # SOS (takes too long)
    #poly = x**4*y**2 + x**2*y**4 - 3*x**2*y**2 + 1 # not SOS but PSD (takes too long)
    #poly = 2 * x ** 4 + 5 * y ** 4 - x ** 2 * y ** 2 + 2 * x ** 3 * y  # is SOS
    print_sos_test(poly)


if __name__ == '__main__':
    main()