# SOS Solver

## Libraries Used
sympy 1.1.1
numpy 1.14.3
cvxpy 0.4.11

## Overview

This is a simple program to test if a polynomial can be recast as a sum of
squares (SOS). In general, testing if a polynimal is non-negative is NP hard, but
we can test if a polynomial can be represented as an SOS in polynomial time.

It is clear that SOS implies non-negative, but the converse is not necessarily
true. However, for several important cases, i.e. univariate, quadratic, the 
converse also holds.

We want to solve (sos_solver.py): `Given p(x), determine if p(x) >= 0 is feasible`

## High Level Code
1. Convert a given SOS\_Program to a semi-definite program

  z is vector of monomials and Q is a symmetric Gram matrix
  Result: p is SOS iff exists Q=Q^T >= 0 such that p = z^TQz

  Given p has n variables and degree 2d
  z has monomials up to degree d using n variables
  z = [1, x1, x2, ..., xn, x1x2, ..., xn^d]
  For homogeneous polynomials, z only need consider monomials with degree exactly d
  We then equate coefficents to get matrix A and vector b
  We need to find Q >= 0 such that Aq=b where q is the columns of Q stacked

2. Solve semi-definite program
Generally, this is done using interior point methods.

3. Convert back to SOS\_Program solution
We have that p = z^T Q z
Once we have a Q and z, we can compute Cholesky factorization Q=V V^T since Q is PSD
p = z^T V^T V z
Then, we have that p(x) = \sum (Vz)\_i^2

4. We can extend, as found in minbound_solver.py, to find lower bounds on functions by solving:
`Maximize \gamma such that f(x) - \gamma >= 0`
