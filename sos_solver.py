"""
Testing if a polynimal is nonegative is generally hard, but we can recast it as SOS

Given p(x), determine if p(x) >= 0 is feasible

High Level Code:

1. Define symbolic variables - say x1 x2
vartable = [x1, x2]

2. Initialize sum of squares program
sosprog = SOS_Program(vartable)

3. We want to define p(x1,x2) >= 0, ineq assume >= 0
p = 2*x1^4 + 2*x1^3*x2 - x1^2*x2^2 + 5*x2^4
sosprog.ineq(p)

4. We solve
sosprog.solve()
"""

"""
Steps required to have functionality
1. Convert given SOS_Program to a semi-definite program

z is vector of monomials and Q is symmetric Gram matrix
Result: p is SOS iff exists Q=Q^T >= 0 such that p = z^TQz
given p has n variables and degree 2d
z has monomials up to degree d using n variables
z = [1, x1, x2, ..., xn, x1x2, ..., xn^d]
for homogeneous polynomials, z only need consider monomials with degree exactly d
equate coefficents to get matrix A and vector b
We need to find Q >= 0 such that Aq=b where q is the columns of Q stacked

2. Solve semi-definite program
Interior point methods

3. Convert back to SOS_Program solution
Since p = z^TQz
Once we have a Q and z, we can compute Cholesky factorization Q=V^T V since Q is PSD
p = z^T V V^T z
Then, we have that p(x) = \sum (Vz)_i^2

4*. (Optional) provide certificate of non-sos by dual semidefinite program
"""


