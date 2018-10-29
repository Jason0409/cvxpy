import cvxpy as cvx
# import numpy as np

# Problem data.
n = 4
A = [1, 2, 3, 4]
m1 = [1, 1, 1, 1]
m2 = [1, -1, 1, -1]

# Construct the problem.
x = cvx.Variable(n)
objective = cvx.Minimize(A*x)
constraints = [0 <= x, m1*x == 1, m2*x == 0]
prob = cvx.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
# The optimal value for x is stored in `x.value`.
print("x:", x.value)
print("optimal objective value:", result)
