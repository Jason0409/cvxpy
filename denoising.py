# Python
import cvxpy as cvx
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Problem data.
x_corr = pickle.load(open('asgn5q3.pkl', 'rb')).reshape(-1)
n = len(x_corr)

# non-zero array
error = np.zeros(50)
# objective array
objectives = np.zeros(50)
# ML array
regular = np.zeros(50)

# values of lambda
lambdas = np.logspace(-5, 2, num = 50)

# Construct the problem
for i in range(len(lambdas)):
    x = cvx.Variable(n)
    objective = cvx.Minimize(cvx.square(cvx.norm(x-x_corr)) +
                             lambdas[i]*cvx.norm(cvx.diff(x), 1))
    constraints = []
    prob = cvx.Problem(objective, constraints)
    result = prob.solve()

    error[i] = np.linalg.norm(x.value - x_corr)
    objectives[i] = result
    regular[i] = np.linalg.norm(np.diff(x.value), 1)

# show the graph
plt.figure(figsize=(10,10))
plt.title("trade-off")
plt.grid(linestyle = "-.")
plt.xlabel("norm2")
plt.ylabel("norm1")
plt.plot(error, regular, color='blue', linewidth=2.0)
plt.show()

# Construct the problem
x = cvx.Variable(n)
objective = cvx.Minimize(cvx.norm(x-x_corr) + 0.24*cvx.norm(cvx.diff(x), 1))
constraints = []
prob = cvx.Problem(objective, constraints)
result = prob.solve()

# show the graph
plt.figure(num=2,figsize=(10,10))
plt.title("comparision")
plt.grid(linestyle = "-.")
plt.plot(x_corr, linewidth=2,label= 'original')
plt.plot(x.value,color = 'red',linewidth=2,label='denoised')
plt.legend(loc='upper right')
plt.show()