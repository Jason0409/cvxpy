import matplotlib.pyplot as plt
import cvxpy as cvx
import numpy as np
import pickle

# Problem data.
X = pickle.load(open('asgn5q2.pkl', 'rb'))
m, n = X.shape
sigmaHat = np.cov(X, rowvar=0)

# non-zero array
non_zeros = np.zeros(20)
# objective array
objectives = np.zeros(20)
# ML array
mls = np.zeros(20)

# off-diagonal matrix
off_dia = 1-np.eye(n, n)
v_dia = np.ones(n)
# values of lambda
lambdas = np.logspace(-6, 1, num = 20)
# print(lambdas)

# Construct the problem.
for i in range(len(lambdas)):
    K = cvx.Variable((n, n), PSD=True)
    # K = cvx.Variable((n, n), symmetric=True)
    objective = cvx.Minimize(-cvx.log_det(K) +
                             cvx.trace(sigmaHat*K) +
                             lambdas[i]*v_dia * cvx.multiply(off_dia, cvx.abs(K)) * v_dia.T)
    constraints = []
    # constraints = [K >> 0]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve()

    non_zeros[i] = np.sum(K.value >= 1e-6)
    objectives[i] = result
    mls[i] = np.log(np.linalg.det(K.value)) - np.trace(np.dot(sigmaHat, K.value))


# show the graph
plt.figure(num=3, figsize=(10,10))
plt.title("number of non-zero,optimal obj,mls")
plt.grid(linestyle = "-.")
plt.xlabel("Lambda")
plt.ylabel("Value")
plt.plot(lambdas, non_zeros, color='blue', linewidth=2.0, linestyle='--', label='non_zero')
plt.plot(lambdas, objectives, color='red', linewidth=2.0, label='objectives')
plt.plot(lambdas, mls, color='green', linewidth=2.0, label='mls')
plt.legend(loc='upper right')
plt.show()

