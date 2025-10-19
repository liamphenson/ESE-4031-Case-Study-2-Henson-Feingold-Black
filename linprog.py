import numpy as np
from scipy.optimize import linprog

#Coefficients of objective function
c = np.array([2500, 5000, 9000, 25000])

#Coefficient matrix for inequality constraints
A = np.array([[-30, -80, -200, -2000], [2500, 5000, 9000, 25000]])

#RHS of constraints
b = np.array([-50, 9500])

#Bounds
bounds = np.array([(2, None), (0, 0), (0, 0), (0, 0)])

#Obtaining and reporting the result
res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)

x_star = res.x
cost = res.fun

print(x_star)
print(cost)