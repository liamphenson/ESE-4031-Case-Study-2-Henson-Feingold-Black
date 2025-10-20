import numpy as np
from scipy.optimize import linprog

#Supplemental code for intermediate steps for Part I (in the month of March)
#Uses SciPy's linprog method to solve intermediate linear programming problems

#Coefficients of objective function
c = np.array([2500, 5000, 10000, 25000])

#Coefficient matrix for inequality constraints
A = np.array([[-30, -80, -200, -2000], [0, -1, -1, -1]])

#RHS of constraints
b = np.array([-50, -1])

#Bounds (upper branch after branching at x_P with k=0: set x_P=0)
bounds = np.array([(0, None), (0, 0), (0, None), (0, None)])

#Obtaining and reporting the result
res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)

x_star = res.x
cost = res.fun

print(f"Optimal solution: {x_star}")
print(f"Minimum cost: {cost}")