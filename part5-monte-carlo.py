import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Part 5: An alternate scenario with an enterprise license.
#
# This problem features an alternate scenario to the original problem presented in Case Study 2.
# Instead of the four license types available from RetailPower, the same basic license as before and
# an enterprise license are available for purchase. The enterprise license costs a one-time fee of $5000
# followed by an additional cost of $log(x) to accommodate x users.
#
# In this script, we will use a Monte Carlo method to approximate a solution to this problem using random
# sampling. We will begin by choosing a feasible integer valued point sampled at uniform from the
# feasible set, then using a variation of the Metropolis-Hastings Algorithm to find a more optimal solution.
#

# Defining the objective function f = 2500x_B0 + 2500x_B1 + 5000x_E0 + 5000x_E1 + log(x_e0) + log(x_e1)
def f_func(x_vec):
    # Function is split up into cases to avoid taking the log of 0
    if x_vec[4] != 0 and x_vec[5] != 0:
        return 2500*x_vec[0] + 2500*x_vec[1] + 5000*x_vec[2] + 5000*x_vec[3] + np.log(x_vec[4]) + np.log(x_vec[5])
    elif x_vec[4] != 0:
        return 2500*x_vec[0] + 2500*x_vec[1] + 5000*x_vec[2] + 5000*x_vec[3] + np.log(x_vec[4])
    elif x_vec[5] != 0:
        return 2500*x_vec[0] + 2500*x_vec[1] + 5000*x_vec[2] + 5000*x_vec[3] + np.log(x_vec[5])
    else:
        return 2500*x_vec[0] + 2500*x_vec[1] + 5000*x_vec[2] + 5000*x_vec[3]

# Part 1: Begin by randomly sampling N points from the feasible set. As noted in Part 5 of the written report
# for this case study, we can place upper bounds of 12 and 355 for the numbers of basic and enterprise license
# purchases respectively.
#
# Define number of points to sample.
N = 10**7

# Randomly sample counts of basic licenses. xB0 denotes basic licenses purchased in January and February and xB1
# denotes licenses purchased in later months. Since there are a total of 355 PCDS employees that need access to
# RetailPower, no more than 12 Basic licenses need to be purchased as each license grants access to 30 users.
xB0 = np.random.randint(low = 0, high = 13, size = N).reshape(-1,1)
xB1 = np.random.randint(low = 0, high = 13, size = N).reshape(-1,1)

# Randomly sample numbers of users on an Enterprise license. xE0 and xE1 denote when an enterprise license was
# purchased and how many users use the license. For xe0 to be nonzero, xE0 must also be nonzero. For xe1 to be
# nonzero, xE0 or xE1 must be nonzero. Since there are 355 employees at PCDS that must be granted access,
# an upper bound of 355 can be placed on xe0 and xe1.
xE0 = np.random.randint(low = 0, high = 2, size = N).reshape(-1,1)
xE1 = np.random.randint(low = 0, high = 2, size = N).reshape(-1,1)
xe0 = np.random.randint(low = 0, high = 356, size = N).reshape(-1,1)
xe1 = np.random.randint(low = 0, high = 356, size = N).reshape(-1,1)

# If an enterprise license hasn't yet been purchased, replace the number of enterprise license users with zero
xe0 = np.where(xE0 == 0, 0, xe0)
xe1 = np.where(np.logical_and(xE0 == 0, xE1 == 0), 0, xe1)

# Concatenate decision variables into single matrix
x = np.concatenate((xB0, xB1, xE0, xE1, xe0, xe1), axis = 1)

# Safely compute log(x_e0) by replacing zeros with ones, then masking out those values by multiplying by x_E0
x4 = x[:,4]
x4 = np.where(x4 == 0, 1, x4)
log_x4 = np.log(x4) * x[:,2]

# Eliminate points that don't comply with constraints
# Constraints: 2500x_B0 + 5000x_E0 + log(x_e0) <= 9500
#              30x_B0 + x_e0 >= 50
#              30x_B0 + 30x_B1 + x_e0 + x_e1 >= 355
#              x_E0 + x_E1 <= 1
#              x_e0 - 355x_E0 <= 0
#              x_e1 - 355x_E0 - 355x_E1 <= 0
constraints = np.logical_and((2500*x[:,0] + 5000*x[:,2] + log_x4 <= 9500), np.logical_and((30*x[:,0] + x[:,4] >= 50), np.logical_and((30*x[:,0] + 30*x[:,1] + x[:,4] + x[:,5] >= 355), np.logical_and((x[:,2] + x[:,3] <= 1), np.logical_and((x[:,4] - 355*x[:,2] <= 0), (x[:,5] - 355*x[:,2] - 355*x[:,3] <= 0))))))
x = x[constraints,:]

# Compute the objective function value for each of the remaining points and choose the point that minimizes the objective function value
# Safely compute log(x_e0)
x4 = x[:,4]
x4 = np.where(x4 == 0, 1, x4)
log_x4 = np.log(x4) * x[:,2]

# Safely compute log(x_e1)
x5 = x[:,5]
x5 = np.where(x5 == 0, 1, x5)
log_x5 = np.log(x5) * np.where(np.logical_or(x[:,2] != 0, x[:,3] != 0), 1, 0)

# Find the objective function value at each point
f = 2500*x[:,0] + 5000*x[:,2] + log_x4 + 2500*x[:,1] + 5000*x[:,3] + log_x5

# Choose the point that minimizes f as the initial solution
min_index = np.argmin(f)
x_init = x[min_index, :]

# Part 2: use a variant of the Metropolis Hastings method to find a more optimal solution. This will be done by
# enumerating a list of perturbations that increase or decrease each decision variable by 1 which we will
# randomly choose from. After choosing a perturbation and applying it to the current point, our solution is
# updated to the new point with probability p where p is the probability that the new point is more optimal
# function value than the previous point. p is defined as min((exp(-f(x_prime)))/(exp(-f(x_t))),1) where x_prime
# is the new point being tested and x_t is the current optimal solution.

# Define number of iterations and possible perturbations
num_its = 10**5
perturbations = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [-1,0,0,0,0,0], [0,-1,0,0,0,0]])

# Add additional perturbations to x_e0 and x_e1 if an appropriate enterprise license is purchased
if x_init[2] == 1:
    perturbations = np.concatenate((perturbations, np.array([[0,0,0,0,-1,0], [0,0,0,0,0,-1], [0,0,0,0,1,0], [0,0,0,0,0,1]])), axis = 0)
elif x_init[3] == 1:
   perturbations = np.concatenate((perturbations, np.array([[0,0,0,0,0,-1], [0,0,0,0,0,1]])), axis = 0)

# Run the algorithm
x_t = x_init
for i in range(num_its):
    # Randomly choose a perturbation from the list
    (N,_) = perturbations.shape
    random_idx = np.random.choice(np.arange(N), size = 1, replace = False)
    random_perturbation = perturbations[random_idx,:]

    # Compute the perturbed minimizer and evaluate whether to move to it
    x_prime = x_t + random_perturbation

    # Safely compute log(x_prime_e0)
    x_prime4 = x_prime[:,4]
    x_prime4 = np.where(x_prime4 == 0, 1, x_prime4)
    log_x_prime4 = np.log(x_prime4) * x_prime[:,2]

    # Check if x_prime complies with constraints
    constraints = np.logical_and((2500*x_prime[:,0] + 5000*x_prime[:,2] + log_x_prime4 <= 9500), np.logical_and((30*x_prime[:,0] + x_prime[:,4] >= 50), np.logical_and((30*x_prime[:,0] + 30*x_prime[:,1] + x_prime[:,4] + x_prime[:,5] >= 355), np.logical_and((x_prime[:,2] + x_prime[:,3] <= 1), np.logical_and((x_prime[:,4] - 355*x_prime[:,2] <= 0), (x_prime[:,5] - 355*x_prime[:,2] - 355*x_prime[:,3] <= 0))))))
    x_prime = (x_prime[constraints,:])

    # If x_prime is feasible, check if we should move to it
    if(x_prime.size != 0):
        # Compute probability of moving to p
        p = np.min(np.array([(np.exp(-f_func(x_prime[0,:])))/(np.exp(-f_func(x_t))),1]))

        #Generate random value alpha at uniform from [0,1), if alpha < p, then move
        alpha = np.random.rand()
        if alpha < p:
            x_t = x_prime

# Compute the cost of our optimal solution
f_star = f_func(x_t)

if x_t[2] != 0:
    # If x_E0 = 1, then an enterprise license will be purchased in January or February
    results = f"""
    We recommend for PCDS to purchase the following RetailPower licenses:
        {x_t[0]} basic licenses in January and February.
        {x_t[1]} basic licenses in March, April, and May.
        An enterprise license in January or February with {x_t[4] + x_t[5]} users.
    The total cost of these licenses is ${f_star:.2f}
    """
elif x_t[3] != 0:
    # If x_E1 = 1, then an enterprise license will be purchased in March, April, or May
    results = f"""
    We recommend for PCDS to purchase the following RetailPower licenses:
        {x_t[0]} basic licenses in January and February.
        {x_t[1]} basic licenses in March, April, and May.
        An enterprise license in March, April, or May with {x_t[5]} users.
    The total cost of these licenses is ${f_star:.2f}
    """
else:
    # If x_E0 and x_E1 are both zero, only basic licenses will be purchased
    results = f"""
    We recommend for PCDS to purchase the following RetailPower licenses:
        {x_t[0]} basic licenses in January and February.
        {x_t[1]} basic licenses in March, April, and May.
    The total cost of these licenses is ${f_star:.2f}
    """

print(results)

# Sample a random subset of 200 of the randomly drawn points for plotting
(N_final,_) = x.shape
random_choices = np.random.choice(np.arange(N_final), size = 1000, replace = False)
x_sub = x[random_choices,:]
basic_counts = x_sub[:,0] + x_sub[:,1]
enterprise_counts = x_sub[:,4] + x_sub[:,5]

# Safely compute log(x_e0)
x_sub4 = x_sub[:,4]
x_sub4 = np.where(x_sub4 == 0, 1, x_sub4)
log_x4 = np.log(x_sub4) * x_sub[:,2]

# Safely compute log(x_e1)
x_sub5 = x_sub[:,5]
x_sub5 = np.where(x_sub5 == 0, 1, x_sub5)
log_x5 = np.log(x_sub5) * np.where(np.logical_or(x_sub[:,2] != 0, x_sub[:,3] != 0), 1, 0)

# Find the objective function value at each point
f = 2500*x_sub[:,0] + 5000*x_sub[:,2] + log_x4 + 2500*x_sub[:,1] + 5000*x_sub[:,3] + log_x5

# Create a 3-D plot of the randomly drawn data points, showing the numbers of users for each license type and the objective function value of each point
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(basic_counts, enterprise_counts, f, c=f, cmap='viridis', label="Random Samples", alpha=0.5)
ax.scatter(x_t[0] + x_t[1], x_t[4] + x_t[5], f_star, c='red', label="Final Minimizer", alpha=1)
ax.set_xlabel("Basic Licenses")
ax.set_ylabel("Enterprise License Users")
ax.set_zlabel("Cost")
ax.set_title("Plot of Random Samples and Minimizer")
plt.colorbar(scatter, label="Cost", location='left')
plt.legend()
plt.show()
