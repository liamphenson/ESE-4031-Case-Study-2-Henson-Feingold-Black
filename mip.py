import numpy as np
from scipy.optimize import linprog
import collections

"""
- x1, x2 is # of Toy 1, Toy 2
- x3, x4 is binary -- use Factory 1, use Factory 2
- x5, x6 is binary -- make Toy 1, make Toy 2

maximize f = 10x1 + 15x2 - 50000x5 - 80000x6
subject to:
x3 + x4 <= 1 (choose factor 1 or 2)
x1 - Mx5 <= 0 (M is large # to force x1 to 0 when x5 is 0)
x2 - Mx6 <= 0 (M is large # to force x2 to 0 when x6 is 0)
Assume no time lost to switching between making Toy 1 and Toy 2, but only one toy may be produced at a time:
x1/50 + x2/40 - 500x3 <= 0 (50 Toy 1, 40 Toy 2 per hour, 500 hours at Factory 1)
x1/40 + x2/25 - 700x4 <= 0 (40 Toy 1, 25 Toy 2 per hour, 700 hours at Factory 2)
(Only one of those two constraints will be active in the final solution.)
x3, x4, x5, x6 are binary
x1, x2 are integer and geq 0

Note: Using collections to manage the sub-problems per suggestion of Gemini Pro AI for Breadth-first search.
How it works:
1. nodes_to_process = collections.deque([root_node]) -- creates the queue w/ first node
2. current_node = nodes_to_process.popleft() -- in the processing loop (while nodes_to_process)
takes the newest node and does simplex on it
3. nodes_to_process.append(subnode_0/1) adds on the two subnodes (x_i = 0/1) so that those are the next up to solve

SOLUTION:
Optimal Solution Found:
Profit = $-0.00
Produce 0 of Toy 1 (x1)
Produce -0 of Toy 2 (x2)
Use Factory 1 (x3): -0.0
Use Factory 2 (x4): 0.0
Setup for Toy 1 (x5): -0.0
Setup for Toy 2 (x6): -0.0

TL;DR: It's more economical to not make any of the toys due to high setup costs.
"""
M = 1e5
c = np.array([10, 15, 0, 0, -50000, -80000])
binary_idx = [2,3,4,5]
 #Coefficient matrix for inequality constraints
A_ub = np.array([[0, 0, 1, 1, 0,  0],
            [1, 0, 0, 0, -M, 0],
            [0, 1, 0, 0,  0, -M],
            [1/50, 1/40, -500, 0, 0, 0], #Production time at Factory 1
            [1/40, 1/25, 0, -700 ,0, 0], #Production time at Factory 2
            ]
            )
   
    #Inequality constraint vector
b_ub = np.array([1, 0, 0, 0, 0])

bounds = [(0, None), (0, None)] + [(0, 1) for _ in binary_idx]

class Node:
    """Represent a node in the branch-and-bound tree."""
    def __init__(self, A_eq=None, b_eq=None):
        self.A_eq = A_eq if A_eq is not None else np.empty([0, len(c)]) #handle initial node not having any equality constraints
        self.b_eq = b_eq if b_eq is not None else np.empty([0])

def solve_lp_subproblem(node):
    res = linprog([-c_i for c_i in c], A_ub=A_ub, b_ub=b_ub, A_eq = node.A_eq, b_eq = node.b_eq)
    return res

def is_integer(x, tol=1e-5):
    """Checks if the binary variables in the solution are integer-feasible."""
    for i in binary_idx:
        if abs(x[i] - round(x[i])) > tol:
            return False
    return True

def branch_and_bound():
    """Main branch-and-bound algorithm. 
    Stops looking down a branch if:
     - problem is infeasible
     - obj_val is worse than current best_objective val
     - Integer solution found (check for new best_objective val)

     Branch 
    """
    root_node = Node() #no constraints initially
    nodes_to_process = collections.deque([root_node])

    best_objective = -np.inf #store for bounding
    best_solution = None #store for final result
    iteration = 0
 
    while nodes_to_process: #until we've popped off / checked every node of interest , keep processing
        iteration += 1
        current_node = nodes_to_process.popleft()

        print(f"\nIteration {iteration}")
        if current_node.A_eq.shape[0] > 0:
            print("Current Node Constraints")
            for row, b_val in zip(current_node.A_eq, current_node.b_eq):
                idx = np.where(row == 1)[0][0] #get index of variables with equality constraint
                print(f"x{idx+1} = {int(b_val)}")

        res = solve_lp_subproblem(current_node) #lp relaxation at the node

        ############### BOUND

        if not res.success:
            print("Eliminating node b/c infeasible subproblem.")
            continue #jump to the next interation

        objective_val = -res.fun # Need maximization
        solution = res.x
        
        print(f"Objective = {objective_val:.2f}")

        # Eliminate branch if obj val is worse than existing best
        if objective_val < best_objective:
            print(f"Eliminating node (bound {objective_val:.2f} < best known {best_objective:.2f}).")
            continue

        # If it is an integer solution, stop looking down that branch and compare w/ current best we have
        if is_integer(solution):
            print(f"Found an integer-feasible solution with objective {objective_val:.2f}.")
            if objective_val > best_objective:
                best_objective = objective_val
                best_solution = solution
            continue
        
        ################ BRANCH
        print("Status: Solution is not integer-feasible.")
        
        # Find the first binary variable that is non-integer / fractional
        branch_var_index = -1
        for i in binary_idx:
            if abs(solution[i] - round(solution[i])) > 1e-1: #same final results if you decrease tolerance to 1e-5
                branch_var_index = i
                break
        
        print(f"Branching on variable x{branch_var_index+1} (value = {solution[branch_var_index]:.3f})")

        #branch for x_i = 0
        new_constraint_0 = np.zeros(len(c)) #new row for A_eq
        new_constraint_0[branch_var_index] = 1
        
        A_eq_0 = np.vstack([current_node.A_eq, new_constraint_0]) #add row to our existing constraints
        b_eq_0 = np.append(current_node.b_eq, 0)
        subnode_0 = Node(A_eq_0, b_eq_0)
        nodes_to_process.append(subnode_0) #add it to the collection stack
        print(f"Created sub-node with constraint: x{branch_var_index+1} = 0")

        # branch for x_i = 1
        A_eq_1 = np.vstack([current_node.A_eq, new_constraint_0])
        b_eq_1 = np.append(current_node.b_eq, 1)
        subnode_1 = Node(A_eq_1, b_eq_1)
        nodes_to_process.append(subnode_1) #the script will process this node in the next iteration  
        print(f"Created sub-node with constraint: x{branch_var_index+1} = 1")
    
    return best_solution, best_objective


############

final_solution, final_objective = branch_and_bound()

print("Branch-and-Bound Complete")
if final_solution is not None:
    print("\nOptimal Solution Found:")
    print(f"Profit = ${final_objective:,.2f}")
    print(f"Produce {final_solution[0]:.0f} of Toy 1 (x1)")
    print(f"Produce {final_solution[1]:.0f} of Toy 2 (x2)")
    print(f"Use Factory 1 (x3): {final_solution[2]}")
    print(f"Use Factory 2 (x4): {final_solution[3]}")
    print(f"Setup for Toy 1 (x5): {final_solution[4]}")
    print(f"Setup for Toy 2 (x6): {final_solution[5]}")
else:
    print("No integer-feasible solution was found.")



