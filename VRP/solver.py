"""Code file for vehicle routing problem created for Advanced Algorithms
Spring 2020 at Olin College. These functions solve the vehicle routing problem
using an integer programming and then a local search approach. This code has
been adapted from functions written by Alice Paul."""

import picos as pic
import numpy as np
from read_files import read_file_type_A, read_file_type_C

# Integer programming approach
def cvrp_ip(C, q, K, Q, obj=True):
    """
    Solves the capacitated vehicle routing problem using an integer programming
    approach.

    C: matrix of edge costs, that represent distances between each node
    q: list of demands associated with each client node
    K: number of vehicles
    Q: capacity of each vehicle
    obj: whether to set objective (ignore unless you are doing local search)
    returns:
        objective_value: value of the minimum travel cost
        x: matrix representing number of routes that use each arc
    """

    # np.append returns a copy, so we can do this w/o side effects outside this function
    C = np.append(C, (C[0])[np.newaxis].T, 1)  # Add a column
    C = np.append(C, [C[0]], 0)  # Add a row

    q = np.append(np.array(q), [0])  # Add a 0 for the destination node

    # set up the picos problem
    prob = pic.Problem()

    x = pic.BinaryVariable("x", C.shape)
    u = pic.RealVariable("u", C.shape[0])
    obj = pic.RealVariable("obj")

    prob.add_constraint(sum(x[-1, :]) == 0)
    prob.add_constraint(sum(x[:, 0]) == 0)

    prob.add_constraint(sum(x[0, :]) <= K)
    prob.add_constraint(sum(x[:, -1]) <= K)
    prob.add_constraint(sum(x[0, :]) == sum(x[:, -1]))

    for i in range(1, C.shape[0] - 1):
        prob.add_constraint(sum(x[i, :]) == 1)
        prob.add_constraint(sum(x[:, i]) == 1)

    for i in range(C.shape[0]):
        prob.add_constraint(q[i] <= u[i])
        prob.add_constraint(u[i] <= Q)
        for j in range(C.shape[0]):
            prob.add_constraint(u[i] - u[j] + (Q * x[i, j]) <= Q - q[j])

    prob.add_constraint(obj == sum(C ^ x))

    prob.set_objective("min", obj)
    soln = prob.solve()

    return obj.value, x.value


# Local search approach (OPTIONAL)
def local_search(C, q, K, Q):
    """
    Solves the capacitated vehicle routing problem using a local search
    approach.

    C: matrix of edge costs, that represent distances between each node
    q: list of demands associated with each client node
    K: number of vehicles
    Q: capacity of each vehicle
    returns:
        bestval: value of the minimum travel cost
        bestx: matrix representing number of routes that use each arc
    """
    bestx = []
    bestval = 0

    # TODO (OPTIONAL): implement local search to solve vehicle routing problem

    return bestval, bestx


if __name__ == "__main__":

    # an example call to test your integer programming implementation
    C, q, K, Q = read_file_type_A("data/A-n05-k04.xml")
    travel_cost, x = cvrp_ip(C, q, K, Q)
    print("Travel cost: " + str(travel_cost))
