#!/usr/bin/env python

##
#
# Set up, solve, and plot the solution for a simple
# reach-avoid problem, where the robot must avoid
# a rectangular obstacle before reaching a rectangular
# goal.
#
##

import numpy as np
import matplotlib.pyplot as plt

from stlpy_local.benchmarks import ReachAvoid
from stlpy_local.solvers import *

# Specification Parameters
goal_bounds = (7,8,8,9)     # (xmin, xmax, ymin, ymax)
obstacle_bounds = (3,5,4,6)
T = 10

# Define the system and specification
scenario = ReachAvoid(goal_bounds, obstacle_bounds, T)
spec = scenario.GetSpecification()
sys = scenario.GetSystem()

# Specify any additional running cost (this helps the numerics in
# a gradient-based method)
Q = 1e-1*np.diag([0,0,1,1])   # just penalize high velocities
R = 1e-1*np.eye(2)

# Initial state
x0 = np.array([1.0,2.0,0,0])

# Choose a solver
# solver = GurobiMICPSolver(spec, sys, x0, T, robustness_cost=False)
# solver = GurobiMICPSolver(spec, sys, x0, T, robustness_cost=True)
solver = GurobiMICPSolver_right_hand(spec, sys, x0, T)
# solver = GurobiMICPSolver_left_hand(spec, sys, x0, T)
# solver = GurobiMICPSolver_integral_approximate(spec, sys, x0, T)
# solver = GurobiMICPSolver_integral_naive(spec, sys, x0, T)

# Set bounds on state and control variables
u_min = np.array([-0.5,-0.5])
u_max = np.array([0.5, 0.5])
x_min = np.array([0.0, 0.0, -1.0, -1.0])
x_max = np.array([10.0, 10.0, 1.0, 1.0])
#solver.AddControlBounds(u_min, u_max)
#solver.AddStateBounds(x_min, x_max)

# Add quadratic running cost (optional)
solver.AddQuadraticCost(Q,R)

# Solve the optimization problem
x, u, _, _ = solver.Solve()

if x is not None:
    # Plot the solution
    ax = plt.gca()
    scenario.add_to_plot(ax)
    plt.scatter(*x[:2,:])
    plt.show()

robustness_list = solver.getRobustness()
for i, robustness in enumerate(robustness_list):
    plt.plot(np.arange(0, T + 1), robustness.flatten(), '-', label=f"robustness_{i}")
plt.legend()
plt.xlabel("timestep (t)")
plt.show()