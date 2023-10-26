import numpy as np
import matplotlib.pyplot as plt
from stlpy_local.systems import LinearSystem
from stlpy_local.STL import LinearPredicate
from stlpy_local.solvers import *



A = np.array([[1.]])
B = np.array([[1.]])
C = np.array([[1.]])
D = np.array([[0.]])
sys = LinearSystem(A, B, C, D)

x0 = np.array([[10]])
T = 10

pi = LinearPredicate(a=[1], b=[2])  # a*y - b > 0

spec = pi.always(0, 5)  # F_[0,5] pi

# Choose a solver
# solver = GurobiMICPSolver(spec, sys, x0, T, robustness_cost=False)
# solver = GurobiMICPSolver(spec, sys, x0, T, robustness_cost=True)
# solver = GurobiMICPSolver_right_hand_full(spec, sys, x0, T)
# solver = GurobiMICPSolver_right_hand(spec, sys, x0, T)
# solver = GurobiMICPSolver_left_hand_full(spec, sys, x0, T)
# solver = GurobiMICPSolver_left_hand(spec, sys, x0, T)
# solver = GurobiMICPSolver_integral_approximate(spec, sys, x0, T)
solver = GurobiMICPSolver_integral_naive(spec, sys, x0, T)


solver.AddQuadraticCost(Q=np.eye(1), R=np.eye(1))
x, u, _, _ = solver.Solve()


plt.plot(np.arange(0, T + 1), x.flatten(), '-', label="y")
plt.plot(np.arange(0, T + 1), u.flatten(), '-', label="u")
plt.legend()
plt.xlabel("timestep (t)")
plt.show()

# robustness_list = solver.getRobustness()
# for i, robustness in enumerate(robustness_list):
#     plt.plot(np.arange(0, T + 1), robustness.flatten(), '-', label=f"robustness_{i}")
# plt.legend()
# plt.xlabel("timestep (t)")
# plt.show()





