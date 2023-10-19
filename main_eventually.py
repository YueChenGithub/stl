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

x0 = np.array([[0]])
t_0 = 0
t_end = 20

pi = LinearPredicate(a=[1], b=[2])  # a*y - b > 0

spec = pi.eventually(0, 5)  # F_[0,5] pi

solver = GurobiMICPSolver(spec, sys, x0, t_end, robustness_cost=False)
# solver = GurobiMICPSolver(spec, sys, x0, t_end, robustness_cost=True)
# solver = GurobiMICPSolver_time(spec, sys, x0, t_end)
# solver = GurobiMICPSolver_time_reduced(spec, sys, x0, t_end)


solver.AddQuadraticCost(Q=np.eye(1), R=np.eye(1))
x, u, _, _ = solver.Solve()
# solvers.AddSpatialCost(spec)

# zt, ct1, ct0, chara = solver.get_variable()


plt.plot(np.arange(t_0, t_end + 1), x.flatten(), '-', label="y")
plt.plot(np.arange(t_0, t_end + 1), u.flatten(), '-', label="u")
# plt.plot(np.arange(t_0, t_end + 1), zt.flatten(), '-', label="zt")
# plt.plot(np.arange(t_0, t_end + 1), ct1.flatten()[0:-1], '-', label="ct1")
# plt.plot(np.arange(t_0, t_end + 1), ct0.flatten()[0:-1], '-', label="ct0")
# plt.plot(np.arange(t_0, t_end + 1), chara.flatten(), '-', label="chara")
plt.legend()
plt.xlabel("timestep (t)")
# plt.ylabel("output signal (y)")
plt.show()


