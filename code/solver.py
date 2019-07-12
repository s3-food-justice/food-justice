import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.nsga2 import nsga2
from pymoo.optimize import minimize
from pymoo.util import plotting
from pymop.problem import Problem
from problem import FoodDistribution, Solution


class SolverProblem(Problem):
    def __init__(self, fd: FoodDistribution):
        n = len(fd.cities) ** 2
        super().__init__(n_var=n, n_obj=2, xl=np.zeros((n,)), xu=np.ones((n,)))
        self.fd = fd

    def _evaluate(self, x, f, *args, **kwargs):
        demand = np.zeros((x.shape[0],))
        cost = np.zeros((x.shape[0],))
        for i in range(x.shape[0]):
            solution = Solution.from_vector(x[i, :], self.fd)
            d, c = self.fd.simulate(solution)
            demand[i] = d
            cost[i] = c
        f['F'] = np.column_stack((demand, cost))


def solve(fd: FoodDistribution):
    problem = SolverProblem(fd)
    method = nsga2(pop_size=10)
    res = minimize(problem,
                   method,
                   termination=('n_gen', 20),
                   seed=1,
                   save_history=True,
                   disp=True)
    plotting.plot(res.F, no_fill=True, show=False)
    plt.xlabel('remaining demand')
    plt.ylabel('cost')
    plt.show()
    return res.X
