import time

import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.nsga2 import nsga2
from pymoo.optimize import minimize
from pymoo.util import plotting
from pymoo.util.non_dominated_sorting import NonDominatedSorting
from pymop.problem import Problem
from problem import FoodDistribution, Solution
import multiprocessing


class SolverProblem(Problem):
    def __init__(self, fd: FoodDistribution):
        n = len(fd.cities) ** 2
        super().__init__(n_var=n, n_obj=2, xl=np.zeros((n,)), xu=np.ones((n,)))
        self.fd = fd

    def compute(self, x):
        solution = Solution.from_vector(x, self.fd)
        d, c, _, _ = self.fd.simulate(solution)
        return d, c

    def _evaluate(self, x, f, *args, **kwargs):
        #demand = np.zeros((x.shape[0],))
        #cost = np.zeros((x.shape[0],))
        #for xx in x:
        #    self.compute(xx)
        with multiprocessing.Pool(4) as p:
            dcs = p.map(self.compute, x)
        f['F'] = np.array(dcs)
        #for i in range(x.shape[0]):
        #    solution = Solution.from_vector(x[i, :], self.fd)
        #    d, c, _, _ = self.fd.simulate(solution)
        #    demand[i] = d
        #    cost[i] = c
        #f['F'] = np.column_stack((demand, cost))


def solve(fd: FoodDistribution):
    problem = SolverProblem(fd)
    method = nsga2(pop_size=70)
    t = time.time()
    print('start')
    res = minimize(problem,
                   method,
                   termination=('n_gen', 20),
                   seed=2,
                   save_history=True,
                   disp=True)
    print('end: ', time.time() - t)
    plt.figure(1)
    plt.clf()
    for g, a in enumerate(res.history):
        a.opt = a.pop.copy()
        a.opt = a.opt[a.opt.collect(lambda ind: ind.feasible)[:, 0]]
        I = NonDominatedSorting().do(a.opt.get("F"),
                                     only_non_dominated_front=True)
        a.opt = a.opt[I]
        X, F, CV, G = a.opt.get("X", "F", "CV", "G")
        plt.figure(1)
        plt.scatter(F[:, 0], F[:, 1], c=[[g / len(res.history), 0, 0]],
                    s=5 ** 2)
        plt.figure(2)
        plt.clf()
        plt.scatter(F[:, 0], F[:, 1],
                    s=5 ** 2)
        plt.xlabel('remaining demand')
        plt.ylabel('cost')
        plt.savefig('../g_{}.eps'.format(g), format='eps')
    plt.figure(1)
    #plotting.plot(res.F, no_fill=True, show=False)
    plt.xlabel('remaining demand')
    plt.ylabel('cost')
    plt.show()
    plt.savefig('../scatter.eps', format='eps')
    return res.X
