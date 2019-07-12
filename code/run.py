import data
from solver import solve
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    #problem = data.load_problem('../data/simulated_data.txt')
    problem = data.load_problem('../data/ourdata.txt')
    for c in problem.cities:
        c.x, c.y = c.y, c.x
    #solution = data.load_solution('../data/simulated_solution.txt', problem)
    plt.figure(1)
    plt.clf()
    ax = plt.axes()
    problem.plot(ax)
    plt.savefig('../initial.eps', format='eps')
    x = solve(problem)
    #data.Solution.from_vector(x[0, :], problem).plot(ax, problem)
    plt.figure(1)
    plt.clf()
    ax = plt.axes()
    _, _, d, e = problem.simulate(data.Solution.from_vector(x[0, :], problem))
    problem.plot(ax, -np.array(d), np.array(e))
    plt.savefig('../final_best_demand.eps', format='eps')
