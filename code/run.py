import data
from solver import solve
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #problem = data.load_problem('../data/simulated_data.txt')
    problem = data.load_problem('../data/goeke-2018/c101C6.txt')
    #solution = data.load_solution('../data/simulated_solution.txt', problem)
    x = solve(problem)
    ax = plt.axes()
    data.Solution.from_vector(x[0, :], problem).plot(ax, problem)
    problem.plot(ax)
    plt.show()
