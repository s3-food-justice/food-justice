import data
from solver import solve

if __name__ == '__main__':
    #problem = data.load_problem('../data/simulated_data.txt')
    problem = data.load_problem('../data/goeke-2018/c101C6.txt')
    #solution = data.load_solution('../data/simulated_solution.txt', problem)
    solve(problem)
