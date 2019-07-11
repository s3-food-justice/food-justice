import data


if __name__ == '__main__':
    problem = data.load_problem('../data/simulated_data.txt')
    solution = data.load_solution('../data/simulated_solution.txt', problem)
    problem.simulate(solution)
    print(problem)
