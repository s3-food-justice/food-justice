from typing import List, Tuple, Dict

from problem import Problem, City, Solution


def load_problem(filename: str) -> Problem:
    with open(filename, 'r') as f:
        lines = f.readlines()
        header = lines[0].split()
        lines_cities = []
        for l in lines[1:]:
            if l.strip() == '':
                break
            lines_cities.append(l)
        cities = load_cities(header, lines_cities)
        return Problem(cities)


def load_cities(h, lines_cities) -> List[City]:
    cities = []
    for l in lines_cities:
        for i, item in enumerate(l.split()):
            if h[i] == 'StringID':
                name = item
            elif h[i] == 'x':
                x = float(item)
            elif h[i] == 'y':
                y = float(item)
            elif h[i] == 'demand':
                balance = int(float(item))
            elif h[i] == 'ReadyTime':
                tmin = int(float(item))
            elif h[i] == 'DueDate':
                tmax = int(float(item))
        # noinspection PyUnboundLocalVariable
        cities.append(City(name, x, y, balance, tmax - tmin))
    return cities


def load_solution(filename: str, problem: Problem) -> Solution:
    with open(filename, 'r') as f:
        lines = f.readlines()
        header = lines[0].split()
        solution_tuples = []
        for l in lines[1:]:
            items = l.split()
            for i, item in enumerate(items):
                if header[i] == 'pointA':
                    a = item
                elif header[i] == 'pointB':
                    b = item
                elif header[i] == 'amount':
                    n = float(item)
            solution_tuples.append((a, b, n))
        return Solution(to_solution_dict(solution_tuples, problem))


def to_solution_dict(tuples: List[Tuple[str, str, float]], problem) -> Dict[
        Tuple[City, City], float]:
    d = dict()
    for t in tuples:
        c1 = problem.get_city(t[0])
        c2 = problem.get_city(t[1])
        d[(c1, c2)] = t[2]
    return d
