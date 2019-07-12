from typing import List, Tuple, Dict

from problem import FoodDistribution, City, Solution


def load_problem(filename: str) -> FoodDistribution:
    with open(filename, 'r') as f:
        lines = f.readlines()
        header = lines[0].split()
        lines_cities = []
        for l in lines[1:]:
            if l.strip() == '':
                break
            lines_cities.append(l)
        cities = load_cities(header, lines_cities)
        return FoodDistribution(cities)


# noinspection PyUnboundLocalVariable
def load_cities(h, lines_cities) -> List[City]:
    cities = dict()
    n = 0
    for l in lines_cities:
        skip = False
        for i, item in enumerate(l.split()):
            if h[i] == 'StringID':
                name = item
            elif h[i] == 'Type':
                if item in ['d', 'f']:
                    skip = True
                    break
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
        if not skip:
            if (x, y) in cities:
                cities[(x, y)][1].balance += balance
            else:
                cities[(x, y)] = (n, City(name, x, y, balance, tmax - tmin))
            n += 1
    return [c for _, c in sorted(cities.values(), key=lambda ic: ic[0])]


def load_solution(filename: str, problem: FoodDistribution) -> Solution:
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
