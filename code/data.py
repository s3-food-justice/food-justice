from problem import Problem, City


def load_problem(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        i = 0
        header = {h: i for i, h in enumerate(lines[0].split())}
        inv_header = lines[0].split()
        lines_cities = []
        for l in lines[1:]:
            if l.strip() == '':
                break
            lines_cities.append(l)
        cities = load_cities(inv_header, lines_cities)
        return Problem(cities)


def load_cities(h, lines_cities):
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


def load_solution(filename):
    pass
