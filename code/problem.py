from dataclasses import dataclass, field
from heapq import heappush, heappop
from math import sqrt
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np


class City:
    def __init__(self, name: str, x: float, y: float, balance: int, time: int):
        self.name = name
        self.x = x
        self.y = y
        self.balance = balance
        self.time = time

    def __eq__(self, o: object) -> bool:
        return isinstance(o, City) and self.name == o.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self):
        return 'City(name={}, x={}, y={}, balance={}, time={})'.format(
            self.name, self.x, self.y, self.balance, self.time
        )


class Solution:
    def __init__(self, solution: Dict[Tuple[City, City], float]):
        self.solution = solution

    @staticmethod
    def from_vector(x, fd):
        sol = dict()
        for j, c1 in enumerate(fd.cities):
            for k, c2 in enumerate(fd.cities):
                if j == k:
                    continue
                idx = j * len(fd.cities) + k
                sol[(fd.cities[j], fd.cities[k])] = x[idx]
        return Solution(sol)

    def get_amount(self, city_a: City, city_b: City):
        if (city_a, city_b) in self.solution:
            key = city_a, city_b
        else:
            raise ValueError()
        return self.solution[key]

    def plot(self, ax: plt.Axes, fd):
        for c1 in fd.cities:
            for c2 in fd.cities:
                if c1 is c2:
                    continue
                ax.plot([c1.x, c2.x], [c1.y, c2.y], '-k',
                        lw=self.get_amount(c1, c2) * 10,
                        zorder=0)


@dataclass(order=True)
class Package:
    amount: int = field(compare=False)
    source: int = field(compare=False)
    destination: int = field(compare=False)
    distance: float = field(compare=True)
    expiration: float = field(compare=True)


@dataclass(order=True)
class Balance:
    amount: float = field(compare=False)
    expiration: float = field(compare=True)


class FoodDistribution:
    def __init__(self, cities: List[City]):
        self.cities = cities
        self.distances = [[dist(c1, c2) for c2 in self.cities]
                          for c1 in self.cities]
        self.speed = 2

    def get_city(self, name: str) -> Optional[City]:
        for c in self.cities:
            if c.name == name:
                return c
        return None

    def plot(self, ax: plt.Axes):
        balances = np.array([c.balance for c in self.cities])
        sizes = np.zeros((len(self.cities),))
        colors = np.zeros((len(self.cities), 3))
        sizes[balances != 0] = 0.3 * balances[balances != 0] ** 2
        sizes[balances == 0] = 10 ** 2
        colors[balances < 0, 0] = 1
        colors[balances > 0, 1] = 1
        colors[balances == 0, 2] = 1
        ax.scatter(
            x=[c.x for c in self.cities],
            y=[c.y for c in self.cities],
            s=sizes,
            c=colors,
            zorder=1
        )

    def simulate(self, solution: Solution):
        print('sim...', end='')
        ratios = self.compute_ratios(solution)
        excesses = []
        demands = []
        for c in self.cities:
            if c.balance > 0:
                excesses.append([Balance(c.balance, c.time)])
                demands.append(0)
            else:
                excesses.append([])
                demands.append(-c.balance)
        total_demand = sum(demands)
        total_cost = 0
        packages = []
        self.dispatch(packages, ratios, excesses)
        while True:
            arrived, cost = self.move(packages)
            total_cost += cost
            if arrived is None:
                break
            dest = arrived.destination
            if arrived.amount <= demands[dest]:
                demands[dest] -= arrived.amount
                total_demand -= arrived.amount
            else:
                heappush(excesses[dest], Balance(arrived.amount - demands[dest],
                                                 arrived.expiration))
                total_demand -= demands[dest]
                demands[arrived.destination] = 0
            if total_demand <= 0:
                break
            self.dispatch(packages, ratios, excesses)
        print('end')
        return total_demand, total_cost

    def compute_ratios(self, solution):
        ratios = []
        for i1, c1 in enumerate(self.cities):
            total = 0
            for i2, c2 in enumerate(self.cities):
                if c1 is c2:
                    continue
                total += solution.get_amount(c1, c2)
            r = []
            for i2, c2 in enumerate(self.cities):
                if c1 is c2 or total == 0:
                    r.append(0)
                else:
                    r.append(solution.get_amount(c1, c2) / total)
            ratios.append(r)
        return ratios

    def dispatch(self, packages, ratios, balances):
        for i1, c1 in enumerate(self.cities):
            if not balances[i1]:
                continue
            while balances[i1]:
                balance = heappop(balances[i1])
                left = balance.amount
                for i2, c2 in enumerate(self.cities):
                    r = ratios[i1][i2]
                    if r == 0:
                        continue
                    amount = balance.amount * r
                    if left - amount > 1e-6:
                        left -= amount
                    else:
                        amount = left
                        left = 0
                    package = Package(amount, i1, i2,
                                      self.distances[i1][i2],
                                      balance.expiration)
                    heappush(packages, package)
                if left > 0:
                    balance.amount = left
                    heappush(balances[i1], balance)

    def move(self, packages: List[Package]) -> Tuple[Optional[Package], float]:
        if not packages:
            return None, 0
        cost = 0
        d = packages[0].distance
        for p in packages:
            p.distance -= d
            p.expiration -= d / self.speed
            cost += d * p.amount
        head = heappop(packages)
        if head.expiration < 0:
            head = None
        else:
            return head, cost
        while packages:
            d = packages[0].distance
            for p in packages:
                p.distance -= d
                p.expiration -= d / self.speed
                cost += d * p.amount
            head = heappop(packages)
            if head.expiration < 0:
                head = None
        return head, cost


def dist(c1: City, c2: City) -> float:
    return sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)
