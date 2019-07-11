from dataclasses import dataclass, field
from heapq import heappush, heappop
from math import sqrt
from typing import List, Dict, Tuple, Optional


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


class Solution:
    def __init__(self, solution: Dict[Tuple[City, City], float]):
        self.solution = solution

    def get_amount(self, city_a: City, city_b: City):
        if (city_a, city_b) in self.solution:
            key = city_a, city_b
        else:
            raise ValueError()
        return self.solution[key]


class Problem:
    @dataclass(order=True)
    class Package:
        amount: int = field(compare=False)
        destination: int = field(compare=False)
        distance: float = field(compare=True)
        expiration: float = field(compare=True)

    @dataclass(order=True)
    class Balance:
        amount: float = field(compare=False)
        expiration: float = field(compare=True)

    def __init__(self, cities: List[City]):
        self.cities = cities
        self.distances = [[dist(c1, c2) for c2 in self.cities]
                          for c1 in self.cities]
        self.speed = 10

    def get_city(self, name: str) -> Optional[City]:
        for c in self.cities:
            if c.name == name:
                return c
        return None

    def simulate(self, solution: Solution):
        ratios = self.compute_ratios(solution)
        excesses = []
        demands = []
        for c in self.cities:
            if c.balance > 0:
                excesses.append([__class__.Balance(c.balance, c.time)])
                demands.append(0)
            else:
                excesses.append([])
                demands.append(-c.balance)
        total_demand = sum(demands)
        packages = []
        self.dispatch(packages, ratios, excesses)
        while True:
            arrived = self.move(packages)
            if arrived is None:
                break
            dest = arrived.destination
            if arrived.amount <= demands[dest]:
                demands[dest] -= arrived.amount
                total_demand -= arrived.amount
            else:
                heappush(excesses[dest],
                         __class__.Balance(arrived.amount - demands[dest],
                                           arrived.expiration))
                total_demand -= demands[dest]
                demands[arrived.destination] = 0
            if total_demand <= 0:
                break
            self.dispatch(packages, ratios, excesses)
        pass

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

    def dispatch(self, packages, ratios, balances) -> List[Package]:
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
                    if amount < left:
                        left -= amount
                    else:
                        amount = left
                        left = 0
                    package = __class__.Package(amount, i2,
                                                self.distances[i1][i2],
                                                balance.expiration)
                    heappush(packages, package)
                if left > 0:
                    balance.amount = left
                    heappush(balances[i1], balance)
        return packages

    def move(self, packages) -> Optional[Package]:
        if not packages:
            return None
        d = packages[0].distance
        for p in packages:
            p.distance -= d
            p.expiration -= d / self.speed
        head = heappop(packages)
        if head.expiration < 0:
            head = None
        else:
            return head
        while packages:
            d = packages[0].distance
            for p in packages:
                p.distance -= d
                p.expiration -= d / self.speed
            head = heappop(packages)
            if head.expiration < 0:
                head = None
        return head


def dist(c1: City, c2: City) -> float:
    return sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)
