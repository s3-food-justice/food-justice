from dataclasses import dataclass, field
from heapq import heappush, heappop
from math import sqrt
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


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
        for i1, c1 in enumerate(fd.cities):
            for i2, c2 in enumerate(fd.cities):
                if c1 is c2:
                    continue
                p1 = np.array([c1.x, c1.y])
                p2 = np.array([c2.x, c2.y])
                norm = p2 - p1
                norm[0], norm[1] = -norm[1], norm[0]
                norm /= np.linalg.norm(norm)
                p1 += norm * 1
                p2 += norm * 1
                npts = 100
                x = np.linspace(p1[0], p2[0], npts)
                y = np.linspace(p1[1], p2[1], npts)
                g = np.linspace(0, 1, npts)
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                nc = plt.Normalize(g.min(), g.max())
                lc = LineCollection(segments, cmap='copper', norm=nc)
                lc.set_array(g)
                lc.set_linewidth(self.get_amount(c1, c2) * 10)
                lc.set_zorder(0)
                ax.add_collection(lc)
                #ax.plot(x, y, '-k',
                #        lw=self.get_amount(c1, c2) * 10,
                #        zorder=0)


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
        self.speed = .3

    def get_city(self, name: str) -> Optional[City]:
        for c in self.cities:
            if c.name == name:
                return c
        return None

    def plot(self, ax: plt.Axes, demand_override=None, excess_override=None):
        balances = np.array([c.balance for c in self.cities])
        if demand_override is not None:
            balances[demand_override < 0] = demand_override[demand_override < 0]
        if excess_override is not None:
            balances[excess_override > 0] = excess_override[excess_override > 0]
        max = np.abs(balances).max()
        sizes = np.zeros((len(self.cities),))
        colors = np.zeros((len(self.cities), 3))
        sizes[balances != 0] = 80 * (balances[balances != 0] / max) ** 2
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
        print('{} sim'.format(id(solution)))
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
        i = 0
        while True:
            #print('{} {}'.format(id(solution), i))
            i += 1
            arrived, cost = self.move(packages, excesses)
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
        return total_demand, total_cost, demands, [sum([e.amount for e in es])
                                                   for es in excesses]

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

    def dispatch(self, packages, ratios, excesses):
        for i1, c1 in enumerate(self.cities):
            if not excesses[i1]:
                continue
            while excesses[i1]:
                excess = heappop(excesses[i1])
                amt = excess.amount
                left = excess.amount
                for i2, c2 in enumerate(self.cities):
                    r = ratios[i1][i2]
                    if r == 0:
                        continue
                    amount = int(round(excess.amount * r))
                    if amount <= 0:
                        continue
                    if amount < left:
                        left -= amount
                    else:
                        amount = left
                        left = 0
                    package = Package(amount, i1, i2,
                                      self.distances[i1][i2],
                                      excess.expiration)
                    heappush(packages, package)
                if left > 0:
                    excess.amount = left
                    heappush(excesses[i1], excess)
                    if left == amt:
                        break

    def move(self, packages: List[Package],
             balances: List[List[Balance]]) -> Tuple[Optional[Package], float]:
        if not packages:
            return None, 0
        cost = 0
        d = packages[0].distance
        t = d / self.speed
        for p in packages:
            p.distance -= d
            p.expiration -= t
            cost += d * p.amount
        for bs in balances:
            for b in bs:
                b.expiration -= t
            while bs and bs[0].expiration <= 0:
                heappop(bs)
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
