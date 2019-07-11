class City:
    def __init__(self, name: str, x: float, y: float, balance: int, time: int):
        self.name = name
        self.x = x
        self.y = y
        self.balance = balance
        self.time = time


class Problem:
    def __init__(self, cities):
        self.cities = cities

