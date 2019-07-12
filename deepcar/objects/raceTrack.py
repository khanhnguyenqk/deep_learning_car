from pymunk import Space
from typing import Tuple, List
import pymunk
from .car import Car
from .wall import Wall
import pygame

class RaceTrack:
    def __init__(self, size=(1000, 500)):
        self.walls = [] # list of Walls
        self.size = size # race track size
        self.startingPoint = (0, 0)
        self.cars = []

    def addWall(self, wall:Wall)->None:
        self.walls.append(wall)

    def addWallByFloats(self, wallEnds:List[Tuple[float, float]])->None:
        wall = Wall(wallEnds[0], wallEnds[1])
        self.walls.append(wall)

    def addCar(self, car:Car):
        self.cars.append(car)

    def startRace(self, space:Space):
        for w in self.walls:
            w.shape.friction = 1
            w.shape.group = 1
            space.add(w.shape)

        for c in self.cars:
            c.addSelfToSpace(space)

        handler = space.add_collision_handler(0, 1)
        handler.post_solve=Car.post_solve_car_hit