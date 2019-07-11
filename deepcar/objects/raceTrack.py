from pymunk import Segment, Space
from typing import Tuple, List
import pymunk
from .car import Car
import pygame

class RaceTrack:
    def __init__(self, size=(1000, 500)):
        self.walls = [] # list of Segments
        self.size = size # race track size
        self.startingPoint = (0, 0)
        self.cars = []

    def addWall(self, wall:Segment)->None:
        self.walls.append(wall)

    def addWallByFloats(self, wallEnds:List[Tuple[float, float]])->None:
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        segment = Segment(body, wallEnds[0], wallEnds[1], 0.0)
        self.walls.append(segment)

    def addCar(self, car:Car):
        self.cars.append(car)

    def startRace(self, space:Space):
        for w in self.walls:
            w.friction = 1
            w.group = 1
        space.add(self.walls)

        for c in self.cars:
            space.add(c.shape)

        handler = space.add_collision_handler(0, 1)
        handler.post_solve=Car.post_solve_car_hit