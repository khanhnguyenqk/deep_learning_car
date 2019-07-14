from typing import Tuple, List
import pymunk
import pygame
from pymunk import Body, Vec2d, Space, Segment
import math

class Wall(Body):
    def __init__(self, a, b):
        super(Wall, self).__init__(body_type=pymunk.Body.STATIC)
        self.shape = Segment(self, a, b, 0.0)
        self.shape.collision_type = 0



class Car(Body):
    def __init__(self, position, angle, id, raceTrack, speed=0.5, size=(20,30)):
        self.id = id
        w, l = size
        vs = [(0, -w/2), (w, -w/2), (l, 0), (w, w/2), (0, w/2)]
        mass = 0
        moment = pymunk.moment_for_poly(mass, vs)
        super(Car, self).__init__(mass, moment)

        self.shape = pymunk.Poly(self, vs)
        self.shape.collision_type = 1
        
        self.speed = speed
        self.can_run = True
        
        self.angle = angle
        self.position = position
        self.sensors = list(self.createSensors(raceTrack))

    def createSensors(self, raceTrack):
        sensorAngles = [60, 30, 0, -30, -60]
        for a in sensorAngles:
            yield RadarSensor(self.position, math.radians(a), raceTrack)

    def move(self):
        if not self.can_run:
            return
        impulse = self.speed * Vec2d(1, 0)
        impulse.rotate(self.angle)
        self.position += impulse
        for sensor in self.sensors:
            sensor.root = self.position

    def steered(self, delta_angle):
        if not self.can_run:
            return
        self.angle += delta_angle
        for sensor in self.sensors:
            sensor.rotateDeltaAboutRoot(delta_angle)

    def addSelfToSpace(self, space):
        space.add(self.shape)

    def assign_nn(self, nn):
        pass


    @staticmethod
    def post_solve_car_hit(arbiter, space, data):
        a, b = arbiter.shapes
        wall = a.body
        car = b.body
        if not isinstance(wall, Wall) or not isinstance(car, Car):
            return
        
        b.collision_type = 0
        car.can_run = False

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

    def addCar(self, car):
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


class RadarSensor:
    def __init__(self, rootPosition, angle, raceTrack, range=300):
        self.range = range
        self._angle = angle
        self._root = Vec2d(rootPosition)
        self.raceTrack = raceTrack
        self.point = None

    def _getRoot(self):
        return self._root
    def _setRoot(self, newRoot):
        self._root = newRoot
        self.updateObstaclePoint()
    root = property(_getRoot, _setRoot)

    def _getAngle(self):
        return self._angle
    def _setAngle(self, newAngle):
        self._angle = newAngle
        self.updateObstaclePoint()
    angle = property(_getAngle, _setAngle)

    def _getTail(self):
        length = Vec2d(self.range, 0)
        length.rotate(self.angle)
        return self.root + length


    def rotateDeltaAboutRoot(self, delta_radian):
        self.angle += delta_radian

    def get_distance(self)->float:
        '''
        Get distance from root to point (where radar hits the wall).
        If radar does not detect wall (out of range), then return range.
        '''
        if not self.point:
            return float(self.range)
        return (self.root - self.point).length

    def updateObstaclePoint(self):
        self.point = None
        walls = [x.shape for x in self.raceTrack.walls]
        obstacles = []
        tail = self._getTail()
        for shape in walls:
            segmentQueryInfo = shape.segment_query(self.root, tail)
            if segmentQueryInfo.shape:
                obstacles.append(segmentQueryInfo.point)
        if not obstacles:
            return
        obstacles.sort(key=lambda x: (x - self.root).length)
        self.point = obstacles[0]

def _test_get_distance():
    raceTrack = RaceTrack()
    raceTrack.addWallByFloats([(100, 10), (100, -10)])
    radar = RadarSensor((0, 0), 0, raceTrack)
    radar.updateObstaclePoint()
    print(radar.get_distance)

if __name__ == '__main__':
    import sys
    from pathlib import Path # if you haven't already done so
    file = Path(__file__).resolve()
    parent, root = file.parent, file.parents[1]
    sys.path.append(str(root))
    
    _test_get_distance()

    # Additionally remove the current file's directory from sys.path
    try:
        sys.path.remove(str(parent))
    except ValueError: # Already removed
        pass