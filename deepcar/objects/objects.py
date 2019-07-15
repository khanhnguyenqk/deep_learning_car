from typing import Tuple, List
import pymunk, pygame
from pymunk import Body, Vec2d, Space, Segment
import math
import json

COLLTYPE_WALL = 0
COLLTYPE_FINISHLINE = 2
COLLTYPE_CAR = 1

class Wall(Body):
    def __init__(self, a, b):
        super(Wall, self).__init__(body_type=pymunk.Body.STATIC)
        self.shape = Segment(self, a, b, 0.0)
        self.shape.collision_type = COLLTYPE_WALL

class FinishLine(Body):
    def __init__(self, a, b, id):
        super(FinishLine, self).__init__(body_type=pymunk.Body.STATIC)
        self.shape = Segment(self, a, b, 0.0)
        self.shape.collision_type = COLLTYPE_FINISHLINE
        self.id = id

degree = math.radians(2)
class Car(Body):
    def __init__(self, position, angle, id, raceTrack, speed=0.1, size=(20,30), steermax=degree):
        self.id = id
        w, l = size
        vs = [(0, -w/2), (w, -w/2), (l, 0), (w, w/2), (0, w/2)]
        mass = 0
        moment = pymunk.moment_for_poly(mass, vs)
        super(Car, self).__init__(mass, moment)

        self.shape = pymunk.Poly(self, vs)
        self.shape.collision_type = COLLTYPE_CAR
        
        self.speed = speed
        self.can_run = True
        
        self.angle = angle
        self.position = position
        self.sensors = list(self.createSensors(raceTrack))
        self.nn = None
        self.steermax = steermax

    def __key__(self):
        return self.id
    
    def __hash__(self):
        return hash(self.__key__())

    def __eq__(self, other):
        if isinstance(other, Car):
            return self.__key__() == other.__key__()
        return NotImplemented

    def createSensors(self, raceTrack):
        sensorAngles = [60, 30, 0, -30, -60]
        for a in sensorAngles:
            yield RadarSensor(self.position, math.radians(a), raceTrack)

    def move(self):
        if not self.can_run:
            return
        if self.nn:
            xs = self.get_scaled_radar_distances()
            ys = self.nn.forward(xs)
            steer_delta = (ys[0] - 0.5) / 0.5 * self.steermax
            self.steered(steer_delta)

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

    def removeSelfFromSpace(self, space):
        space.remove(self.shape)

    def assign_nn(self, nn):
        self.nn = nn

    def get_scaled_radar_distances(self)->List[float]:
        return [x.get_distance() / x.range for x in self.sensors]

class RaceTrack:
    def __init__(self, size=(1000, 500), starting_point=(0,0), starting_angle=0):
        self.walls = [] # list of Walls
        self.size = size # race track size
        self.starting_point = starting_point
        self.cars = []
        self.finish_lines = [] # mini finish lines to determine car's fitness
        self.starting_angle = starting_angle # starting angle of cars
        self.best_cars = {} # car(s) cross furthest finish_line
        self.best_line_crossed = -1
        self.running_car_ids = {car.id for car in self.cars}
        self.space = None

    def addWall(self, wallEnds:List[Tuple[float, float]])->None:
        wall = Wall(wallEnds[0], wallEnds[1])
        self.walls.append(wall)

    def addFinishLine(self, lineEnds:List[Tuple[float, float]], id)->None:
        line = FinishLine(lineEnds[0], lineEnds[1], id)
        self.finish_lines.append(line)

    def addCar(self, car):
        self.cars.append(car)
        car.position = self.starting_point
        if car.id in self.running_car_ids:
            raise Exception('Cannot add cars with same IDs')
        self.running_car_ids.add(car.id)

    def addSelfToSpace(self, space:Space):
        self.space = space
        for w in self.walls:
            w.shape.friction = 1
            w.shape.group = 1
            space.add(w.shape)

        for l in self.finish_lines:
            l.shape.friction = 1
            l.shape.group = 3
            space.add(l.shape)

        handler = space.add_collision_handler(COLLTYPE_WALL, COLLTYPE_CAR)
        handler.pre_solve=self.pre_solve_car_hit_wall

        handler = space.add_collision_handler(COLLTYPE_FINISHLINE, COLLTYPE_CAR)
        handler.pre_solve=self.pre_solve_car_pass_line

        handler = space.add_collision_handler(COLLTYPE_CAR, COLLTYPE_CAR)
        handler.pre_solve=RaceTrack.pre_solve_car_hit_car

    def isRaceFinished(self):
        return not bool(self.running_car_ids)

    def clearCars(self):
        '''
        Remove all cars, and remove them from space if they are in space
        '''
        for c in self.cars:
            try:
                self.space.remove(c.shape)
            except:
                pass
        self.cars = []

    def init_race(self):
        if not self.space:
            raise Exception('Physic space is not defined. Call addSelfToSpace first.')

        self.best_cars = {}
        self.best_line_crossed = -1

        for c in self.cars:
            c.addSelfToSpace(self.space)
    
    @staticmethod
    def pre_solve_car_hit_car(arbiter, space, data):
        return False

    def pre_solve_car_pass_line(self, arbiter, space, data):
        a, b = arbiter.shapes
        line = a.body
        car = b.body
        if not isinstance(line, FinishLine) or not isinstance(car, Car):
            return
        
        if line.id > self.best_line_crossed:
            self.best_line_crossed = line.id
            self.best_cars = {car}
        elif line.id == self.best_line_crossed:
            self.best_cars.add(car)
        return False

    def pre_solve_car_hit_wall(self, arbiter, space, data):
        a, b = arbiter.shapes
        wall = a.body
        car = b.body
        if not isinstance(wall, Wall) or not isinstance(car, Car):
            return True
        
        if car.id in self.running_car_ids:
            car.can_run = False
            self.running_car_ids.remove(car.id)
        return False

    def step(self)->None:
        '''
        Make the next step in the game.
            - Tell each car to move.
            - Calculate other metrics
        '''
        for car in self.cars:
            car.move()
        

    @staticmethod
    def createFromJson(fp:str):
        '''
        Create a RaceTrack from a json file

        INPUT
            fp: file path of json file
        OUTPUT
            a RaceTrack
        '''
        with open(fp, 'r') as file:
            dic = json.load(file)
        raceTrack = RaceTrack(size=(dic['w'], dic['h']), starting_point=dic['starting_point'])
        for w in dic['walls']:
            raceTrack.addWall(w)
        for i, l in enumerate(dic['finish_lines']):
            raceTrack.addFinishLine(l, i)
        return raceTrack


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

def _test_Radar_get_distance():
    space = pymunk.Space()
    
    raceTrack = RaceTrack()
    raceTrack.addWall([(100, 10), (100, -10)])
    raceTrack.addSelfToSpace(space)

    radar = RadarSensor((0, 0), 0, raceTrack)
    radar.updateObstaclePoint()

    print(radar.get_distance())

def _test_Car_get_scaled_radar_distances():
    space = pymunk.Space()
    
    raceTrack = RaceTrack()
    raceTrack.addWall([(100, 100), (100, -100)])
    raceTrack.addSelfToSpace(space)

    car = Car((0, 0), 0, 0, raceTrack)
    car.addSelfToSpace(space)
    car.move()

    print(car.get_scaled_radar_distances())

if __name__ == '__main__':    
    _test_Radar_get_distance()
    _test_Car_get_scaled_radar_distances()