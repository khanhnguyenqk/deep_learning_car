from pymunk import Body, Vec2d
import pymunk
from .wall import Wall
from .radarSensor import RadarSensor
import math

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


    @staticmethod
    def post_solve_car_hit(arbiter, space, data):
        a, b = arbiter.shapes
        wall = a.body
        car = b.body
        if not isinstance(wall, Wall) or not isinstance(car, Car):
            return
        
        b.collision_type = 0
        car.can_run = False