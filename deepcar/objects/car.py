from pymunk import Body, Vec2d
import pymunk

class Car(Body):
    def __init__(self, position, angle, id, speed=2, size=(20,30)):
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

    def move(self):
        if not self.can_run:
            return
        impulse = self.speed * Vec2d(1, 0)
        impulse.rotate(self.angle)
        self.position += impulse

    def steered(self, delta_angle):
        if not self.can_run:
            return
        self.angle += delta_angle

    @staticmethod
    def post_solve_car_hit(arbiter, space, data):
        _, b = arbiter.shapes
        car = b.body
        car.can_run = False