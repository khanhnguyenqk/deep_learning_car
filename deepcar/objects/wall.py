import pymunk
from pymunk import Body, Segment

class Wall(Body):
    def __init__(self, a, b):
        super(Wall, self).__init__(body_type=pymunk.Body.STATIC)
        self.shape = Segment(self, a, b, 0.0)