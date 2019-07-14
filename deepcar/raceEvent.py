from objects import Car, RaceTrack, RadarSensor
from neural_network import NN
import sys

import pygame
from pygame.locals import *
from pygame.color import *
import math
    
import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util

def main():
    pygame.init()
    raceTrack = RaceTrack((1000, 500))
    screen = pygame.display.set_mode(raceTrack.size)

    space = pymunk.Space()
    space.gravity = 0, 0
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    raceTrack.addWallByFloats([(10, 10), (600, 10)])
    raceTrack.addWallByFloats([(600, 10), (600, 400)])
    raceTrack.addWallByFloats([(600, 400), (10, 400)])

    """ raceTrack.addCar(Car((100, 100), 0, 0, raceTrack, size=(10, 15), speed=0.5))
    raceTrack.addCar(Car((100, 50), 0, 1, raceTrack, size=(10, 15), speed=0.7)) """
    car = Car((100, 150), 0, 2, raceTrack, size=(10, 15), speed=0.3)
    nn = NN(len(car.sensors), 2, [10, 10])
    nn.randomize_weights_biases()
    car.assign_nn(nn)
    raceTrack.addCar(car)
    raceTrack.addSelfToSpace(space)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT or \
                event.type == KEYDOWN and (event.key in [K_ESCAPE, K_q]):  
                running = False
              
        ### Clear screen
        screen.fill(pygame.color.THECOLORS["black"])

        for c in raceTrack.cars:
            c.move()
            #c.steered(math.radians(0.05))

        sensors = [s for c in raceTrack.cars for s in c.sensors]
        radarPoints = [tuple(s.point) for s in sensors if s.point]
        for p in radarPoints:
            p = int(round(p[0])), raceTrack.size[1]-int(round(p[1]))
            pygame.draw.circle(screen, (0, 0, 255), p, 5)
        
        ### Draw stuff
        space.debug_draw(draw_options)

        pygame.display.flip()
        
        ### Update physics
        fps = 60
        dt = 1./fps
        space.step(dt)

if __name__ == '__main__':
    sys.exit(main())