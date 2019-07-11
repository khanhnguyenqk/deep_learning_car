from deepcar import Car, RaceTrack
import sys

import pygame
from pygame.locals import *
from pygame.color import *
    
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

    raceTrack.addCar(Car((100, 100), 0, 0, size=(10, 15), speed=1))
    raceTrack.addCar(Car((100, 50), 0, 1, size=(10, 15), speed=2))
    raceTrack.addCar(Car((100, 150), 0, 2, size=(10, 15), speed=3))
    raceTrack.startRace(space)

    segmentBody = pymunk.Body()
    segment = pymunk.Segment(segmentBody, (100, 100), (200, 200), 0.0)
    space.add(segment)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT or \
                event.type == KEYDOWN and (event.key in [K_ESCAPE, K_q]):  
                running = False

        for c in raceTrack.cars:
            c.move()
        
        segmentBody.position += Vec2d(5, 0)

        ### Clear screen
        screen.fill(pygame.color.THECOLORS["black"])
        
        ### Draw stuff
        space.debug_draw(draw_options)

        pygame.display.flip()
        
        ### Update physics
        fps = 60
        dt = 1./fps
        space.step(dt)

if __name__ == '__main__':
    sys.exit(main())