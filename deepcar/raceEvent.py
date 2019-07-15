from objects import Car, RaceTrack, RadarSensor
from neural_network import NN
import os, sys
import pygame
from pygame.locals import *
from pygame.color import THECOLORS
import math
import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util

def main():
    CARS_PER_LAP = 10
    DRAW_SENSORS = False
    LAP_CNT = 1000
    CAR_SPEED = 0.6
    CAR_STEER_MAX = math.radians(4)

    # Put window at position (0, 0) of the monitor
    os.environ['SDL_VIDEO_WINDOW_POS'] = "10,10"

    pygame.init()
    
    # Load race track from json file
    dirname, _ = os.path.split(os.path.abspath(__file__))
    fp = os.path.join(dirname, 'race_track.json')
    raceTrack = RaceTrack.createFromJson(fp)

    # Create screen
    screen = pygame.display.set_mode(raceTrack.size)
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    # Create physic space
    space = pymunk.Space()
    space.gravity = 0, 0

    raceTrack.addSelfToSpace(space)
    
    for lap in range(LAP_CNT):
        # Add cars and NNs
        if lap == 0:
            for i in range(CARS_PER_LAP):
                car = Car((100, 150), 0, i, raceTrack, size=(10, 15), speed=CAR_SPEED, steermax=CAR_STEER_MAX)
                nn = NN(len(car.sensors), 2, [10, 10])
                nn.randomize_weights_biases()
                car.assign_nn(nn)
                raceTrack.addCar(car)
        else:
            best_car = next(iter(raceTrack.best_cars))
            nn = best_car.nn

            raceTrack.clearCars()

            for i in range(CARS_PER_LAP-1):
                new_nn = nn.deep_copy()
                new_nn.mutate_randomly()
                car = Car((100, 150), 0, i, raceTrack, size=(10, 15), speed=CAR_SPEED, steermax=CAR_STEER_MAX)
                car.assign_nn(new_nn)
                raceTrack.addCar(car)
            car = Car((100, 150), 0, i+1, raceTrack, size=(10, 15), speed=CAR_SPEED, steermax=CAR_STEER_MAX)
            car.assign_nn(nn)
            raceTrack.addCar(car)
        raceTrack.init_race()

        quit = False
        while not raceTrack.isRaceFinished() and not quit:
            for event in pygame.event.get():
                if event.type == QUIT or \
                    event.type == KEYDOWN and (event.key in [K_ESCAPE, K_q]):  
                    quit = True
                
            ### Clear screen
            screen.fill(pygame.color.THECOLORS["white"])

            # Next step of race track
            raceTrack.step()

            # Draw cars' radar sensors
            if DRAW_SENSORS:
                sensors = [s for c in raceTrack.cars for s in c.sensors]
                radarPoints = [tuple(s.point) for s in sensors if s.point]
                for p in radarPoints:
                    p = int(round(p[0])), raceTrack.size[1]-int(round(p[1]))
                    pygame.draw.circle(screen, (0, 0, 255), p, 5)
            
            ### Draw stuff
            space.debug_draw(draw_options)
            
            # Display some text
            font = pygame.font.Font(None, 16)
            text = f'Lap: {lap}'
            y = 5
            for line in text.splitlines():
                text = font.render(line, 1,THECOLORS["black"])
                screen.blit(text, (5,y))
                y += 10
                
            pygame.display.flip()
            
            ### Update physics
            fps = 120
            dt = 1./fps
            space.step(dt)

        if quit:
            break

if __name__ == '__main__':
    sys.exit(main())