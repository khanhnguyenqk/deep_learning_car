from objects import Car, RaceTrack, RadarSensor
from neural_network import NNNumpy, NNTorch, NNLoader
import os, sys
import pygame
from pygame.locals import *
from pygame.color import THECOLORS
import math
import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util

def main():
    CARS_PER_LAP = 50
    DRAW_SENSORS = True
    LAP_CNT = 1000
    CAR_SPEED = 0.6
    CAR_STEER_MAX = math.radians(30)
    CAR_SENSOR_ANGLES = [-60, -30, 30, 60]
    RADAR_RANGE = 400
    NN_MUTATION_PROB_MIN = 0.01
    NN_MUTATION_PROB_MAX = 0.05
    NN_HIDDEN_LAYERS = [8, 8]
    USE_PYTORCH = True
    LOAD_NN_FP = False

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

    # Create a neural network
    if LOAD_NN_FP:
        if USE_PYTORCH:
            init_nn = NNLoader.load('current_best_nn.pth')
        else:
            init_nn = NNLoader.load('current_best_nn.json')
    else:
        if USE_PYTORCH:
            init_nn = NNTorch(len(CAR_SENSOR_ANGLES), 2, NN_HIDDEN_LAYERS)
        else:
            init_nn = NNNumpy(len(CAR_SENSOR_ANGLES), 2, NN_HIDDEN_LAYERS)
    
    current_best_nn = init_nn

    for lap in range(LAP_CNT):
        # Add cars and NNs
        if lap == 0:
            for i in range(CARS_PER_LAP):
                car = Car((0, 0), 0, i, raceTrack, size=(10, 15), speed=CAR_SPEED, steermax=CAR_STEER_MAX, radar_range=RADAR_RANGE, sensor_angles=CAR_SENSOR_ANGLES)
                nn = init_nn.deep_copy()
                if LOAD_NN_FP:
                    nn.mutate_randomly(intensity=1, min_prob=NN_MUTATION_PROB_MIN, max_prob=NN_MUTATION_PROB_MAX)
                else:
                    nn.randomize_weights_biases()
                car.assign_nn(nn)
                raceTrack.addCar(car)
        else:
            best_car = next(iter(raceTrack.best_cars))
            fit_score = (raceTrack.best_line_crossed + 1) / len(raceTrack.finish_lines)
            current_best_nn = best_car.nn

            raceTrack.clearCars()

            for i in range(CARS_PER_LAP):
                new_nn = current_best_nn.deep_copy()
                new_nn.mutate_randomly(intensity=1-fit_score, min_prob=NN_MUTATION_PROB_MIN, max_prob=NN_MUTATION_PROB_MAX)
                car = Car((0, 0), 0, i, raceTrack, size=(10, 15), speed=CAR_SPEED, steermax=CAR_STEER_MAX, radar_range=RADAR_RANGE, sensor_angles=CAR_SENSOR_ANGLES)
                car.assign_nn(new_nn)
                raceTrack.addCar(car)
                
        raceTrack.init_race()

        quit = False
        manual_next_lap = False
        paused = False
        while ((not raceTrack.isRaceFinished() or lap == LAP_CNT-1) 
            and not quit and not manual_next_lap):
            for event in pygame.event.get():
                if event.type == QUIT or \
                    event.type == KEYDOWN and (event.key in [K_ESCAPE, K_q]):  
                    quit = True
                if event.type == KEYDOWN and event.key == K_n:
                    manual_next_lap = True
                if event.type == KEYDOWN and event.key == K_e:
                    NNLoader.save(current_best_nn, 'current_best_nn')
                if event.type == KEYDOWN and event.key == K_SPACE:
                    paused = not paused

            if paused:
                continue
                
            ### Clear screen
            screen.fill(pygame.color.THECOLORS["white"])

            # Next step of race track
            raceTrack.step()

            # Draw cars' radar sensors
            if DRAW_SENSORS:
                sensors = [s for c in raceTrack.cars for s in c.sensors if c.can_run]
                radarPoints = [tuple(s.point) for s in sensors if s.point]
                for p in radarPoints:
                    p = int(round(p[0])), raceTrack.size[1]-int(round(p[1]))
                    pygame.draw.circle(screen, (0, 0, 255), p, 5)
            
            ### Draw stuff
            space.debug_draw(draw_options)
            
            # Display some text
            font = pygame.font.Font(None, 16)
            text = f'''
ESC: quit
n: terminate current lap
e: serialize current best car to disk
space: pause

Pytorch: {USE_PYTORCH}
Lap (generation): {lap}'''
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