import pygame
from pygame.locals import *
from pygame.color import *
import pymunk
from pymunk import Vec2d
import argparse
import json
import os.path

X, Y = 0, 1

def getArgParse()->argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='UI to create race track.')
    parser.add_argument('w', type=int, help='width of the race track')
    parser.add_argument('h', type=int, help='width of the race track')
    return parser

def main():
    parser = getArgParse()
    args = parser.parse_args()
    w = args.w
    h = args.h
    pygame.init()
    screen = pygame.display.set_mode((w, h))

    space = pymunk.Space()

    flipY = lambda y:h - y

    running = True

    line_point1 = None

    raceDict = {'w':w, 'h':h, 'walls':[], 'starting_point':None, 'finish_lines':[]}

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                running = False
            elif event.type == KEYDOWN and event.key == K_e:
                main_dir = os.path.split(os.path.abspath(__file__))[0]
                fp = os.path.join(main_dir, 'race_track.json')
                with open(fp, 'w') as f:
                    json.dump(raceDict, f)
            elif pygame.key.get_mods() & KMOD_CTRL and pygame.mouse.get_pressed()[0]:
                starting_point = (event.pos[X], flipY(event.pos[Y]))
                raceDict['starting_point'] = starting_point
            elif event.type == MOUSEBUTTONDOWN and (event.button == 1 or event.button == 3):
                if line_point1 is None:
                    line_point1 = (event.pos[X], flipY(event.pos[Y]))
            elif event.type == MOUSEBUTTONUP and (event.button == 1 or event.button == 3):
                if line_point1 is not None:
                    line_point2 = (event.pos[X], flipY(event.pos[Y]))
                    if event.button == 1:
                        raceDict['walls'].append((line_point1, line_point2))
                    else:
                        raceDict['finish_lines'].append((line_point1, line_point2))
                    line_point1 = None

        ### Draw stuff
        screen.fill(THECOLORS["white"])

        # Display some text
        font = pygame.font.Font(None, 16)
        text = """LMB: Drag to create wall
RMB: Drag to create mini finish line
CTRL + LMB: Create starting point
E: Export to json"""
        y = 5
        for line in text.splitlines():
            text = font.render(line, 1,THECOLORS["black"])
            screen.blit(text, (5,y))
            y += 10

        p = pygame.mouse.get_pos()
        mouse_pos = Vec2d(p[X],flipY(p[Y]))

        if line_point1 is not None:
            p1 = line_point1[X], flipY(line_point1[Y])
            p2 = mouse_pos.x, flipY(mouse_pos.y)
            pygame.draw.lines(screen, THECOLORS["lightgray"], False, [p1,p2])
        
        for line in raceDict['walls']:
            p1 = line[0]
            p2 = line[1]
            p1 = p1[0], flipY(p1[1])
            p2 = p2[0], flipY(p2[1])
            pygame.draw.lines(screen, THECOLORS["red"], False, [p1,p2])
        
        for line in raceDict['finish_lines']:
            p1 = line[0]
            p2 = line[1]
            p1 = p1[0], flipY(p1[1])
            p2 = p2[0], flipY(p2[1])
            pygame.draw.lines(screen, THECOLORS["green"], False, [p1,p2])

        if raceDict['starting_point']:
            x, y = raceDict['starting_point']
            y = flipY(y)
            pygame.draw.circle(screen, THECOLORS["blue"], (x, y), 5, 2)

        ### Flip screen
        pygame.display.flip()

if __name__ == '__main__':
    main()