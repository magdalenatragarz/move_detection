import math
import pygame
import colors
import cv2 as cv
import numpy as np

class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

class Person(object):
    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.history = []
        self.history.append((x,y))
        self.changes = 0
        self.is_alive = True
        self.updated = True

    def update(self,x,y):
        self.x = x
        self.y = y
        self.history.append((x,y))

    def is_ok_with_direction(self,x,y):
        if len(self.history) < 3 :
            return True


        self.left = False
        self.right = False
        self.up = False
        self.down = False
        self.y_min = 0
        self.y_max = 0
        self.x_min = 0
        self.x_max = 0

        (self.x_prev,self.y_prev) = self.history[len(self.history)-2]
        self.horizontal = self.x_prev - self.x
        self.vertical = self.y_prev - self.y
        if self.horizontal >= 0 :
            self.left = True
        elif self.horizontal < 0:
            self.right = True

        if self.vertical >= 0:
            self.up = True
        elif self.vertical < 0:
            self.down = True


        if(self.left):
            self.x_min = self.x_prev
            self.x_max = self.x - abs(self.x-self.x_prev)

        if (self.right):
            self.x_min = self.x_prev
            self.x_max = self.x + abs(self.x - self.x_prev)

        if (self.up) :
            self.y_min = self.y_prev
            self.y_max = self.y - abs(self.y - self.y_prev)

        if (self.down) :
            self.y_min = self.y_prev
            self.y_max = self.y + abs(self.y - self.y_prev)

        return (self.x_min,self.x_max,self.y_min,self.y_max)

    def dist(self,x,y):
        return math.hypot(x - self.x, y - self.y)


    def die(self):
        self.is_alive = False


    def mark_not_updated(self):
        self.updated = False


    def mark_updated(self):
        self.updated = True

    def is_person_alive(self):
        return self.is_alive


# -----------------------------------------------------------------------

# Instantiate OCV kalman filter
class KalmanFilter:

    kf = cv.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted


# -----------------------------------------------------------------------


def draw_people(PEOPLE_LIST):
    running = True
    background_colour = (255, 255, 255)
    (width, height) = (700, 500)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Tutorial 1')
    screen.fill(background_colour)
    i = 0
    for person in PEOPLE_LIST:
        if (len(person.history) > 4):
            pygame.draw.lines(screen, colors.colors[i % len(colors.colors)], False, person.history, 2)
            print(person.history)
            pygame.display.update()
            pygame.display.flip()
            i += 1
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

