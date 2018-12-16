import math
import cv2 as cv
import numpy as np
import pygame
import colors


class Person(object):
    def __init__(self,x,y,frame_count):
        self.x = x
        self.y = y
        self.history = []
        self.history.append((x,y,frame_count))
        self.updated = True
        self.tracker_initialized = False
        self.kalman_filter = cv.KalmanFilter(4,2)
        self.kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.tracker = cv.TrackerMOSSE_create()

    def update(self, x, y, frame_count):
        self.x = x
        self.y = y
        self.history.append((x, y,frame_count))

    def init_tracker(self, frame):
        self.tracker.init(frame,(self.x,self.y,self.x+15,self.y+25))
        self.tracker_initialized = True

    def update_tracker(self,frame):
        return self.tracker.upate(frame)

    def dist(self, x, y):
        return math.hypot(x - self.x, y - self.y)

    def mark_not_updated(self):
        self.updated = False


    def mark_updated(self):
        self.updated = True


    def get_current_bounding_box(self):
        return (self.x,self.y,15,25)

    def predict_move(self):
        self.predicted = []
        self.measured = np.array([[np.float32(self.x)], [np.float32(self.y)]])
        self.kalman_filter.correct(self.measured)
        self.estimate = self.kalman_filter.predict()
        self.predicted.append(int(self.estimate[0]))
        self.predicted.append(int(self.estimate[1]))
        return self.predicted




def draw_people(PEOPLE_LIST):
    running = True
    background_colour = (255, 255, 255)
    (width, height) = (700, 500)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Tutorial 1')
    screen.fill(background_colour)
    i = 0
    for person in PEOPLE_LIST:
        if (len(person.history) > 40):
            pygame.draw.lines(screen, colors.colors[i % len(colors.colors)], False, person.history, 2)
            print(person.history)
            pygame.display.update()
            pygame.display.flip()
            i += 1
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False



