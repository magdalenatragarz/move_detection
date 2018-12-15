import math
import pygame
import colors
import cv2 as cv
import numpy as np
import pykalman

class Person(object):
    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.history = []
        self.interpoled_history = []
        self.history.append((x,y))
        self.changes = 0
        self.is_alive = True
        self.updated = True
        self.kalman_filter = cv.KalmanFilter(4,2)
        self.kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.how_long_not_updated = 0

    def update(self, x, y):
        self.x = x
        self.y = y
        self.history.append((x, y))
        self.how_long_not_updated = 0

    def dist(self, x, y):
        return math.hypot(x - self.x, y - self.y)


    def die(self):
        self.is_alive = False


    def mark_not_updated(self):
        self.updated = False


    def mark_updated(self):
        self.updated = True

    def is_person_alive(self):
        return self.is_alive

    def predict_move(self):
        self.predicted = []
        self.measured = np.array([[np.float32(self.x)], [np.float32(self.y)]])
        self.kalman_filter.correct(self.measured)
        self.estimate = self.kalman_filter.predict()
        self.predicted.append(int(self.estimate[0]))
        self.predicted.append(int(self.estimate[1]))
        return self.predicted

    def interpole(self):
        self.x = np.asarray(self.history);
        self.initial_state_mean = [self.x[0, 0],
                                0,
                                   self.x[0, 1],
                                0]

        self.transition_matrix = [[1, 1, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 1],
                             [0, 0, 0, 1]]

        self.observation_matrix = [[1, 0, 0, 0],
                              [0, 0, 1, 0]]

        self.kf1 = pykalman.KalmanFilter(transition_matrices=self.transition_matrix,
                           observation_matrices=self.observation_matrix,
                           initial_state_mean=self.initial_state_mean)

        self.kf1 = self.kf1.em(self.history, n_iter=5)
        (self.smoothed_state_means, self.smoothed_state_covariances) = self.kf1.smooth(self.history)

        self.interpoled_history = self.smoothed_state_means;

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
        if (len(person.interpoled_history) > 4):
            pygame.draw.lines(screen, colors.colors[i % len(colors.colors)], False, person.history, 2)
            print(person.history)
            pygame.display.update()
            pygame.display.flip()
            i += 1
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

