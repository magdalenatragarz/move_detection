import math
import cv2 as cv
import numpy as np
import pygame
import colors
import scipy.stats
import circular_buffer
import common


class Person(object):
    def __init__(self,x,y,frame_count):
        self.x = x
        self.y = y
        self.frame_count = frame_count
        self.history = []
        self.history_points = []
        self.history.append((x,y,frame_count))
        self.history_points.append((x,y))
        self.history_x = circular_buffer.RingBuffer(15)
        self.history_x.append(x)
        self.history_y = circular_buffer.RingBuffer(15)
        self.history_y.append(y)
        self.updated = True
        self.kalman_filter = cv.KalmanFilter(4,2)
        self.kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.how_many_predicted = 0

    def update(self, x, y, frame_count):
        self.x = x
        self.y = y
        self.frame_count = frame_count
        self.history.append((x, y,frame_count))
        self.history_points.append((x,y))
        self.history_x.append(x)
        self.history_y.append(y)

    def dist(self, x, y):
        return math.hypot(x - self.x, y - self.y)

    def mark_not_updated(self):
        self.updated = False

    def mark_updated(self):
        self.updated = True

    def get_current_bounding_box(self):
        return self.x,self.y,common.BBOX_WIDTH,common.BBOX_HEIGHT

    def predict_move(self):
        self.predicted = []
        self.measured = np.array([[np.float32(self.x)], [np.float32(self.y)]])
        self.kalman_filter.correct(self.measured)
        self.estimate = self.kalman_filter.predict()
        self.predicted.append(int(self.estimate[0]))
        self.predicted.append(int(self.estimate[1]))
        return self.predicted

    def get_standard_deviation(self):
        _, _, _, _, std_err = scipy.stats.linregress(self.history_x.get(), self.history_y.get())
        return std_err

    def get_standard_deviation_with_new_point(self,x,y):
        self.new_history_x = self.history_x.get()
        self.new_history_x.append(x)
        self.new_history_y = self.history_y.get()
        self.new_history_y.append(y)
        _, _, _, _, std_err = scipy.stats.linregress(self.new_history_x, self.new_history_y)
        return std_err

#----------------------------------------------------------------

def draw_people(PEOPLE_LIST):
    running = True
    background_colour = (255, 255, 255)
    (width, height) = (700, 500)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Tutorial 1')
    screen.fill(background_colour)
    i = 0
    for person in PEOPLE_LIST:
        if (len(person.history_points) > 40):
            pygame.draw.lines(screen, colors.colors[i % len(colors.colors)], False, person.history_points, 2)
            print(person.history)
            pygame.display.update()
            pygame.display.flip()
            i += 1
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False



