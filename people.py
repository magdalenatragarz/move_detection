import math
import cv2 as cv
import numpy as np


class Person(object):
    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.history = []
        self.history.append((x,y))
        self.updated = True
        self.tracker_initialized = False
        self.kalman_filter = cv.KalmanFilter(4,2)
        self.kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.tracker = cv.TrackerMOSSE_create()

    def update(self, x, y):
        self.x = x
        self.y = y
        self.history.append((x, y))

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


    def predict_move(self):
        self.predicted = []
        self.measured = np.array([[np.float32(self.x)], [np.float32(self.y)]])
        self.kalman_filter.correct(self.measured)
        self.estimate = self.kalman_filter.predict()
        self.predicted.append(int(self.estimate[0]))
        self.predicted.append(int(self.estimate[1]))
        return self.predicted






