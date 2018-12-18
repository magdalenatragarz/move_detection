import math
import cv2 as cv
import numpy as np
import pygame
import colors
import scipy.stats
import circular_buffer
import common


class Person(object):
    def __init__(self, x, y, frame_count):
        self.x = x
        self.y = y
        self.frame_count = frame_count


        self.history = []
        self.history_points = []
        self.history_x = circular_buffer.RingBuffer(15)
        self.history_y = circular_buffer.RingBuffer(15)

        self.history.append([x, y, frame_count])
        self.history_x.append(x)
        self.history_y.append(y)

        self.how_many_predicted = 0
        self.updated = True

        self.kalman_filter = cv.KalmanFilter(4,2)
        self.kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

        self.estimate = None
        self.measured = None
        self.history_x_copy = None
        self.history_y_copy = None

    def update(self, x, y, frame_count):
        self.x = x
        self.y = y
        self.frame_count = frame_count

        self.history.append([x, y,frame_count])
        self.history_x.append(x)
        self.history_y.append(y)

        self.mark_updated()

    def dist(self, x, y):
        return math.hypot(x - self.x, y - self.y)

    def mark_not_updated(self):
        self.updated = False

    def mark_updated(self):
        self.updated = True

    def is_person_predictable(self,frame_count):
        self.long_history_length = len(self.history) > common.CREDIBLE_HISTORY_LENGTH
        self.low_deviation = self.get_standard_deviation() < common.MAX_DEVIATION
        self.shortly_predicted = self.how_many_predicted < common.MAX_PREDICTIONS_QUANTITY
        self.not_too_old = frame_count - self.frame_count < common.MAX_FRAME_DIFFERENCE
        return not self.updated  and self.long_history_length and self.low_deviation and self.shortly_predicted and self.not_too_old

    def is_person_ready_to_draw(self):
        self.credible_history = len(self.history) > common.CREDIBLE_HISTORY_LENGTH
        self.not_a_deviant = self.get_standard_deviation() < common.MAX_DEVIATION
        return self.credible_history and self.not_a_deviant and self.updated

    def get_current_bounding_box(self):
        return self.x, self.y, common.BBOX_WIDTH, common.BBOX_HEIGHT

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
        self.history_x_copy = self.history_x.get()
        self.history_x_copy.append(x)
        self.history_y_copy = self.history_y.get()
        self.history_y_copy.append(y)
        _, _, _, _, std_err = scipy.stats.linregress(self.history_x_copy, self.history_y_copy)
        return std_err



