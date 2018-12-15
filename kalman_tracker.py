"""
As implemented in https://github.com/abewley/sort but with some modifications
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


'''Motion Model'''
class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox,img=None):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=2, dim_z=1)
    self.kf.F = np.array([[1., 1.],[0., 1.]])
    self.kf.H = np.array([[1., 0.]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P = np.array([[1000., 0.],[0., 1000.]])
    self.kf.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
    self.kf.R = 5

    self.kf.x = bbox

    self.history = []

  def update(self,bbox,img=None):
    """
    Updates the state vector with observed bbox.
    """
    if bbox != []:
      self.kf.update(np.array([[bbox[0],bbox[1]]], np.float32))

  def predict(self,img=None):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    self.kf.predict()
    self.history.append(convert_x_to_bbox(self.kf.x))
    return convert_x_to_bbox(self.kf.x)[0]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)[0]


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2]-bbox[0]
  h = bbox[3]-bbox[1]
  x = bbox[0]+w/2.
  y = bbox[1]+h/2.
  s = w*h    #scale is just area
  r = w/float(h)
  return np.array([x,y,s,r]).reshape((4,1))

def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2]*x[3])
  h = x[2]/w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))