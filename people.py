import math

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
        self.history.append([x,y])
        self.changes = 0
        self.is_alive = True

    def update(self,x,y):
        self.x = x
        self.y = y
        #self.w = w
        #self.h = h
        self.history.append([x,y])

    def define_direction(self):
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

