import cv2
import math
import imutils
from array import array

people = []


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

    def update(self,x,y):
        self.x = x
        self.y = y
        #self.w = w
        #self.h = h
        self.history.append((x,y))


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
# --------------------------------------



def main():

    video_path = 'D:\krk3.mov'
    cv2.ocl.setUseOpenCL(False)

    #read video file
    cap = cv2.VideoCapture(video_path)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    counter = 0
    while ret:

        for x in people:
            x.mark_not_updated()

        _, frame = cap.read()
        frame1 = imutils.resize(frame1, width=700)
        frame2 = imutils.resize(frame2, width=700)
        d = cv2.absdiff(frame1, frame2)

        imgray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imgray, (3, 3), 0)
        ret, thresh = cv2.threshold(blur, 15, 255, cv2.THRESH_BINARY)
        #thresh = cv2.dilate(thresh, None, iterations=2)
        _, contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #contours = contours[0] if imutils.is_cv2() else contours[1]

        for c in contours:
            if cv2.contourArea(c) < 100 or cv2.contourArea(c) > 200:
                continue

            # get bounding box from countour
            (x, y, w, h) = cv2.boundingRect(c)

            if y < 250:
                continue

            if not people:
                person_x = Person(x, y, w, h)
                person_x.mark_updated()
                people.append(person_x)
                continue

            updated = False
            for person in people:
                if person.dist(x,y) < 40 and not(person.updated) and person.is_person_alive():
                    if person.dist(x,y) < 10:
                        person.die()
                        continue
                    #print("person from ",person.x," ",person.y, "updated to ",x," ",y)
                    person.update(x,y)
                    person.mark_updated()
                    updated = True
                    break
            if updated==False:
                person_y = Person(x, y, w, h)
                person_y.mark_updated()
                people.append(person_y)
            # draw bounding box

        for person in people:
            if (person.is_alive):
                #cv2.rectangle(frame1, (person.x, person.y), (person.x + person.w, person.y + person.h), (0, 255, 0), 2)
                cv2.rectangle(frame1, (person.x, person.y), (person.x+15, person.y+25), (0, 255, 0), 2)
            #else:
                #print ("dead person!")

        #cv2.imshow("inter", frame1)
        cv2.imshow("Thresh", thresh)
        #cv2.imshow("Frame Delta", d)
        #cv2.imshow("Frame Delta", imgray)

        if cv2.waitKey(40) == 27:
            break

        counter = counter+1
        print("counter: ",counter)
        frame1 = frame2
        ret, frame2 = cap.read()
        # j=j+1`

    cv2.destroyAllWindows()
    cap.release()


main()

# x = Person(1,2,3,4)
# people.append(x)
# y = Person(4,6,7,8)
# people.append(y)
# y.update(2,5)
#
# print(people[0].history)
# print(people[1].history)

