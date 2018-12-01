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

    vs = cv2.VideoCapture(video_path)
    # initialize the first frame in the video stream
    firstFrame = None
    # loop over the frames of the video
    while True:
        for x in people:
            x.mark_not_updated()

        # grab the current frame and initialize the occupied/unoccupied
        # text
        frame = vs.read()
        frame = frame[1]
        text = "Unoccupied"

        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if frame is None:
            break

        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=700)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (25, 25), 1)

        # if the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = gray
            continue

        frameDelta = cv2.absdiff(firstFrame, gray)

        blur = cv2.GaussianBlur(frameDelta, (3, 3), 0)
        ret, thresh = cv2.threshold(blur, 15, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 100 or cv2.contourArea(c) > 200:
                continue

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
                if person.dist(x, y) < 40 and not (person.updated) and person.is_person_alive():
                    if person.dist(x, y) < 10:
                        person.die()
                        break
                    # print("person from ",person.x," ",person.y, "updated to ",x," ",y)
                    person.update(x, y)
                    person.mark_updated()
                    updated = True
                    break
            if updated == False:
                person_y = Person(x, y, w, h)
                person_y.mark_updated()
                people.append(person_y)
            # draw bounding box

        for person in people:
            if (person.is_alive):
                # cv2.rectangle(frame1, (person.x, person.y), (person.x + person.w, person.y + person.h), (0, 255, 0), 2)
                cv2.rectangle(frame, (person.x, person.y), (person.x + 15, person.y + 25), (0, 255, 0), 2)
            # else:
            # print ("dead person!")

        cv2.imshow("Security Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    vs.release()
    cv2.destroyAllWindows()

main()
for p in people:
    print(p.history)