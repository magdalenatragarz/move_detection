import datetime
import imutils
import cv2
import math

people = []


class Person:
    x = 0
    y = 0
    w = 0
    h = 0

    current_state = (x,y)
    history = []

    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.current_state = (x,y)
        self.history.append((x,y,w,h))

    def update(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.current_state = (x, y)
        self.history.append((x, y, w, h))

    def dist(self,x,y):
        return math.hypot(x - self.x, y - self.y)
# --------------------------------------


def main():
    video_path = 'D:\krk4.mp4'

    vs = cv2.VideoCapture(video_path)
    # initialize the first frame in the video stream
    firstFrame = None
    # loop over the frames of the video
    while True:
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

        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 50 or cv2.contourArea(c) > 200:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x,y,w,h) = cv2.boundingRect(c)

            if y < 250:
                continue

            if not people:
                people.append(Person(x, y, w, h))
                continue

            updated = False
            print(len(people))
            for person in people:
                print("porownanie wspolrzednych:", x, ", ",y , " z ", person.x, ", ", person.y)
                print("wynik:")
                if person.dist(x,y) < 50:
                    person.update(x,y,w,h)
                    print("update")
                    updated = True
                    break
            if updated==False:
                people.append(Person(x, y, w, h))
                print("dodaje nowa")


            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for person in people:
            cv2.rectangle(frame, (person.x, person.y), (person.x + person.w, person.y + person.h), (0, 255, 0), 2)

        # draw the text and timestamp on the frame
        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # show the frame and record if the user presses a key
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

    # cleanup the camera and close any open windows
    vs.release()
    cv2.destroyAllWindows()


#x = Person(1,2,3,4)

#print(x.dist(2,5))

main()

