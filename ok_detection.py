import cv2
import math

people = []


class Person:
    x = 0
    y = 0
    w = 0
    h = 0

    is_alive =True
    changes = 0

    current_state = (x,y)
    history = []

    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.current_state = (x,y)
        self.history.append((x,y,w,h))
        self.is_alive=True

    def update(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.current_state = (x, y)
        self.history.append((x, y, w, h))

    def dist(self,x,y):
        return math.hypot(x - self.x, y - self.y)

    def die(self):
        self.is_alive = False
# --------------------------------------



def main():

    video_path = 'D:\krk.mp4'
    cv2.ocl.setUseOpenCL(False)

    #read video file
    cap = cv2.VideoCapture(video_path)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    counter = 0
    while ret:

        _, frame = cap.read()

        d = cv2.absdiff(frame1, frame2)

        imgray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imgray, (1, 1), 0)
        ret, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        _, contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 200 or cv2.contourArea(c) > 400:
                continue

            # get bounding box from countour
            (x, y, w, h) = cv2.boundingRect(c)

            if y < 550:
                continue

            if not people:
                people.append(Person(x, y, w, h))
                continue

            updated = False
            print(len(people))
            for person in people:
                if person.dist(x,y) < 80:
                    print("person from ",person.x," ",person.y, "updated to ",x," ",y)
                    person.update(x,y,w,h)
                    updated = True
                    person.changes = person.changes +1
                    break
            if updated==False:
                people.append(Person(x, y, w, h))
            # draw bounding box

        for person in people:
            if (counter-person.changes > 5):
                person.die()

        for person in people:
            if (person.is_alive):
                cv2.rectangle(frame1, (person.x, person.y), (person.x + person.w, person.y + person.h), (0, 255, 0), 2)
            else:
                print ("dead person!")

        cv2.imshow("inter", frame1)

        if cv2.waitKey(40) == 27:
            break

        counter = counter+1
        frame1 = frame2
        ret, frame2 = cap.read()
        # j=j+1`

    cv2.destroyAllWindows()
    cap.release()


main()

