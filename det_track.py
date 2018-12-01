import cv2
import imutils
import people
import numpy as np

VIDEO_PATH = 'D:\krk3.mov'
PEOPLE_LIST = []

def detect():

    cap = cv2.VideoCapture(VIDEO_PATH)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while ret:

        frame1 = imutils.resize(frame1, width=700)
        frame2 = imutils.resize(frame2, width=700)
        diff = cv2.absdiff(frame1, frame2)
        imgray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imgray, (5, 5), 1)
        ret, thresh = cv2.threshold(blur, 15, 255, cv2.THRESH_BINARY)
        #deleteNoises = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, None)
        filtered = cv2.dilate(thresh, None, iterations=2)

        _, contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)

            # if cv2.contourArea(c) < 20 or cv2.contourArea(c) > 150:
            #      continue

            if w/h > 0.8 or w/h < 0.2:
                continue

            if y < 250:
                continue

            if not PEOPLE_LIST:
                person = people.Person(x,y,w,h)
                person.update(x,y)
                person.mark_updated()
                PEOPLE_LIST.append(person)
            else:
                for p in PEOPLE_LIST:
                    if p.dist(x,y) < 20:
                        p.update(x,y)
                        p.mark_updated()
                        break
                person = people.Person(x,y,w,h)
                person.update(x,y)
                person.mark_updated()
                PEOPLE_LIST.append(person)

            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("inter", frame1)
        cv2.imshow("Final", filtered)

        if cv2.waitKey(40) == 27:
            break

        frame1 = frame2
        ret, frame2 = cap.read()
    cv2.destroyAllWindows()
    cap.release()



#detect()

kalman = cv2.KalmanFilter()
kalman.transitionMatrix = np.array([[1., 1.], [0., 1.]])
kalman.measurementMatrix = 1. * np.ones((1, 2))
kalman.processNoiseCov = 1e-5 * np.eye(2)
kalman.measurementNoiseCov = 1e-1 * np.ones((1, 1))
kalman.errorCovPost = 1. * np.ones((2, 2))
kalman.statePost = 0.1 * np.random.randn(2, 1)

for p in PEOPLE_LIST:
    if(len(p.history) > 4):
        print(p.history)

#people.draw_people(PEOPLE_LIST)


