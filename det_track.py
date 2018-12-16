import cv2
import imutils
import people
import pygame
import colors
import numpy as np
import math

VIDEO_PATH = 'D:\krk3.mov'
PEOPLE_LIST = []


def detect():

    cv2.ocl.setUseOpenCL(False)
    cap = cv2.VideoCapture(VIDEO_PATH)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    frame_count = 0
    while ret:
        frame_count += 1

        for x in PEOPLE_LIST:
            x.mark_not_updated()

        _, frame = cap.read()
        frame1 = imutils.resize(frame1, width=700)
        frame2 = imutils.resize(frame2, width=700)
        diff = cv2.absdiff(frame1, frame2)
        imgray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imgray, (5, 5), 1)
        ret, thresh = cv2.threshold(blur, 15, 255, cv2.THRESH_BINARY)
        delete_noises = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, None)
        kernel = np.ones((6,3), np.uint8)
        filtered = cv2.dilate(delete_noises, kernel, iterations=3)

        _, contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)

            if cv2.contourArea(c) < 40:
                 continue

            if y < 250:
                continue

            if not PEOPLE_LIST:
                person = people.Person(x,y,frame_count)
                person.mark_updated()
                PEOPLE_LIST.append(person)
                break

            updated = False
            for p in PEOPLE_LIST :
                if len(p.history) > 5:
                    [coord_x, coord_y] = p.predict_move()
                    if dist(x,y,coord_x,coord_y) < 5 and not p.updated:
                        p.update(x,y,frame_count)
                        p.mark_updated()
                        updated = True
                        break
                else:
                    [coord_x, coord_y] = p.predict_move()
                    if p.dist(x,y) < 10 and p.dist(x,y) > 0 and not p.updated:
                        p.update(x,y,frame_count)
                        p.mark_updated()
                        updated = True
                        break
            if not updated:
                person = people.Person(x,y,frame_count)
                person.mark_updated()
                PEOPLE_LIST.append(person)
                break

        for p in PEOPLE_LIST:
            if not p.updated and len(p.history) > 10 and p.get_standard_deviation() < 0.01 and p.how_many_predicted < 3:
                [coord_x, coord_y] = p.predict_move()
                p.update(coord_x, coord_y, frame_count)
                p.how_many_predicted += 1
                p.mark_updated()

        for p in PEOPLE_LIST:
            if p.updated:
                if len(p.history) > 15 and p.get_standard_deviation() < 0.01:
                    cv2.rectangle(frame1, (p.x, p.y), (p.x + 15, p.y + 25), (0, 255, 0), 2)
                    index = PEOPLE_LIST.index(p)
                    print(index)
                    cv2.putText(frame1, str(index + 1), (p.x - 10, p.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.imshow("inter", frame1)
        cv2.imshow("inter2", filtered)

        if cv2.waitKey(40) == 27:
            break

        frame1 = frame2
        ret, frame2 = cap.read()
    cv2.destroyAllWindows()
    cap.release()



def dist(x1,y1,x2,y2):
    return math.hypot(x1 - x2, y1 - y2)


detect()
for p in PEOPLE_LIST:
    if len(p.history) > 10:
        index = PEOPLE_LIST.index(p)
        print(index)
        print(p.history_x)
        print(p.history_y)
