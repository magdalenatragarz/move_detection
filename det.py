import cv2
import imutils
import people
import pygame
import colors
import numpy as np
import math

VIDEO_PATH = 'D:\krk.mp4'



def is_move_good_for_person(move,person):
    ret = True
    if len(person.history) > 5:
        [coord_x, coord_y] = person.predict_move()
        if dist(move[0], move[1], coord_x, coord_y) < 10 and person.dist(move[0], move[1]) < 20 and person.dist(move[0], move[1]) > 5 :
            ret = False
    else:
        if person.dist(move[0], move[1]) < 20 and person.dist(move[0], move[1]) > 5:
            ret = False
    return ret




def detect():
    POTENTIAL_PEOPLE_LIST = []
    PEOPLE_LIST = []
    NEW_POTENTIAL_LIST = []

    cv2.ocl.setUseOpenCL(False)
    cap = cv2.VideoCapture(VIDEO_PATH)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while ret:

        for x in POTENTIAL_PEOPLE_LIST:
            x.mark_not_updated()

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
        kernel = np.ones((6, 3), np.uint8)
        filtered = cv2.dilate(delete_noises, kernel, iterations=3)

        _, contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)

            if cv2.contourArea(c) < 40:
                 continue

            if y < 250:
                continue

            updated = False
            for p in PEOPLE_LIST:
                if (is_move_good_for_person([x,y], p)):
                    p.update(x, y)
                    p.mark_updated()
                    updated = True
                    continue
                if not updated:
                    [coord_x, coord_y] = p.predict_move()
                    p.update(coord_x, coord_y)
                    break


            if not POTENTIAL_PEOPLE_LIST:
                person = people.Person(x,y,w,h)
                person.mark_updated()
                POTENTIAL_PEOPLE_LIST.append(person)
                break

            updated = False
            for p in POTENTIAL_PEOPLE_LIST :
                if len(p.history) > 5:
                    [coord_x, coord_y] = p.predict_move()
                    if dist(x,y,coord_x,coord_y) < 20 and not p.updated:
                        p.update(x,y)
                        p.mark_updated()
                        updated = True
                        break
                else:
                    if p.dist(x,y) < 20 and not p.updated:
                        p.update(x,y)
                        p.mark_updated()
                        updated = True
                        break
            if not updated:
                person = people.Person(x,y,w,h)
                person.mark_updated()
                POTENTIAL_PEOPLE_LIST.append(person)
                break


        NEW_POTENTIAL_LIST = POTENTIAL_PEOPLE_LIST
        for pp in POTENTIAL_PEOPLE_LIST:
            if len(pp.history) > 5:
                PEOPLE_LIST.append(pp)
                NEW_POTENTIAL_LIST.remove(pp)
        POTENTIAL_PEOPLE_LIST = NEW_POTENTIAL_LIST

        for p in PEOPLE_LIST:
           if p.updated:
                cv2.rectangle(frame1, (p.x, p.y), (p.x + p.w, p.y + p.h), colors.blue, 2)

        for pp in POTENTIAL_PEOPLE_LIST:
            if pp.updated:
                cv2.rectangle(frame1, (pp.x, pp.y), (pp.x + pp.w, pp.y + pp.h), colors.red, 2)
                print(pp.history)




        cv2.imshow("inter", frame1)
        cv2.imshow("Final", filtered)

        if cv2.waitKey(40) == 27:
            break

        ret, frame1 = cap.read()
        ret, frame2 = cap.read()
    cv2.destroyAllWindows()
    cap.release()


def dist(x1,y1,x2,y2):
    return math.hypot(x1 - x2, y1 - y2)


detect()