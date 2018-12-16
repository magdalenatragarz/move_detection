import cv2
import imutils
import people
import pygame
import colors
import numpy as np
import math

VIDEO_PATH = 'D:\krk.mp4'
PEOPLE_LIST = []
potential_people = []
moves_after_people = []
moves_after_potential_people = []

def dist(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def potential_person_to_person(potential_people,people):
    new_potential_people = []
    for pp in potential_people:
        if len(pp.history) >= 5:
            people.append(pp)
        else:
            new_potential_people.append(pp)
    return  (new_potential_people,people)

def is_move_good_for_person(move,person):
    ret = True
    (x, y, w, h) = cv2.boundingRect(move)
    if cv2.contourArea(move) < 40:
        return False
    if y < 250:
        return False
    if len(person.history) > 5:
        [coord_x, coord_y] = person.predict_move()
        if dist(x, y, coord_x, coord_y) < 10 and person.dist(x, y) < 20 and person.dist(x, y) > 5 :
            ret = False
    else:
        if person.dist(x, y) < 20 and person.dist(x, y) > 5:
            ret = False
    return ret


def detect():
    potential_people = []
    real_people = []

    cv2.ocl.setUseOpenCL(False)
    cap = cv2.VideoCapture(VIDEO_PATH)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while ret:
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
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, None)

        _, all_moves, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("all_moves ok")
        moves_after_people = all_moves;

        if len(real_people)!=0 :
            moves_after_people.clear()

        updated = False
        for p in real_people:
            for move in all_moves:
                (x, y, w, h) = cv2.boundingRect(move)
                if cv2.contourArea(move) < 40:
                    continue
                if y < 250:
                    continue

                if (is_move_good_for_person(move, p)):
                    p.update(x, y)
                    updated = True
                else:
                    moves_after_people.append(move)
            if not updated:
                [coord_x, coord_y] = p.predict_move()
                p.update(coord_x, coord_y)


        for move in moves_after_people:
            (x, y, w, h) = cv2.boundingRect(move)
            if cv2.contourArea(move) < 40:
                continue
            if y < 250:
                continue

            if not potential_people:
                person = people.Person(x, y, w, h)
                potential_people.append(person)
                continue

            for pp in potential_people:
                if (is_move_good_for_person(move,pp)):
                    pp.update(x,y)

        (new_potential_people,new_real_people) = potential_person_to_person(potential_people, real_people)
        potential_people = new_potential_people
        real_people = new_real_people

        for p in real_people:
            cv2.rectangle(frame1, (p.x, p.y), (p.x + p.w, p.y + p.h), colors.blue, 2)


        for pp in potential_people:
            cv2.rectangle(frame1, (pp.x, pp.y), (pp.x + pp.w, pp.y + pp.h), colors.red, 2)

        # ========
        #
        #
        # for m in moves:
        #     (x, y, w, h) = cv2.boundingRect(m)
        #
        #     if cv2.contourArea(m) < 40:
        #          continue
        #
        #     if y < 250:
        #         continue
        #
        #
        #
        #     updated = False
        #     for p in PEOPLE_LIST :
        #         if len(p.history) > 5:
        #             [coord_x, coord_y] = p.predict_move()
        #             if  dist(x,y,coord_x,coord_y) < 10 and p.dist(x,y) < 20 and p.dist(x,y) > 5 and not p.updated:
        #                 p.update(x,y)
        #                 p.mark_updated()
        #                 updated = True
        #                 break
        #         else:
        #             if p.dist(x,y) < 20 and p.dist(x,y) > 5 and not p.updated:
        #                 p.update(x,y)
        #                 p.mark_updated()
        #                 updated = True
        #                 break
        #
        #
        #     if not updated and x < 20 and x > 680 and y < 270 and counter > 3:
        #         person = people.Person(x,y,w,h)
        #         person.mark_updated()
        #         PEOPLE_LIST.append(person)
        #         break
        #     elif not updated and counter >= 3:
        #         person = people.Person(x, y, w, h)
        #         person.mark_updated()
        #         PEOPLE_LIST.append(person)
        #         break
        #     else:
        #         potential_people.append((x,y))
        #
        # for p in PEOPLE_LIST:
        #     if p.updated:
        #         cv2.rectangle(frame1, (p.x, p.y), (p.x + p.w, p.y + p.h), (0, 255, 0), 2)
        #     else:
        #         p.how_long_not_updated += 1


        cv2.imshow("inter", frame1)
        cv2.imshow("Final", filtered)

        if cv2.waitKey(40) == 27:
            break

        ret, frame1 = cap.read()
        ret, frame2 = cap.read()
    cv2.destroyAllWindows()
    cap.release()


    for p in real_people:
        print(p.history)

    for p in potential_people:
        print(p.history)




detect()


