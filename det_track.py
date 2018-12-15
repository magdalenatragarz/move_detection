import cv2
import imutils
import people
import pygame
import colors
import numpy as np
import math

VIDEO_PATH = 'D:\krk.mp4'
PEOPLE_LIST = []


def detect():

    (width, height) = (700, 500)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Tutorial 1')
    screen.fill(colors.white)

    cv2.ocl.setUseOpenCL(False);
    cap = cv2.VideoCapture(VIDEO_PATH)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while ret:

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
        kernel = np.ones((5, 5), np.uint8)
        filtered = cv2.dilate(delete_noises, kernel, iterations=1)

        _, contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)

            if cv2.contourArea(c) < 40:
                 continue

            if y < 250:
                continue

            if not PEOPLE_LIST:
                person = people.Person(x,y,w,h)
                person.mark_updated()
                PEOPLE_LIST.append(person)
                break

            updated = False
            for p in PEOPLE_LIST :
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
                PEOPLE_LIST.append(person)
                break

        for p in PEOPLE_LIST:
            if p.updated:
                cv2.rectangle(frame1, (p.x, p.y), (p.x + p.w, p.y + p.h), (0, 255, 0), 2)
                pygame.draw.circle(screen, colors.black, [p.x, p.y], 2, 2)
                pygame.display.update()
                pygame.display.flip()


        cv2.imshow("inter", frame1)
        cv2.imshow("Final", filtered)

        if cv2.waitKey(40) == 27:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        frame1 = frame2
        ret, frame2 = cap.read()
    cv2.destroyAllWindows()
    cap.release()


def dist(x1,y1,x2,y2):
    return math.hypot(x1 - x2, y1 - y2)


detect()


