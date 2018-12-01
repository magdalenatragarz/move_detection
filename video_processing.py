import cv2
import math
import imutils
import pygame
import people
import colors


people_list = []


def create_video_capture(video_path):
    return cv2.VideoCapture(video_path)


def mark_people_not_updated(people):
    for x in people:
        x.mark_not_updated()

def check_constraints(contour,y):
    flag = True
    if cv2.contourArea(contour) < 70 or cv2.contourArea(contour) > 150:
        flag = False
    if y < 250:
        flag = False
    return flag

def main():
    running = True
    background_colour = (255, 255, 255)
    (width, height) = (700, 500)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Tutorial 1')
    screen.fill(background_colour)

    video_path = 'D:\krk.mp4'
    cap = create_video_capture(video_path)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    counter = 0
    while ret:

        mark_people_not_updated(people_list)
        frame1 = imutils.resize(frame1, width=700)
        frame2 = imutils.resize(frame2, width=700)
        d = cv2.absdiff(frame1, frame2)
        imgray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imgray, (3, 3), 0)
        ret, thresh = cv2.threshold(blur, 15, 255, cv2.THRESH_BINARY)
        final = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, None)
        final = cv2.dilate(final, None, iterations=2)
        _, contours, h = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        for c in contours:

            (x, y, w, h) = cv2.boundingRect(c)

            if not check_constraints(c,y):
                continue

            if not people_list:
                person_x = people.Person(x, y, w, h)
                person_x.mark_updated()
                people_list.append(person_x)
                continue

            updated = False
            for person in people_list:
                if person.dist(x,y) < 30 and not(person.updated): #and person.is_person_alive():
                    if person.dist(x,y) < 10:
                        person.update(x,y)
                        person.mark_updated()
                        person.die()
                        continue
                    if len(person.history) < counter * 0.1:
                        person.die()
                        continue

                    (x1,x2,y1,y2) = person.define_direction()
                    if (x-20 < x1 and x+ 20 > x2 and y - 20 < y1 and y+20 > y2):
                        continue
                    #print("person from ",person.x," ",person.y, "updated to ",x," ",y)
                    person.update(x,y)
                    person.mark_updated()
                    updated = True
                    break
            if updated==False:
                person_y = people.Person(x, y, w, h)
                person_y.mark_updated()
                people_list.append(person_y)
            # draw bounding box

        for person in people_list:
            if (person.is_alive):
                if len(person.history) > 1:
                    pygame.draw.circle(screen, colors.red, [person.x,person.y], 2, 2)
                    pygame.display.update()
                    pygame.display.flip()
                #cv2.rectangle(frame1, (person.x, person.y), (person.x + person.w, person.y + person.h), (0, 255, 0), 2)
                cv2.rectangle(frame1, (person.x, person.y), (person.x+15, person.y+25), (0, 255, 0), 2)
            #else:
                #print ("dead person!")

        cv2.imshow("inter", frame1)
        # cv2.imshow("Thresh", thresh)
        # cv2.imshow("Frame Delta", d)
        # cv2.imshow("Frame Delta", imgray)
        # cv2.imshow("Final", final)

        if cv2.waitKey(40) == 27:
            break

        counter = counter+1
        print("counter: ",counter)
        frame1 = frame2
        ret, frame2 = cap.read()
        # j=j+1`

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    cv2.destroyAllWindows()
    cap.release()

main()

