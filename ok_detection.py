import cv2
import math
import imutils
import pygame

people = []

white = (255, 255, 255)
blue = (0, 0, 255)
red = (255, 0, 0)
black = (0, 0, 0)
beige = (164, 168, 128)
purple = (75, 0, 130)
lb = (253, 245, 230)
lr = (255, 99, 71)
orange = (255, 165, 0)
yellow = (255, 255, 255)
ly = (255, 255, 51)
colors = [blue,red,black,beige,purple,lb,lr,orange,yellow,ly]

def draw_people():
    running = True
    background_colour = (255, 255, 255)
    (width, height) = (700, 500)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Tutorial 1')
    screen.fill(background_colour)
    i=0
    for person in people_ok:
        if (len(person.history) > 1):
            pygame.draw.lines(screen, colors[i % len(colors)], False, person.history, 2)
            print(person.history)
            pygame.display.update()
            pygame.display.flip()
            i += 1
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

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
        self.history.append([x,y])
        self.changes = 0
        self.is_alive = True

    def update(self,x,y):
        self.x = x
        self.y = y
        #self.w = w
        #self.h = h
        self.history.append([x,y])


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

    cap = cv2.VideoCapture(video_path)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    background_colour = (255, 255, 255)
    (width, height) = (700, 500)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Tutorial 1')
    screen.fill(background_colour)

    while ret:

        for x in people:
            x.mark_not_updated()

        _, frame = cap.read()
        frame1 = imutils.resize(frame1, width=700)
        frame2 = imutils.resize(frame2, width=700)

        d = cv2.absdiff(frame1, frame2)
        imgray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        #blur = cv2.GaussianBlur(imgray, (3, 3), 0)
        ret, thresh = cv2.threshold(imgray, 15, 255, cv2.THRESH_BINARY)
        final = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,None)
        final = cv2.dilate(final, None, iterations=2)
        _, contours, h = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 100 or cv2.contourArea(c) > 150:
                continue

            (x, y, w, h) = cv2.boundingRect(c)

            if y < 250:
                continue

            if not people:
                person_x = Person(x, y, w, h)
                person_x.mark_updated()
                people.append(person_x)
                break

            updated = False
            for person in people:
                if person.dist(x,y) < 50 and not(person.updated): #and person.is_person_alive():
                    if person.dist(x,y) < 5 :
                        person.mark_updated()
                        break
                    person.update(x,y)
                    person.mark_updated()
                    updated = True
                    break
            if updated==False:
                person_y = Person(x, y, w, h)
                person_y.mark_updated()
                people.append(person_y)
                break

        for person in people:
            if (person.is_alive) and len(person.history)>3:
                cv2.rectangle(frame1, (person.x, person.y), (person.x+15, person.y+25), (0, 255, 0), 2)

        cv2.imshow("inter", frame1)
        #cv2.imshow("blur", blur)
        #cv2.imshow("Thresh", thresh)
        #cv2.imshow("Frame Delta", d)
        #cv2.imshow("Frame Delta", imgray)
        cv2.imshow("Final", final)

        if cv2.waitKey(40) == 27:
            break

        frame1 = frame2
        ret, frame2 = cap.read()


    cv2.destroyAllWindows()
    cap.release()


main()


people_ok =[]
for p in people:
    if p.is_alive and len(p.history)>2:
        print(p.history)
        people_ok.append(p)

running = True
background_colour = (255, 255, 255)
(width, height) = (700, 500)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Tutorial 1')
screen.fill(background_colour)
i=0
for person in people_ok:
    if (len(person.history) > 4):
        pygame.draw.lines(screen, colors[i % len(colors)], False, person.history, 2)
        print(person.history)
        pygame.display.update()
        pygame.display.flip()
        i += 1
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

