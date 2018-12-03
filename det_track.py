import cv2
import imutils
import people
import pygame
import colors

VIDEO_PATH = 'D:\krk3.mov'
PEOPLE_LIST = []

def detect():

    (width, height) = (700, 500)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Tutorial 1')
    screen.fill(colors.white)

    cap = cv2.VideoCapture(VIDEO_PATH)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()


    while ret:

        for x in PEOPLE_LIST:
            x.mark_not_updated()

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
                if p.dist(x,y) < 20 and not(person.updated):
                    p.update(x,y)
                    p.mark_updated()
                    updated = True
                    break
            if updated==False:
                person = people.Person(x,y,w,h)
                person.mark_updated()
                PEOPLE_LIST.append(person)
                break

            # cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # pygame.draw.circle(screen,colors.black,[x,y],2,2)
            # pygame.display.update()
            # pygame.display.flip()

        for p in PEOPLE_LIST:
            if p.updated:
                #cv2.rectangle(frame1, (p.x, p.y), (p.x + 15, p.y + 25), (0, 255, 0), 2)
                cv2.rectangle(frame1, (p.x, p.y), (p.x + p.w, p.y + p.h), (0, 255, 0), 2)
                pygame.draw.circle(screen, colors.black, [p.x, p.y], 2, 2)
                pygame.display.update()
                pygame.display.flip()

        cv2.imshow("inter", frame1)
        #cv2.imshow("Final", filtered)
        #cv2.imshow("Final", frame2)
        #cv2.imshow("Final", diff)
        #cv2.imshow("Final", thresh)

        if cv2.waitKey(40) == 27:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        frame1 = frame2
        ret, frame2 = cap.read()
    cv2.destroyAllWindows()
    cap.release()



detect()

# kf = KalmanFilter(dim_x=2, dim_z=2)
# kf.F = np.array([[1., 1.],[0., 1.]])
# kf.H = np.array([[1., 0.]])
# kf.R[2:,2:] *= 10.
# kf.P *= 1000.
# kf.P = np.array([[1000., 0.],[0., 1000.]])
# kf.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
# kf.R = 5
# kf.x = np.array([15., 10.])
#
# kf.update(np.array([17., 11.]).reshape(2,1))
# kf.predict()
# print(kf.x)

ok = []

for p in PEOPLE_LIST:
    if(len(p.history) > 4):
        print(p.history)
        ok.append(p.history)




#people.draw_people(PEOPLE_LIST)


