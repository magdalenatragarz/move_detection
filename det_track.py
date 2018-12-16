import cv2
import imutils
import people
import numpy as np
import math
import common
import colors

VIDEO_PATH = 'D:\krakau.mov'
PEOPLE_LIST = []
REAL_PEOPLE = []

def detect():

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
        kernel = np.ones(common.DILATE_VECTOR, np.uint8)
        filtered = cv2.dilate(delete_noises, kernel, iterations=common.DILATE_ITERATIONS)

        _, contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)

            if cv2.contourArea(c) < common.MINIMUM_CONTOUR_AREA:
                 continue

            if y < common.SKYLINE:
                continue

            updated = False
            for p in PEOPLE_LIST :
                if len(p.history) > common.CREDIBLE_HISTORY_LENGTH:
                    [coord_x, coord_y] = p.predict_move()
                    if dist(x,y,coord_x,coord_y) < common.CLOSE_NEIGHBOURHOOD and not p.updated and p.get_standard_deviation_with_new_point(x,y) < common.MAX_DEVIATION and (frame_count - p.frame_count)<common.MAX_FRAME_DIFFERENCE:
                        p.update(x,y,frame_count)
                        p.mark_updated()
                        p.how_many_predicted = 0
                        updated = True
                        break
                else:
                    if p.dist(x,y) < common.MAX_DISTANCE  and not p.updated and p.get_standard_deviation_with_new_point(x,y) < common.MAX_DEVIATION and (frame_count - p.frame_count)<common.MAX_FRAME_DIFFERENCE :
                        p.update(x,y,frame_count)
                        p.mark_updated()
                        p.how_many_predicted = 0
                        updated = True
                        break
            if not updated:
                person = people.Person(x,y,frame_count)
                person.mark_updated()
                PEOPLE_LIST.append(person)

        for p in PEOPLE_LIST:
            if not p.updated and len(p.history) > common.CREDIBLE_HISTORY_LENGTH and p.get_standard_deviation() < common.MAX_DEVIATION and p.how_many_predicted < common.MAX_PREDICTIONS_QUANTITY and (frame_count - p.frame_count)<common.MAX_FRAME_DIFFERENCE:
                [coord_x, coord_y] = p.predict_move()
                p.update(coord_x, coord_y, frame_count)
                p.how_many_predicted += 1
                p.mark_updated()

        for p in PEOPLE_LIST:
            if p.updated:
                if len(p.history) > common.CREDIBLE_HISTORY_LENGTH and p.get_standard_deviation() < common.MAX_DEVIATION:
                    cv2.rectangle(frame1, (p.x, p.y), (p.x + common.BBOX_WIDTH, p.y + common.BBOX_HEIGHT), colors.blue, common.LINE_WIDTH)
                    index = PEOPLE_LIST.index(p)
                    cv2.putText(frame1, str(index + 1), (p.x - common.TEXT_MARIGIN, p.y - common.TEXT_MARIGIN), cv2.FONT_HERSHEY_SIMPLEX, common.TEXT_SIZE, colors.black, common.LINE_WIDTH)

        cv2.imshow(common.WINDOW_NAME, frame1)

        if cv2.waitKey(common.DELIM_WAIT) == common.ESC_PRESSED:
            break

        frame1 = frame2
        ret, frame2 = cap.read()

    cv2.destroyAllWindows()
    cap.release()



def dist(x1,y1,x2,y2):
    return math.hypot(x1 - x2, y1 - y2)

long_people = []
detect()
for p in PEOPLE_LIST:
    if len(p.history) > 10 :
        long_people.append(p)

people.draw_people(long_people)