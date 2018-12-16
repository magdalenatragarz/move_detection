import cv2
import imutils
import people
import numpy as np
import math

def detect():
    VIDEO_PATH = 'D:\krk3.mov'
    PEOPLE_LIST = []
    TRACKERS = []

    cv2.ocl.setUseOpenCL(False);
    cap = cv2.VideoCapture(VIDEO_PATH)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    frame_count = 0
    while ret:

        ok, frame = cap.read()
        frame = imutils.resize(frame, width=700)

        for x in PEOPLE_LIST:
            x.mark_not_updated()

        frame1 = imutils.resize(frame1, width=700)
        frame2 = imutils.resize(frame2, width=700)
        diff = cv2.absdiff(frame1, frame2)
        imgray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imgray, (5, 5), 1)
        ret, thresh = cv2.threshold(blur, 15, 255, cv2.THRESH_BINARY)
        delete_noises = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, None)
        kernel1 = np.ones((3, 3), np.uint8)
        filtered1 = cv2.erode(delete_noises, kernel1, iterations=1)
        kernel = np.ones((6, 2), np.uint8)
        filtered = cv2.dilate(filtered1, kernel, iterations=3)

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
                    if dist(x,y,coord_x,coord_y) < 10 and not p.updated and not p.tracker_initialized:
                        p.update(x,y,frame_count)
                        p.mark_updated()
                        updated = True
                        if len(p.history) >= 10 and len(TRACKERS) <= 10:
                            tracker = cv2.TrackerMOSSE_create()
                            print("initializing tracker")
                            ok = tracker.init(frame, p.get_current_bounding_box())
                            print(p.get_current_bounding_box())
                            print(ok)
                            if ok:
                                p.tracker_initialized = True
                                TRACKERS.append(tracker)
                        break
                else:
                    if p.dist(x,y) < 20 and not p.updated and not p.tracker_initialized:
                        p.update(x,y,frame_count)
                        p.mark_updated()
                        updated = True
                        if len(p.history) >= 10 and len(TRACKERS) <= 10:
                            tracker = cv2.TrackerMOSSE_create()
                            print("initializing tracker")
                            ok = tracker.init(frame, p.get_current_bounding_box())
                            print(p.get_current_bounding_box())
                            print(ok)
                            if ok:
                                p.tracker_initialized = True
                                TRACKERS.append(tracker)
                        break
            if not updated:
                person = people.Person(x,y,frame_count)
                person.mark_updated()
                PEOPLE_LIST.append(person)
                break

        for t in TRACKERS:
            (ok,box) = t.update(frame)
            print("updated tracker" )
            print(ok)
            if ok:
                (x, y, w, h) = [int(v) for v in box]

                cv2.rectangle(frame, (x, y), (x+15,y+25), (0, 255, 0), 2)

        cv2.imshow("inter", frame)
        cv2.imshow("",filtered)

        if cv2.waitKey(40) == 27:
            break

        frame_count += 1
        frame1 = frame2
        ret, frame2 = cap.read()

    cv2.destroyAllWindows()
    cap.release()


    for p in PEOPLE_LIST:
        print(p.history)

def dist(x1,y1,x2,y2):
    return math.hypot(x1 - x2, y1 - y2)


detect()
