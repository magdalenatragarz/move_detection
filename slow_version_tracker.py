import cv2
import sys
import math
trackers = []
actual_people = []

def dist(x1,y1,x2,y2):
    return math.hypot(x2 - x1, y2 - y1)

if __name__ == '__main__':

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[2]

    i=0
    multi_tracker = cv2.MultiTracker_create()


    # Read video
    video = cv2.VideoCapture("D:\krk.mp4")

    # Exit if video not opened.
    if not video.isOpened():
        print
        "Could not open video"
        sys.exit()
    # -----------------------------------------------------

    ok, frame1 = video.read()
    ok, frame2 = video.read()

    d = cv2.absdiff(frame1, frame2)

    imgray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(imgray, (1, 1), 0)
    ret, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    _, contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 200 or cv2.contourArea(c) > 400:
            continue

        bbox = cv2.boundingRect(c)

        if bbox[1] < 350:
            continue

        tracker = cv2.TrackerMIL_create()
        #ok = tracker.init()
        trackers.append(tracker)
        multi_tracker.add(tracker,frame1,bbox)
        actual_people.append((bbox[1],bbox[2]))

    if not ok:
        print
        'Cannot read video file'
        sys.exit()


    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        d2 = cv2.absdiff(frame1, frame2)

        imgray = cv2.cvtColor(d2, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imgray, (1, 1), 0)
        ret, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        _, contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 100 or cv2.contourArea(c) > 200:
                continue

            box = cv2.boundingRect(c)

            for x in actual_people:
                if dist(box[0],box[1],x[0],x[1]) < 100:
                    continue

            tracker1 = cv2.TrackerBoosting_create()
            #tracker1.init(frame1,box)
            multi_tracker.add(tracker1,frame1,box);

        ok,boxes = multi_tracker.update(frame1)
        for i,bbox in enumerate(boxes):
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(40) == 27:
            break
        frame1 = frame2
        ret, frame2 = video.read()
    cv2.destroyAllWindows()
    video.release()