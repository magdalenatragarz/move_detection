import cv2
import imutils

cap = cv2.VideoCapture("D:\krk3.mov")

_,frame = cap.read()
frame = imutils.resize(frame, width=900)

trackers = cv2.MultiTracker_create()
# bbox1 = cv2.selectROI(frame, False)
# tracker1 = cv2.TrackerMOSSE_create()
# ok = tracker1.init(frame, bbox1)
#
# bbox2 = cv2.selectROI(frame, False)
# tracker2 = cv2.TrackerMOSSE_create()
# ok = tracker2.init(frame, bbox2)
#
#
# bbox3 = cv2.selectROI(frame, False)
# tracker3 = cv2.TrackerMOSSE_create()
# ok = tracker3.init(frame, bbox3)

box = cv2.selectROI("Frame", frame, fromCenter=False)
tracker = cv2.TrackerMOSSE_create()
#trackers.add(tracker, frame, box)
ok =tracker.init(frame,box)

box2 = cv2.selectROI("Frame", frame, fromCenter=False)
tracker2 = cv2.TrackerMOSSE_create()
#trackers.add(tracker2, frame, box2)
ok =tracker2.init(frame,box2)

box3 = cv2.selectROI("Frame", frame, fromCenter=False)
tracker3 = cv2.TrackerMOSSE_create()
#trackers.add(tracker3, frame, box3)
ok =tracker3.init(frame,box3)

box4 = cv2.selectROI("Frame", frame, fromCenter=False)
tracker4 = cv2.TrackerMOSSE_create()
#trackers.add(tracker, frame, box)
ok =tracker4.init(frame,box4)

box5 = cv2.selectROI("Frame", frame, fromCenter=False)
tracker5 = cv2.TrackerMOSSE_create()
#trackers.add(tracker2, frame, box2)
ok =tracker5.init(frame,box5)

box6 = cv2.selectROI("Frame", frame, fromCenter=False)
tracker6 = cv2.TrackerMOSSE_create()
#trackers.add(tracker3, frame, box3)
ok =tracker6.init(frame,box6)

box7 = cv2.selectROI("Frame", frame, fromCenter=False)
tracker7 = cv2.TrackerMOSSE_create()
#trackers.add(tracker, frame, box)
ok =tracker7.init(frame,box7)

while True:
    # Read a new frame
    ok, frame = cap.read()
    frame = imutils.resize(frame, width=900)

    #(success, boxes) = trackers.update(frame)

    (ok, box) = tracker.update(frame)
    (ok1, box1) = tracker2.update(frame)
    (ok2, box2) = tracker3.update(frame)
    (ok3, box3) = tracker3.update(frame)
    (ok4, box4) = tracker4.update(frame)
    (ok5, box5) = tracker5.update(frame)
    (ok6, box6) = tracker6.update(frame)
    (ok7, box7) = tracker7.update(frame)

    # Start timer
    timer = cv2.getTickCount()



    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    if(ok):
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if (ok1):
        (x, y, w, h) = [int(v) for v in box1]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if (ok2):
        (x, y, w, h) = [int(v) for v in box2]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if (ok3):
        (x, y, w, h) = [int(v) for v in box3]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if (ok4):
        (x, y, w, h) = [int(v) for v in box4]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if (ok5):
        (x, y, w, h) = [int(v) for v in box5]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if (ok6):
        (x, y, w, h) = [int(v) for v in box6]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if (ok7):
        (x, y, w, h) = [int(v) for v in box7]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display tracker type on frame

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

