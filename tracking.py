import cv2
import imutils

cap = cv2.VideoCapture("D:\krk.mp4")

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

# select the bounding box of the object we want to track (make
# sure you press ENTER or SPACE after selecting the ROI)
box = cv2.selectROI("Frame", frame, fromCenter=False)
# create a new object tracker for the bounding box and add it
# to our multi-object tracker
tracker = cv2.TrackerKCF_create()
trackers.add(tracker, frame, box)

box2 = cv2.selectROI("Frame", frame, fromCenter=False)
# create a new object tracker for the bounding box and add it
# to our multi-object tracker
tracker2 = cv2.TrackerKCF_create()
trackers.add(tracker2, frame, box2)

box3 = cv2.selectROI("Frame", frame, fromCenter=False)
# create a new object tracker for the bounding box and add it
# to our multi-object tracker
tracker3 = cv2.TrackerKCF_create()
trackers.add(tracker3, frame, box3)


while True:
    # Read a new frame
    ok, frame = cap.read()
    frame = imutils.resize(frame, width=900)

    (success, boxes) = trackers.update(frame)

    # Start timer
    timer = cv2.getTickCount()



    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box

    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
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

