import cv2
import imutils

cap = cv2.VideoCapture("D:\krk3.mov")
bboxes = []
_,frame = cap.read()
frame = imutils.resize(frame, width=900)

bbox1 = cv2.selectROI('MultiTracker', frame)
bboxes.append(bbox1)

bbox2 = cv2.selectROI('MultiTracker', frame)
bboxes.append(bbox2)

bbox3 = cv2.selectROI('MultiTracker', frame)
bboxes.append(bbox3)

bbox4 = cv2.selectROI('MultiTracker', frame)
bboxes.append(bbox4)



trackers = cv2.MultiTracker_create()

for bbox in bboxes:
  trackers.add(cv2.TrackerCSRT_create(), frame, bbox)


while True:
    # Read a new frame
    ok, frame = cap.read()
    frame = imutils.resize(frame, width=900)

    (success, boxes) = trackers.update(frame)

    if(success):
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Start tier
    timer = cv2.getTickCount()

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);


    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

