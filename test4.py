import cv2
import sys

trackers = []

if __name__ == '__main__':

    # Set up tracker.
    # Instead of MIL, you can also use

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
    # Read first frame.

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

        # get bounding box from countour
        bbox = cv2.boundingRect(c)

        if bbox[1] < 350:
            continue

        tracker = cv2.TrackerBoosting_create()
        ok = tracker.init(frame1, bbox)
        trackers.append(tracker)
        multi_tracker.add(tracker,frame1,bbox)

    if not ok:
        print
        'Cannot read video file'
        sys.exit()

    # Define an initial bounding box
    #bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    #bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box


    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        #timer = cv2.getTickCount()

        # d = cv2.absdiff(frame1, frame2)
        #
        # imgray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(imgray, (1, 1), 0)
        # ret, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        # thresh = cv2.dilate(thresh, None, iterations=2)
        # _, contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Calculate Frames per second (FPS)
        # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        # for c in contours:
        #     if cv2.contourArea(c) < 200 or cv2.contourArea(c) > 400:
        #         continue
        #
        #     # get bounding box from countour
        #     (x, y, w, h) = cv2.boundingRect(c)
        #
        #     if y < 450:
        #         continue

        # Update tracker

        d2 = cv2.absdiff(frame1, frame2)

        imgray = cv2.cvtColor(d2, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imgray, (1, 1), 0)
        ret, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        _, contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 100 or cv2.contourArea(c) > 200:
                continue

            # get bounding box from contour
            box = cv2.boundingRect(c)
            tracker1 = cv2.TrackerBoosting_create()
            tracker1.init(frame1,box)
            multi_tracker.add(tracker1,frame1,box);

        ok,boxes = multi_tracker.update(frame1)
        #for tracker in trackers:
        for i,bbox in enumerate(boxes):
            #i = i+1
            #ok, bbox = tracker.update(frame)
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)


        # Display tracker type on frame
        #cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        #cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        if cv2.waitKey(40) == 27:
            break
        frame1 = frame2
        ret, frame2 = video.read()
    cv2.destroyAllWindows()
    video.release()