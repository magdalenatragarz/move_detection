# import cv2
#
#
# def main():
#
#     video_path = 'D:\krk.mp4'
#     cv2.ocl.setUseOpenCL(False)
#
#     tracker = cv2.TrackerBoosting_create()
#
#
#     #read video file
#     cap = cv2.VideoCapture(video_path)
#
#     ret, frame1 = cap.read()
#     ret, frame2 = cap.read()
#
#     bbox = (287, 23, 86, 320)
#     ok = tracker.init(frame1, bbox)
#
#
#     while ret:
#         i = 1
#         _, frame = cap.read()
#
#         d = cv2.absdiff(frame1, frame2)
#
#         imgray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
#         blur = cv2.GaussianBlur(imgray, (1, 1), 0)
#         ret, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
#         thresh = cv2.dilate(thresh, None, iterations=2)
#         _, contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         for c in contours:
#             if cv2.contourArea(c) < 200 or cv2.contourArea(c) > 400:
#                 continue
#
#             # get bounding box from countour
#             (x, y, w, h) = cv2.boundingRect(c)
#
#             if y < 450:
#                 continue
#
#             # draw bounding box
#             cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#
#         cv2.imshow("inter", frame1)
#
#         if cv2.waitKey(40) == 27:
#             break
#
#         frame1 = frame2
#         ret, frame2 = cap.read()
#         # j=j+1
#     cv2.destroyAllWindows()
#     cap.release()
#
#
# main()

# installed with whl files!!!!!


import cv2
import sys


if __name__ == '__main__':

    # Set up tracker.
    # Instead of MIL, you can also use

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[2]


    tracker = cv2.TrackerBoosting_create()


    # Read video
    video = cv2.VideoCapture("D:\krk.mp4")

    # Exit if video not opened.
    if not video.isOpened():
        print
        "Could not open video"
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print
        'Cannot read video file'
        sys.exit()

    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break