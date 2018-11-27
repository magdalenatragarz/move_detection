import cv2
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression


def main():

    video_path = 'D:\krk.mp4'

    #read video file
    cap = cv2.VideoCapture(video_path)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    if cap.isOpened():

        ret, frame = cap.read()

    else:
        ret = False

    fgbg = cv2.createBackgroundSubtractorMOG2()
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    while ret:
        gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        (rects, weights) = hog.detectMultiScale(gray, winStride=(8, 8), padding=(16, 16), scale=1.06)
        rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        for (x, y, w, h) in rects:
            # if cv2.contourArea(c) < 100 or cv2.contourArea(c) > 200:
            #     continue

            # get bounding box from contour
            #(x, y, w, h) = cv2.boundingRect(c)

            # draw bounding box
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # cv2.imshow("win1",frame2)
        cv2.imshow("inter", frame1)

        if cv2.waitKey(40) == 27:
            break
        frame1 = frame2
        ret, frame2 = cap.read()

    cv2.destroyAllWindows()
    cap.release()


main()