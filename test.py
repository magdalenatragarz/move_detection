import cv2
import numpy as np

def main():

    video_path = 'D:\krk.mp4'
    cv2.ocl.setUseOpenCL(False)

    #read video file
    cap = cv2.VideoCapture(video_path)

    if cap.isOpened():

        ret, frame = cap.read()

    else:
        ret = False

    fgbg = cv2.createBackgroundSubtractorMOG2()
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    while ret:

        fgmask = fgbg.apply(frame1)

        (_, contours, _) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 100 or cv2.contourArea(c) > 200:
                continue

            # get bounding box from contour
            (x, y, w, h) = cv2.boundingRect(c)

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