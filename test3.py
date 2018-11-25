import cv2
import numpy as np

def main():

    video_path = 'D:\krk.mp4'
    cv2.ocl.setUseOpenCL(False)

    # read video file
    cap = cv2.VideoCapture(video_path)

    # fgbg = cv2.createBackgroundSubtractorMOG2()
    if cap.isOpened():

        ret, frame = cap.read()

    else:
        ret = False

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while ret:
        ret, frame = cap.read()
        if ret:
            d = cv2.absdiff(frame1, frame2)

            # grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

            # blur = cv2.GaussianBlur(grey, (5, 5), 0)
            # ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            # dilated = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=3)
            # img, c, h = cv2.findContours(dilated, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)

            gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
            ret, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, np.ones((2, 2), np.uint8), iterations=2)
            # thresh = cv2.dilate(thresh, None, iterations=2)
            _, contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for c in contours:
                if cv2.contourArea(c) < 100:
                    continue

                # get bounding box from contour
                (x, y, w, h) = cv2.boundingRect(c)

                # draw bounding box
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.drawContours(frame1, contours, -1, (0, 0, 255), 3)

            cv2.imshow("inter", frame1)

            if cv2.waitKey(40) == 27:
                break

            frame1 = frame2
            ret, frame2 = cap.read()

    cv2.destroyAllWindows()
    cap.release()

main()