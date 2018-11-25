import cv2
import numpy as np

def main():

    videoPath = 'D:\krk.mp4'
    cv2.ocl.setUseOpenCL(False)

    #read video file
    cap = cv2.VideoCapture('D:\krk.mp4')



    #filename = 'D:\krk.mov'
    #codec=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    #framerate=30
    #resolution = (500,500)

    #VideoFileOutput = cv2.VideoWriter(filename,codec,framerate,resolution)

    if cap.isOpened():

        ret, frame = cap.read()

    else:
        ret = False

    fgbg = cv2.createBackgroundSubtractorMOG2()
    ret, frame = cap.read()
    while ret:
        # VideoFileOutput.write(frame)

            # d = cv2.absdiff(frame1, frame2)
            #
            # # grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
            #
            # # blur = cv2.GaussianBlur(grey, (5, 5), 0)
            # # ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            # # dilated = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=3)
            # # img, c, h = cv2.findContours(dilated, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
            #
            # imgray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
            # blur = cv2.GaussianBlur(imgray, (5, 5), 0)
            # #ret, thresh = cv2.threshold(imgray, 127, 255, 0)
            # ret, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            # #dilated = cv2.dilate(thresh, np.ones((2, 2), np.uint8), iterations=2)
            # thresh = cv2.dilate(thresh, None, iterations=2)
            # _, contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # #im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


            # cv2.drawContours(frame1, contours, -1, (0, 0, 255), 3)

        fgmask = fgbg.apply(frame)

        (im2, contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 100:
                continue

            # get bounding box from contour
            (x, y, w, h) = cv2.boundingRect(c)

            # draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # cv2.imshow("win1",frame2)
        cv2.imshow("inter", frame)

        if cv2.waitKey(40) == 27:
            break
        # frame1 = frame2
        ret, frame = cap.read()

    cv2.destroyAllWindows()
    cap.release()


main()