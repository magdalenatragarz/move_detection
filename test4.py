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

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while ret:
        _, frame = cap.read()
        # VideoFileOutput.write(frame)

        d = cv2.absdiff(frame1, frame2)

        # grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

        # blur = cv2.GaussianBlur(grey, (5, 5), 0)
        # ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        # dilated = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=3)
        # img, c, h = cv2.findContours(dilated, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)

        imgray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imgray, (5, 5), 0)
        #ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        ret, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        #dilated = cv2.dilate(thresh, np.ones((2, 2), np.uint8), iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        _, contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # if len(contours) != 0:
        #     # If the contour is big enough
        #
        #     # Set largest contour to first contour
        #     largest = 0
        #
        #     # For each contour
        #     for i in range(len(contours)):
        #         # If this contour is larger than the largest
        #         if i != 0 & int(cv2.contourArea(contours[i])) > int(cv2.contourArea(contours[largest])):
        #             # This contour is the largest
        #             largest = i
        #
        #     if cv2.contourArea(contours[largest]) > 1000:
        #         # Create a bounding box for our contour
        #         (x, y, w, h) = cv2.boundingRect(contours[0])
        #         # Convert from float to int, and scale up our boudning box
        #         (x, y, w, h) = (int(x), int(y), int(w), int(h))
        #         # Initialize tracker
        #         bbox = (x, y, w, h)
        #         ok = tracker.init(frame1, bbox)
        #         # Switch from finding motion to tracking
        #         status = 'tracking'
        #
        #     # If we are tracking
        # if status == 'tracking':
        #     # Update our tracker
        #     ok, bbox = tracker.update(frame1)
        #     # Create a visible rectangle for our viewing pleasure
        #     if ok:
        #         p1 = (int(bbox[0]), int(bbox[1]))
        #         p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        #         cv2.rectangle(frame1, p1, p2, (0, 0, 255), 10)


        #cv2.drawContours(frame1, contours, -1, (0, 0, 255), 3)

        for c in contours:
            #if cv2.contourArea(c) < 10:
             #   continue

            # get bounding box from countour
            (x, y, w, h) = cv2.boundingRect(c)

            # draw bounding box
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)


        #cv2.imshow("win1",frame2)
        cv2.imshow("inter", frame1)

        if cv2.waitKey(40) == 27:
            break
        frame1 = frame2
        ret, frame2 = cap.read()

    cv2.destroyAllWindows()
    #VideoFileOutput.release()
    cap.release()


main()