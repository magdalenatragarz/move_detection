import cv2


def main():

    video_path = 'D:\krk.mp4'
    cv2.ocl.setUseOpenCL(False)

    tracker = cv2.TrackerBoosting_create()


    #read video file
    cap = cv2.VideoCapture(video_path)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    bbox = (287, 23, 86, 320)
    ok = tracker.init(frame1, bbox)


    while ret:
        i = 1
        _, frame = cap.read()

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
            (x, y, w, h) = cv2.boundingRect(c)

            if y < 450:
                continue

            # draw bounding box
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)


        cv2.imshow("inter", frame1)

        if cv2.waitKey(40) == 27:
            break

        frame1 = frame2
        ret, frame2 = cap.read()
        # j=j+1
    cv2.destroyAllWindows()
    cap.release()


main()

#installed with whl files!!!!!