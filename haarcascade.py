import cv2
import numpy as np


def main():

    video_path = 'D:\krk.mp4'
    cv2.ocl.setUseOpenCL(False)

    #read video file
    cap = cv2.VideoCapture(video_path)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    human_cascade = cv2.CascadeClassifier('D:\haarcascade_fullbody.xml')

    while ret:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        human = human_cascade.detectMultiScale(gray, 1.1, 4)

        for (x,y,w,h) in human:
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("inter", frame1)

        if cv2.waitKey(40) == 27:
            break

        frame1 = frame2
        ret, frame2 = cap.read()

    cv2.destroyAllWindows()
    cap.release()

main()