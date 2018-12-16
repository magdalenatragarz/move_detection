import cv2
import imutils
import pygame
import colors

class History(object):
    def __init__(self, bbox):
        self.history = []
        self.history.append(bbox)

    def update(self, bbox):
        self.history.append(bbox)


cap = cv2.VideoCapture("D:\krk3.mov")
TRACKERS = []
HISTORY = []


(width, height) = (700, 500)
screen = pygame.display.set_mode((width, height))
screen.fill(colors.white)

ret,frame = cap.read()
frame = imutils.resize(frame, width=900)
for i in range(5):
    box = cv2.selectROI("Frame", frame, fromCenter=False)
    (x, y, w, h) = [int(v) for v in box]
    tracker = cv2.TrackerKCF_create()
    ok = tracker.init(frame, box)
    if ok:
        TRACKERS.append(tracker)
        HISTORY.append(History((x,y,0)))


frame_count = 0
while ret:
    # Read a new frame
    frame_count += 1

    frame = imutils.resize(frame, width=900)

    #(success, boxes) = trackers.update(frame)

    for t in TRACKERS:
        (ok,box) = t.update(frame)
        if (ok):
            index = TRACKERS.index(t)
            (x, y, w, h) = [int(v) for v in box]
            HISTORY[index].update((x, y,frame_count ))
            cv2.rectangle(frame, (x, y), (x + 15, y + 25), (0, 255, 0), 2)
            cv2.putText(frame, str(index+1), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            pygame.draw.circle(screen, colors.black, [x, y], 2, 2)
            pygame.display.update()
            pygame.display.flip()

    # Start timer
    timer = cv2.getTickCount()

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display result
    cv2.imshow("Tracking", frame)

    ret, frame = cap.read()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Exit if ESC pressed
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

for h in HISTORY:
    print(h.history)