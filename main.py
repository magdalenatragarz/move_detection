import people
import detection
import common

long_people = []
PEOPLE_LIST = detection.detect()
for p in PEOPLE_LIST:
    if len(p.history) > 15 :
        long_people.append(p)
        print(p.history)

