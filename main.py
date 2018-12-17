import people
import detection

long_people = []
PEOPLE_LIST = detection.detect()
for p in PEOPLE_LIST:
    if len(p.history) > 15 :
        long_people.append(p)

people.draw_people(long_people)