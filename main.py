import people
import det_track

long_people = []
PEOPLE_LIST = det_track.detect()
for p in PEOPLE_LIST:
    if len(p.history) > 15 :
        long_people.append(p)

people.draw_people(long_people)