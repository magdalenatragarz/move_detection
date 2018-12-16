#import detect
import people
import scipy.stats
import circular_buffer

VIDEO_PATH = 'D:\krk.mp4'

#detect.detect(VIDEO_PATH)

x = circular_buffer.RingBuffer(3)
x.append(1)
x.append(2)
x.append(3)
x.append(4)
x.append(5)
print(x.get())
