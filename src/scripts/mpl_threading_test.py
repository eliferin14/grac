import matplotlib.pyplot as plt
from queue import Queue, Full, Empty
import threading
import time

data_queue = Queue(maxsize=1)
finished = False

def main_loop():
    for n in range(1000):
        print(f"Iter: {n}")
        data_queue.put(n)
        time.sleep(0.1)
        
    global finished 
    finished = True
        
main_loop_thread = threading.Thread(target=main_loop, daemon=True)
main_loop_thread.start()


    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.show(block=False)
while True:
    n = data_queue.get()
    print(f"Received {n}")
    ax.scatter(n, 0, 0, c='g', marker='o')
    plt.draw(), plt.pause(0.1)
    if finished and data_queue.empty(): break


        
        
        