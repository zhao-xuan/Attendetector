'''
Using camera to capture image
This files is used to capture images for training_set
'''

import os
import numpy as np
import cv2
import threading
from time import sleep

cap = cv2.VideoCapture(4)
#img_index: img_index[0] -> busy, img_index[1] -> free
status = 'busy'
img_index_list = {"busy" : 0, "free" : 0}
img_index = img_index_list[status]
training_set_index = 0
path = "~/facialExpRecPytorch/images/training_set"
filename = ""

def run():
    global img_index, filename, status, training_set_index, path
    training_set_index = input("What's the current batch of training set?")
    path = path + str(training_set_index)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Save an image every 5 sec
        filename = status + '_' + str(img_index) + '_img.jpg'
        cv2.imwrite(os.path.join(path, filename), frame)
        img_index += 1
        sleep(5)

class change_status(threading.Thread):
    def run(self):
        global status
        temp_status = status
        while (True):
            status = input("What's your status now?")
            print("Ok! Set your status to" + status)
            if temp_status != status:
                img_index_list[temp_status] = img_index
                img_index = img_index_list[status]
                

change_status = change_status()
change_status.start()
run()


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()