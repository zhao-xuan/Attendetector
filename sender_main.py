from faceprocess import *
import requests
import cv2

url = 'http://localhost:5000/update_status'

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

def run():
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        result = facial_processing(frame)
        with requests.Session() as s:
            p = s.post(url, data = {"result" : result})

        '''font = cv2.FONT_HERSHEY_SIMPLEX 
        org = (50, 50)
        fontScale = 1
        color = (0, 255, 0)
        thickness = 2
        preview = cv2.putText(frame, result, org, font,  
                fontScale, color, thickness, cv2.LINE_AA)
        
        cv2.imshow('frame', preview)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break'''
        sleep(1)

run()

# When everything done, release the capture
'''cap.release()
cv2.destroyAllWindows()'''