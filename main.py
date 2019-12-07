"""
Image processing and facial expression recognition in realtime(per second)
Image is cropped using haarcascade_frontalface_default.xml in OpenCV
Facial Expression Recognition using the open source github project involving:
    VGG19 or ResNet18
    Pretrained VVG19 model(currently using this model)
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
from time import sleep
from torch.autograd import Variable

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *
import facecrop as fcrop

cut_size = 44
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def plot_result(img_show, score, predicted):
    plt.rcParams['figure.figsize'] = (13.5,5.5)
    axes=plt.subplot(1, 3, 1)
    plt.imshow(img_show)
    plt.xlabel('Input Image', fontsize=16)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.tight_layout()


    plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)

    plt.subplot(1, 3, 2)
    ind = 0.1+0.6*np.arange(len(class_names))    # the x locations for the groups
    width = 0.4       # the width of the bars: can also be len(x) sequence
    color_list = ['red','orangered','darkorange','limegreen','darkgreen','royalblue','navy']
    for i in range(len(class_names)):
        plt.bar(ind[i], score.data.cpu().numpy()[i], width, color=color_list[i])
    plt.title("Classification results ",fontsize=20)
    plt.xlabel(" Expression Category ",fontsize=16)
    plt.ylabel(" Classification Score ",fontsize=16)
    plt.xticks(ind, class_names, rotation=45, fontsize=14)

    axes=plt.subplot(1, 3, 3)
    emojis_img = io.imread('images/emojis/%s.png' % str(class_names[int(predicted.cpu().numpy())]))
    plt.imshow(emojis_img)
    plt.xlabel('Emoji Expression', fontsize=16)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.tight_layout()
    #show emojis

    #plt.show()

    plt.savefig(os.path.join('images/results/' + "result_" + str(class_names[int(predicted.cpu().numpy())]) + '.png'))
    plt.close()

def facial_processing(raw_img):
        gray = rgb2gray(raw_img)

        face_detector = fcrop.FaceCropper()
        face_region = face_detector.generate(gray)
        if face_region == 0:
            return "No face detected!"

        (nx, ny, nr) = face_region

        gray = resize(gray[ny:ny+nr, nx:nx+nr], (48,48), mode='symmetric').astype(np.uint8)

        img = gray[:, :, np.newaxis]

        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        inputs = transform_test(img)

        net = VGG('VGG19')
        #checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'), map_location=torch.device('cpu'))
        #if torch.cuda.is_available():
        net.cuda()
        checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'), map_location=torch.device('cuda'))
        net.load_state_dict(checkpoint['net'])
        #if torch.cuda.is_available():
        net.cuda()
        net.eval()

        ncrops, c, h, w = np.shape(inputs)

        inputs = inputs.view(-1, c, h, w)
        inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)
        outputs = net(inputs)

        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

        score = F.softmax(outputs_avg)
        _, predicted = torch.max(outputs_avg.data, 0)
        result_string = str(class_names[int(predicted.cpu().numpy())])

        print("The Expression is %s" %result_string)

        return result_string

"""Video capturing per second"""

cap = cv2.VideoCapture(0)


def run():
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Scan frame per second
        cv2.imwrite('/dev/shm/img.bmp', frame)
        img = io.imread('/dev/shm/img.bmp')
        result_string = facial_processing(img)

        font = cv2.FONT_HERSHEY_SIMPLEX 
        org = (50, 50)  
        fontScale = 1 
        color = (255, 0, 0) 
        thickness = 2
        preview = cv2.putText(frame, result_string, org, font,  
                fontScale, color, thickness, cv2.LINE_AA)
        
        cv2.imshow('frame', preview)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
run()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()