import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
offset=20
imgsize=300
cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)
folder="Data/R"
counter=0

while True:
    success, img=cap.read()
    hands, img=detector.findHands(img)
    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']
        imgwhite=np.ones((imgsize,imgsize,3),np.uint8)*255
        
        imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]
        imgcropshape=imgCrop.shape

        aspectratio=h/w
        if aspectratio>1:
            k=imgsize/h
            wcalculated=math.ceil(k*w)
            imgresize=cv2.resize(imgCrop,(wcalculated,imgsize))
            imgresizeshape=imgresize.shape
            wgap=math.ceil((imgsize-wcalculated)/2)
            imgwhite[:,wgap:wcalculated+wgap]=imgresize
        else:
            k=imgsize/w
            hcalculated=math.ceil(k*h)
            imgresize=cv2.resize(imgCrop,(imgsize,hcalculated))
            imgresizeshape=imgresize.shape
            hgap=math.ceil((imgsize-hcalculated)/2)
            imgwhite[hgap:hcalculated+hgap,:]=imgresize  
        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("Image White",imgwhite)

    cv2. imshow("Image",img)  
    key=cv2.waitKey(1)
    if key==ord("s"):
        counter+=1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg",imgwhite)
        print(counter)