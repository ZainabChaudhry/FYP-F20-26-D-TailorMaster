# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 08:38:47 2021

@author: Zaini
"""
import cv2 as cv
def videoProcessing(path):  #extract frames
    vidcap = cv.VideoCapture(path)
    count = 0
    success = True
    fps = int(vidcap.get(cv.CAP_PROP_FPS))
    i=1
    im=0
    imagepaths=[]
    tempstr=""
    while success:
        success,image = vidcap.read()
        if(fps!=0):
            if count%(i*fps) == 0 :
                if (len(image)<len(image[0])):
                    image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
                tempstr="img"+str(im)+".jpg"
                cv.imwrite("img"+str(im)+".jpg",image)
                imagepaths.append(tempstr)
                im+=1
                i+=3      
        count+=1
    return imagepaths
