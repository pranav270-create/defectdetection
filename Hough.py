import cv2
import numpy as np

def Hough(image, blur, p1, p2, minR, maxR):
    gray = image.astype('uint8')
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, blur)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1,gray.shape[0]/8, param1=p1,param2=p2,minRadius=minR,maxRadius=maxR)
    circles = np.uint16(np.around(circles))
    return circles[0,:]
