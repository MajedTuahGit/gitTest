import cv2
import numpy as np

def getContours(img):
    contour, _ = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contour:
        area = cv2.contourArea(cnt)
        print(area)
        cv2.drawContours(imgContour,cnt,-1,(255,0,0),3)


img = cv2.imread('images/shape.jpg')
imgGrayScale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGrayScale,(7,7), 1)
imgCanny = cv2.Canny(imgBlur, 30,30)
imgContour = img.copy()
getContours(imgCanny)

cv2.imshow("Original Picture", img)
##cv2.imshow("Gray Scale", imgGrayScale)
cv2.imshow("Blur Image" , imgBlur)
cv2.imshow("Canny Edge Detector", imgCanny)
cv2.imshow("Contour Image",imgContour)

cv2.waitKey(0)