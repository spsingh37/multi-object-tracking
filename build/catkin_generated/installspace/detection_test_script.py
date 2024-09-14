import cv2
import numpy as np 
from phone_detection import *

image = cv2.imread('/home/biorobotics-ms/catkin_ws/src/wombat_tracking/images/s22/2.png')

frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# crop_mask = np.zeros_like(image)
# CROP_X = 210
# CROP_W = 250
# CROP_Y = 0
# CROP_H = 450
# crop_mask[CROP_Y:CROP_Y+CROP_H, CROP_X:CROP_X + CROP_W]  = np.ones((CROP_H, CROP_W))
# frame = cv2.bitwise_and(image, image, mask  = crop_mask)
# frame = image.copy()
# frame_col = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
# thresh = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 7, 3)
# kernel = np.ones((5,5),np.uint8)
# thresh = cv2.erode(thresh, kernel, iterations=2)
# thresh = cv2.dilate(thresh, kernel, iterations=2)
# thresh =cv2.bitwise_not(thresh)
# # thresh[:,CROP_X:CROP_X+2] = 0
# # thresh[:,CROP_X+CROP_W-2:CROP_X+CROP_W] = 0

# cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# cnt = sorted(cnts, key = cv2.contourArea)
# cnts = cnt[1:]
# c = max(cnts, key=cv2.contourArea)
# print(cv2.contourArea(c))
# min_area_rect = cv2.minAreaRect(c)
# bbox = cv2.boxPoints(min_area_rect)
# bbox = np.int0(bbox)
# cv2.drawContours(frame_col,cnts,-1,(255,0,255),2)
# cv2.drawContours(frame_col,[bbox],0,(255,0,0),2

# print(min_area_rect)
# cv2.imshow('output', frame_col)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

detector = MultiDeviceDetection()
min_area_rects, rects = detector.detect_phone(frame)


for min_area_rect in min_area_rects:
    bbox = cv2.boxPoints(min_area_rect)
    bbox = np.int0(bbox)
    cv2.drawContours(image,[bbox],0,(255,0,0),2)


cv2.imshow('output', image)
cv2.waitKey(0)
cv2.destroyAllWindows()