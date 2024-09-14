import cv2
import numpy as np 

image = cv2.imread('/home/biorobotics/wombat_arm_ws/src/wombat_tracking/images/image.png', 0)


crop_mask = np.zeros_like(image)
CROP_X = 300
CROP_W = 75
CROP_Y = 10
CROP_H = 85
crop_mask[CROP_Y:CROP_Y+CROP_H, CROP_X:CROP_X + CROP_W]  = np.ones((CROP_H, CROP_W))
frame = cv2.bitwise_and(image, image, mask  = crop_mask)

cv2.imshow("output", frame)

cv2.waitKey(0)
cv2.destroyAllWindows()

