import cv2
import matplotlib as plt
import numpy as np


# read the image
image = cv2.imread(r'C:\Users\Lenovo\Downloads\wombat_tracking\images\s22\8.png')

# Cropping an image
CROP_X = 30
CROP_W = 1000
CROP_Y = 750
CROP_H = 480
THRESHOLD = 4800

cropped_image = image[CROP_X:CROP_X+CROP_W, CROP_Y:CROP_Y+CROP_H]
# cropped_image = image

# Display cropped image
# cv2.imshow("cropped", cropped_image)

# convert the image to grayscale format
img_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

thresh_img = cv2.adaptiveThreshold(img_gray, 
                                255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,
                                17,
                                3)

# cv2.imshow('Adaptive Threshold', thresh_img)          

kernel  =  np.ones((5,5),np.uint8)
segmented_img = cv2.erode(thresh_img, kernel, iterations=2)
segmented_img = cv2.dilate(segmented_img, kernel, iterations=2)
segmented_img =cv2.bitwise_not(segmented_img)
# cv2.imshow('Segmented image', segmented_img) 

cnts,_ = cv2.findContours(segmented_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

for c in cnts:
    if cv2.contourArea(c)>THRESHOLD and cv2.contourArea(c)<40000: 
        # print("contour_area: ", cv2.contourArea(c)) 
        # cv2.contourArea returns the no. of pixels inside the corresponding contour
        min_area_rect = cv2.minAreaRect(c) # for finding minimum area rotated rectangle [center coord, size=(width, height), orintation in clockwise]
        print("min_area: ", min_area_rect)
        orientation = min_area_rect[2]
        orientation = max(orientation, (90-orientation))
        print("orientation: ", orientation)
        # if min_area_rect[2]<10:
        #     if ((min_area_rect[1][0]/min_area_rect[1][1])>1.5):
        #         self.phone_present = False
        #         continue
        # if min_area_rect[2]>80:
        #     if ((min_area_rect[1][1]/min_area_rect[1][0])>1.5):
        #         self.phone_present = False
        #         continue
        
        rect = cv2.boundingRect(c) #It is a straight rectangle, so area of the bounding rectangle won't be minimum..returns (top-left x,y; width, height)
        print("rect: ", rect)
        bbox = cv2.boxPoints(min_area_rect) # returns corner points
        # print("bbox: ", bbox)
        bbox = np.int0(bbox)
        # print("bbox: ", bbox)
        cv2.drawContours(cropped_image, [bbox], 0, (255, 0, 0), 2)

        # Get the center of the bounding box to display the orientation text
        center = (int(min_area_rect[0][0])-20, int(min_area_rect[0][1])-int(rect[3]/2)-20)

        # Display orientation as text near the bounding box
        cv2.putText(cropped_image, f"Angle: {orientation:.2f}", center,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Phone Detected', cropped_image)
        temp_image_path = r'8_detected.png'
        cv2.imwrite(temp_image_path, cropped_image)
# De-allocate any associated memory usage          
if cv2.waitKey(0) & 0xff == 27: 
    cv2.destroyAllWindows() 