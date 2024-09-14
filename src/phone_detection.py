

'''
Residual Code: Not to be used. Just kept here for reference
'''
'''
This class is used to detect phones.
Has The Below functions
    Phone Detection
        Input: image frame 
        Output: The min area rectangle object 
    Is Phone Present(img)
        Returns Bool
'''

import cv2
import numpy as np 


CROP_X = 210
CROP_W = 230
CROP_Y = 0
CROP_H = 475
THRESHOLD = 4800


class PhoneDetection:
    def __init__(self):
        self.phone_present = False
        self.thresh = None
        self.area = 0
    
        pass


    def detect_phone(self, gray):
        # gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.uint8)
        self.thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 7, 3)
        kernel = np.ones((5,5),np.uint8)
        self.thresh = cv2.erode(self.thresh, kernel, iterations=2)
        self.thresh = cv2.dilate(self.thresh, kernel, iterations=2)
        self.thresh =cv2.bitwise_not(self.thresh)
        kernelSize = (1,1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
        opening = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, kernel)

        self.thresh[:,CROP_X:CROP_X+2] = 0
        self.thresh[:,CROP_X+CROP_W-2:CROP_X+CROP_W] = 0
        # self.thresh = cv2.bilateralFilter(self.thresh,21, 75, 75)
        cnts,_ = cv2.findContours(self.thresh.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cnt = sorted(cnts, key = cv2.contourArea)
        cnts = cnt[1:]
        c = max(cnts, key=cv2.contourArea)
        rects = []
        min_area_rects = []
        self.contours = cnts
        
        if True:
            min_area_rect = cv2.minAreaRect(c)
            rect  = cv2.boundingRect(c)
            rects.append(rect)
            # bbox = cv2.boxPoints(rect)
            # bbox = np.int0(bbox)
            # cv2.drawContours(image,[bbox],0,(0,0,255),2)
            # Use for debugging the obj detector 
            self.area = cv2.contourArea(c)
            if cv2.contourArea(c)>5000:

                self.phone_present = True

            if min_area_rect[2]<10:
                if ((min_area_rect[1][0]/min_area_rect[1][1])>1.75):
                    self.phone_present = False
            if min_area_rect[2]>80:
                if ((min_area_rect[1][1]/min_area_rect[1][0])>1.75):
                    self.phone_present = False


            return min_area_rect, rects

    def is_phone_present(self):
        return self.phone_present

'''
This class is used to detect phones.
Has The Below functions
    Phone Detection
        Input: image frame 
        Output: The min area rectangle object 
    Is Phone Present(img)
        Returns Bool
'''


class MultiDeviceDetection_1:
    def __init__(self):
        self.phone_present = False
        self.thresh = None
        self.area = 0
    


    def detect_phone(self, gray):
        # gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.uint8)
        self.thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 7, 3)
        # self.thresh = cv2.dilate(self.thresh, kernel, iterations=1)

        kernel = np.ones((5,5),np.uint8)
        self.thresh = cv2.erode(self.thresh, kernel, iterations=2)
        self.thresh = cv2.dilate(self.thresh, kernel, iterations=2)
        self.thresh =cv2.bitwise_not(self.thresh)
        kernelSize = (1,1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)

        self.thresh[:,CROP_X:CROP_X+2] = 0
        self.thresh[:,CROP_X+CROP_W-2:CROP_X+CROP_W] = 0
        cnts,_ = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        self.contours = cnts
        rects = []
        min_area_rects = []        
        for c in cnts:
            if cv2.contourArea(c)>THRESHOLD: 
                min_area_rect = cv2.minAreaRect(c)
                if min_area_rect[2]<10:
                    if ((min_area_rect[1][0]/min_area_rect[1][1])>1.5):
                        self.phone_present = False
                        continue
                if min_area_rect[2]>80:
                    if ((min_area_rect[1][1]/min_area_rect[1][0])>1.5):
                        self.phone_present = False
                        continue

                rect = cv2.boundingRect(c)
                bbox = cv2.boxPoints(min_area_rect)
                bbox = np.int0(bbox)
                # center = (int(min_area_rect[0][0]),int(min_area_rect[0][1]))
                min_area_rects.append(min_area_rect)
                rects.append(rect)
                self.phone_present = True
            # bbox = cv2.boxPoints(rect)
            # bbox = np.int0(bbox)
            # cv2.drawContours(self.thresh,[bbox],0,(0,0,255),2)
            # Use for debugging the obj detector 




        return min_area_rects, rects

    def is_phone_present(self):
        return self.phone_present



class MultiDeviceDetection:
    def __init__(self):
        self.phone_present = False
        
    def __threshold(self, gray):
        thresh_img = cv2.adaptiveThreshold(gray, 
                                            255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY,
                                            17,
                                            3)
        return thresh_img

    def __segment_devices(self, thresh_img):
        kernel  =  np.ones((5,5),np.uint8)
        segmented_img = cv2.erode(thresh_img, kernel, iterations=2)
        segmented_img = cv2.dilate(segmented_img, kernel, iterations=2)
        segmented_img =cv2.bitwise_not(segmented_img)
        # segmented_img[:,210:220] = 0
        # segmented_img[:, -5:]= 0
        return segmented_img

    def __find_contours(self, segmented_img):
        cnts,_ = cv2.findContours(segmented_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        return cnts
    
    def detect_phone(self, gray):
        self.thresh = self.__threshold(gray)
        self.segmented_image = self.__segment_devices(self.thresh)
        self.contours = self.__find_contours(self.segmented_image)
        min_area_rects = []
        rects = []
        print("total_cont: ",len(self.contours))
        for c in self.contours:
            print(cv2.contourArea(c))
            if cv2.contourArea(c)>THRESHOLD and cv2.contourArea(c)<6000: 
                min_area_rect = cv2.minAreaRect(c)
                # if min_area_rect[2]<10:
                #     if ((min_area_rect[1][0]/min_area_rect[1][1])>1.5):
                #         self.phone_present = False
                #         continue
                # if min_area_rect[2]>80:
                #     if ((min_area_rect[1][1]/min_area_rect[1][0])>1.5):
                #         self.phone_present = False
                #         continue

                rect = cv2.boundingRect(c)
                bbox = cv2.boxPoints(min_area_rect)
                bbox = np.int0(bbox)
                # center = (int(min_area_rect[0][0]),int(min_area_rect[0][1]))
                min_area_rects.append(min_area_rect)
                rects.append(rect)
                self.phone_present = True

        return min_area_rects, rects            
         

    
