from numpy.lib.shape_base import expand_dims
from phone_detection import PhoneDetection
from sort import Sort
from sort import *
import pyrealsense2 as rs
import cv2
import rospy 
from std_msgs.msg import Empty,Bool
from geometry_msgs.msg import Pose, PoseStamped, Vector3
from cv_bridge import CvBridge,CvBridgeError
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from scipy.spatial.transform import Rotation as R

SKIP_FRAMES = 1

frame_counter = 0

class TrackingTester:

    def __init__(self):
        # Add the list of publishers and subscribers here 
        # cam info and other stuff
        # TODO : Check the name of the pub/sub topics
        self.camera_info = CameraInfo()
        self.depth_image = None
        self.bridge = CvBridge()
        self.tracked_min_area_rect = [(0,0),(0,0),0]
        self.camera_img_sub = rospy.Subscriber("/camera/color/image_raw",Image, self.image_tracker )
        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)
        self.camera_depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.camera_depth_callback)
        self.is_phone_present_pub = rospy.Publisher("/is_phone_present", Bool, queue_size = 30)
        self.process_image_pub = rospy.Publisher("/processed_image", Image, queue_size = 30)
        self.phone_pose_camera_pub = rospy.Publisher("/phone_pose_camera", PoseStamped, queue_size = 30)
        

    def crop_image(self,image):
        crop_mask = np.zeros_like(image)
        CROP_X = 150
        CROP_W = 350
        CROP_Y = 0
        CROP_H = 450
        crop_mask[CROP_Y:CROP_Y+CROP_H, CROP_X:CROP_X + CROP_W]  = np.ones((CROP_H, CROP_W))
        frame = cv2.bitwise_and(image, image, mask  = crop_mask)
        return frame


        
    
    def camera_info_callback(self, msg):
        cam_info = msg
        self.camera_info.header = cam_info.header
    
    def camera_depth_callback(self,msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg)
        except CvBridgeError as e:
            print(e)

    def image_tracker(self, msg):
        global frame_counter
        
        try:
            image = self.bridge.imgmsg_to_cv2(msg)
        except CvBridgeError as e:
            print(e)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame = self.crop_image(image)
        frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        



    # if (frame_counter%SKIP_FRAMES)==0:
        
        detector = PhoneDetection()
        

        min_area_rect, rects = detector.detect_phone(frame)
        rect = rects[0]
        bbox = cv2.boxPoints(min_area_rect)
        bbox = np.int0(bbox)

        print(min_area_rect[0])
        x, y, w, h = rect
		# print((x+x+w)/2,(y+y+h)/2)
        cv2.rectangle(frame_color,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.drawContours(frame_color, [bbox], 0, (255,0,0),2)


    # if DISPLAY_IMAGE:
        try:
            self.process_image_pub.publish(self.bridge.cv2_to_imgmsg(frame_color))
        except CvBridgeError as e:
                print(e)



                
        frame_counter+=1

rospy.init_node('tracking_tester')
TrackingTester()
rospy.spin()



