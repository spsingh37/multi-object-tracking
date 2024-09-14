#! /usr/bin/env python3

from numpy.lib.shape_base import expand_dims
from phone_detection import MultiDeviceDetection, PhoneDetection
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

# CONSTANTS
SKIP_FRAMES = 1
DISPLAY_IMAGE  = True
# Can also take this from args


#Global Variables 
frame_counter = 0

class PhoneTracking:

    def __init__(self):
        # Add the list of publishers and subscribers here 
        # cam info and other stuff
        # TODO : Check the name of the pub/sub topics
        self.camera_info = CameraInfo()
        self.depth_image = Image()
        self.bridge = CvBridge()
        self.tracked_min_area_rect = [(0,0),(0,0),0]
        # self.camera_depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.camera_depth_callback)
        self.camera_img_sub = rospy.Subscriber("/rear/color/image_raw",Image, self.image_tracker )
        self.camera_info_sub = rospy.Subscriber("/rear/color/camera_info", CameraInfo, self.camera_info_callback)
        self.is_phone_present_pub = rospy.Publisher("/is_phone_present", Bool, queue_size = 30)
        self.tracked_image_pub = rospy.Publisher("/rear/tracked_image", Image, queue_size = 30)
        self.phone_pose_camera_pub = rospy.Publisher("/rear/phone_pose_camera", PoseStamped, queue_size = 70)
        self.projection_mat = None
        self.inv_projection_mat = None
        if os.path.exists("/home/biorobotics/wombat_arm_ws/src/wombat_tracking/src/projection_mat_rear.npy"):
            self.projection_mat = np.load("/home/biorobotics/wombat_arm_ws/src/wombat_tracking/src/projection_mat_rear.npy")
            print("Projection Matrix = ",self.projection_mat)
        if os.path.exists("/home/biorobotics/wombat_arm_ws/src/wombat_tracking/src/inv_projection_mat_rear.npy"):
            self.inv_projection_mat = np.load("/home/biorobotics/wombat_arm_ws/src/wombat_tracking/src/inv_projection_mat_rear.npy")
            print("Inv Projection Matrix = ",self.inv_projection_mat)
        
    def crop_image(self,image):
        crop_mask = np.zeros_like(image)
        # print(f"crop mask {crop_mask.shape}")

        CROP_X = 162
        CROP_W = 251
        CROP_Y = 75
        CROP_H = 480 - 75
        crop_mask[CROP_Y:CROP_Y+ CROP_H, CROP_X:CROP_X + CROP_W]  = np.ones((CROP_H, CROP_W))
        frame = cv2.bitwise_and(image, image, mask  = crop_mask)
        # frame[:,:CROP_X+20] = 255
        # frame[:,CROP_X + CROP_W -10:] = 255
        # frame[:CROP_Y+20,:] = 255
        # frame[CROP_Y + CROP_H:, :] = 255
        return frame


    def transform_rects_to_detection(self, rects):
        
        detections = []
        for rect in rects:
            x, y, w, h = rect
            confidence = 0.5
            detections.append([x, y, x+w, y+h, confidence])
        return detections


    def transform_detection_to_centroid(self, det):
        x1 = det[0]
        y1 = det[1]
        x2 = det[2]
        y2 = det[3]
        center_x = int((x1+x2)/2)
        center_y = int((y1+y2)/2)
        return center_x, center_y


    def convert_pixel_to_point(self, x, y, depth, cameraInfo):
        _intrinsics = rs.intrinsics()
        _intrinsics.width = cameraInfo.width
        _intrinsics.height = cameraInfo.height
        _intrinsics.ppx = cameraInfo.K[2]
        _intrinsics.ppy = cameraInfo.K[5]
        _intrinsics.fx = cameraInfo.K[0]
        _intrinsics.fy = cameraInfo.K[4]
        #_intrinsics.model = cameraInfo.distortion_model
        _intrinsics.model  = rs.distortion.none     
        _intrinsics.coeffs = [i for i in cameraInfo.D]
        result = rs.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)
        # result[0]: right, result[1]: down, result[2]: forward
        x = result[2]
        y = -result[0]
        z = -result[1]
        return x,y,z

    def predict_robot_point(self,image_point):
        """
        Predicts Robot/World Point by using Projection Matrix
        """
      
        image_point = image_point.tolist()      
        image_point.append(1)
        robot_point_pred = np.transpose(np.dot(self.inv_projection_mat,np.transpose(image_point)))
        robot_point_pred = robot_point_pred/robot_point_pred[-1]
        robot_point_pred = robot_point_pred[:-1]
        return robot_point_pred

    def transform_to_pose(self, p_x, p_y, p_z, orientation):
        
        phone_pose = Pose()
        phone_pose.position.x = p_x
        phone_pose.position.y = p_y
        phone_pose.position.z = p_z


        r = R.from_euler('z', orientation, degrees=True)
        quat = r.as_quat()
        phone_pose.orientation.x = quat[0]
        phone_pose.orientation.y = quat[1]
        phone_pose.orientation.z = quat[2]
        phone_pose.orientation.w = quat[3]
        
        return phone_pose
        
    
    def camera_info_callback(self, msg):
        cam_info = msg
        self.camera_info = cam_info
    
    def camera_depth_callback(self,msg):
        self.depth_image = msg

    def image_tracker(self, msg):
        global frame_counter
        # print("Running img track..")
        try:
            image = self.bridge.imgmsg_to_cv2(msg)
        except CvBridgeError as e:
            print(e)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame = self.crop_image(image)
        frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        

        tracker = Sort()
        memory = {}

        if (frame_counter%SKIP_FRAMES)==0:
            
            detector = PhoneDetection()
            min_area_rect, rects = detector.detect_phone(frame)
            self.is_phone_present_pub.publish(detector.phone_present)
            if detector.phone_present==True:
                #print(min_area_rect[2], min_area_rect[1], detector.phone_present)
                detections = self.transform_rects_to_detection(rects)
                
                detections = np.asarray(detections)
                # print(f"detections: {detections}")
                tracks = tracker.update(detections)
                boxes = []
                indexIDs = []
                previous = memory.copy()
                memory = {}


                for track in tracks:
                    boxes.append([track[0], track[1], track[2], track[3]])
                    indexIDs.append(int(track[4]))
                    memory[indexIDs[-1]] = boxes[-1]
                
                if len(boxes)>0:
                    for i, box in enumerate(boxes):
                        
                        center_x, center_y = self.transform_detection_to_centroid(box)
                        center = (center_x, center_y)
                        self.tracked_min_area_rect = [center, min_area_rect[1], min_area_rect[2]]
                        orientation = self.tracked_min_area_rect[2]
                        width, height = self.tracked_min_area_rect[1]
                        if width >height:
                            orientation = orientation - 90

                        orientation =90 - orientation
                        # orientation = max(orientation, (90-orientation))
                        # print(f"orientation: {orientation}")
                        # try:
                        # depth_image = self.bridge.imgmsg_to_cv2(self.depth_image)
                        # except CvBridgeError as e:
                        #     print(e)

                        # depth_value = depth_image[int(center[1])][int(center[0])]
                        # print("Depth Values = ",depth_value)
                        # TODO Currently Hardcoded depth value :-> depth tuning needed
                        depth_value = 0.8 
                        # c_x, c_y, c_z = self.convert_pixel_to_point(depth_value, center[0] - 320, center[1] - 240,self.camera_info)
                        # c_x, c_y, c_z = self.convert_pixel_to_point(depth_value, 0 - 320, 0 - 240,self.camera_info)
                        image_coord = np.array([center[0],center[1],depth_value])
                        # print(f' Center = {image_coord}')
                        robot_coords = self.predict_robot_point(image_coord)
                        phone_pose = PoseStamped()
                        phone_pose.header.frame_id = 'base_link'
                        # print(robot_coords)
                        phone_pose.pose = self.transform_to_pose(robot_coords[0], robot_coords[1], robot_coords[2], orientation) 
                        # self.transform_to_pose(c_y + 50, c_x - 250, c_z + 1750, orientation)
                        self.phone_pose_camera_pub.publish(phone_pose)
                        bbox = cv2.boxPoints(self.tracked_min_area_rect)
                        bbox = np.int0(bbox)
                        cv2.drawContours(frame_color, [bbox], 0, (255,0,0), 2)

        if DISPLAY_IMAGE:
            try:
                self.tracked_image_pub.publish(self.bridge.cv2_to_imgmsg(frame_color))
            except CvBridgeError as e:
                print(e)



                
        frame_counter+=1

rospy.init_node('phone_tracker_rear')
PhoneTracking()
rospy.spin()



    
    






