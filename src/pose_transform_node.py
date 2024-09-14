#! /usr/bin/env python3
import numpy as np
import rospy 
from geometry_msgs.msg import Pose, PoseStamped
import argparse
from scipy.spatial.transform import Rotation as R 
from vision_msgs.msg import BoundingBox3D

class PoseTransformer:
    def __init__(self):
        self._publisher = rospy.Publisher("/front/phone_pose_camera", PoseStamped, queue_size=20)
        self._bbox_sub = rospy.Subscriber("/front/bbox", BoundingBox3D, self._bbox_callback)
        self.item_class = rospy.get_param('/item_class')

        
    def _bbox_callback(self, msg):
        quat  = msg.center.orientation
        quat = np.array([quat.x, quat.y, quat.z, quat.w])
        r = R.from_quat(quat)
        angle = r.as_euler('zyx', degrees = True)[0]

        width, height= msg.size.x, msg.size.y

        if self.item_class == 'watch':
            if width<height:
                angle =  -(180+angle)

        if self.item_class == 'iPhone' or self.item_class is None:
            if(width>height):
                angle = -(180+angle)
        
        if(self.item_class == 'iPad'):
            if(width<height):
                angle = -(180+angle)
        


        r_euler = R.from_euler('z', angle, degrees=True)
        quat = r_euler.as_quat()
        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.pose.position = msg.center.position
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]
        self._publisher.publish(pose)

if __name__=='__main__':

    rospy.init_node('pose_transform_node')
    PoseTransformer()
    rospy.spin()
