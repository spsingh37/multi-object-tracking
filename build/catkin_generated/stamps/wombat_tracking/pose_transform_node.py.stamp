import rospy 
from geometry_msgs.msg import Pose
import argparse
from scipy.spatial.transform import Rotation as R 
from vision_msgs.msg import BoundingBox3D

class PoseTransformer:
    def __init__(self, item_class =None):
        self._publisher = rospy.Publisher("/front/phone_pose_camera", Pose, queue_size=20)
        self._bbox_sub = rospy.Subscriber("/front/bbox", BoundingBox3D, self._bbox_callback)
        self.item_class = item_class

        
    def _bbox_callback(self, msg):
        quat  = msg.center.orientation 
        r = R.from_quat(quat)
        angle = r.as_euler('zyx', degrees = True)[0]

        phone_position = msg.center
        width, height= msg.size.x, msg.size.y

        print(f"width: {width}, height: {height}")

        



if __name__=='__main__':
    parser = argparse.ArgumentParser("Pose Transformer")
    parser.add_argument("item_class", choices = ['iPhone', 'iPad', 'watch'])
    args = parser.parse_args()
    item_class = args.item_class

    rospy.init_node('pose_transform_node')
    PoseTransformer(item_class)
    rospy.spin()
