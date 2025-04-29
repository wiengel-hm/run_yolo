import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
import cv2
from ament_index_python.packages import get_package_share_directory, get_package_prefix
from rclpy.qos import qos_profile_sensor_data  # Quality of Service settings for real-time data
from ultralytics import YOLO
from ros2_numpy import image_to_np, np_to_compressedimage

class RunYOLO(Node):
    def __init__(self, pkg_dir):
        super().__init__('run_yolo')

        # Use yolov8n for speed; change path/model if needed
        model_path = pkg_dir + '/models/yolov8n.pt'

        # Define Quality of Service (QoS) for communication
        qos_profile = qos_profile_sensor_data  # Suitable for sensor data
        qos_profile.depth = 1  # Keep only the latest message

        self.model = YOLO(model_path)  

        # Subscriber to receive camera images
        self.im_subscriber = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, qos_profile)
       
        # Publisher to send processed result images for visualization
        self.im_publisher = self.create_publisher(CompressedImage, '/result', qos_profile)


        self.get_logger().info('YOLO started...')

    def image_callback(self, msg):

        # Convert ROS image to numpy format
        # timestamp_unix is the image timestamp in seconds (Unix time)
        image, timestamp_unix = image_to_np(msg)

        # Run YOLO inference
        prediction = self.model(image, verbose = False)[0]

        # Draw results on the image
        plot = prediction.plot()

        # Convert back to ROS2 Image and publish
        im_msg = np_to_compressedimage(cv2.cvtColor(plot, cv2.COLOR_BGR2RGB))
        
        # Publish
        self.im_publisher.publish(im_msg)

def main(args=None):
    pkg_dir = get_package_prefix('run_yolo').replace('install', 'src') # /mxck2_ws/install/run_yolo → /mxck2_ws/src/run_yolo
    rclpy.init(args=args)
    node = RunYOLO(pkg_dir)
    rclpy.spin(node)
    node
