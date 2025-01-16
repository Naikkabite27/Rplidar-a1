#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import numpy as np

class FeatureObstacleDetector:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('feature_obstacle_detector', anonymous=True)

        # Subscriptions
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)
        self.lidar_sub = rospy.Subscriber("/scan", LaserScan, self.lidar_callback)

        # Publishers
        self.image_pub = rospy.Publisher("/processed_image", Image, queue_size=10)

        # Utilities
        self.bridge = CvBridge()
        self.lidar_data = None

    def lidar_callback(self, msg):
        # Store LiDAR data for potential integration
        self.lidar_data = msg

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV Image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Process the image (Feature Detection)
            processed_image = self.process_image(cv_image)

            # Publish the processed image
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(processed_image, "bgr8"))
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def process_image(self, cv_image):
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect edges (Canny Edge Detection)
        edges = cv2.Canny(gray, 50, 150)

        # Detect contours (optional)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Draw contours on the image
            cv2.drawContours(cv_image, [contour], -1, (0, 255, 0), 2)

        return cv_image

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    detector = FeatureObstacleDetector()
    detector.run()
