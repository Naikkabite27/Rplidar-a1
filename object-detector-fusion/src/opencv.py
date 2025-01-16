#!/usr/bin/env python3

import cv2
import rospy
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge

def object_detection():
    rospy.init_node('opencv_object_detection', anonymous=True)
    image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
    marker_pub = rospy.Publisher('/object_markers', MarkerArray, queue_size=10)
    bridge = CvBridge()

    cap = cv2.VideoCapture(0)  # Open the webcam

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            continue

        # Perform object detection with OpenCV (e.g., using Haar cascades, etc.)
        # For example, detecting faces (this is just an example, you would implement your own detection method)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Create MarkerArray to visualize detected objects in RViz
        markers = MarkerArray()
        for (x, y, w, h) in faces:
            marker = Marker()
            marker.header.frame_id = "camera_link"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "faces"
            marker.id = len(markers.markers)
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = x + w / 2
            marker.pose.position.y = y + h / 2
            marker.pose.position.z = 0
            marker.scale.x = w
            marker.scale.y = h
            marker.scale.z = 0.1  # Height of the box (arbitrary)
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            markers.markers.append(marker)

        # Publish Image and Markers
        image_msg = bridge.cv2_to_imgmsg(frame, "bgr8")
        image_pub.publish(image_msg)
        marker_pub.publish(markers)

        # Display the image
        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        object_detection()
    except rospy.ROSInterruptException:
        pass
