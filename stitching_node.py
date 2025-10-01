#!/usr/bin/env python3
# ----------------------------------------------------
# Image Stitcher Node (ROS2 + OpenCV in Python)
# ----------------------------------------------------
# - Subscribes to raw images
# - Subscribes to mission state (string)
# - Subscribes to MAVROS GPS (lat/lon/alt)
# - Takes pictures at waypoints, saves with GPS info
# - Stitches images after enough are collected
# ----------------------------------------------------

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image, NavSatFix
from cv_bridge import CvBridge
from std_msgs.msg import String
from mavros_msgs.msg import WaypointReached

import cv2
import numpy as np
import os
from datetime import datetime

MAX_PICS = 2  # how many pictures before stitching


class ImageStitcherNode(Node):
    def __init__(self):
        super().__init__('image_stitcher_node')

        # --------------------------
        # Internal state
        # --------------------------
        self.transition_in_progress = False
        self.waypoint_current = 0
        self.waypoint_prev = -1
        self.latest_frame = None
        self.state = "lap"
        self.reached = False
        self.pic_counter = 0

        # GPS storage
        self.current_lat = None
        self.current_lon = None
        self.current_alt = None

        # --------------------------
        # Parameters
        # --------------------------
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('output_path', '/home/ubuntu/stitched_image')
        self.declare_parameter('crop', True)
        self.declare_parameter('preprocessing', False)
        self.declare_parameter('stitch_interval_sec', 5.0)

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.output_path = self.get_parameter('output_path').get_parameter_value().string_value
        self.crop = self.get_parameter('crop').get_parameter_value().bool_value
        self.preprocessing = self.get_parameter('preprocessing').get_parameter_value().bool_value
        self.stitch_interval_sec = self.get_parameter('stitch_interval_sec').get_parameter_value().double_value

        # --------------------------
        # Subscriptions
        # --------------------------
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10
        )

        self.mission_state_sub = self.create_subscription(
            String, '/mission_state', self.state_callback, 10
        )

        self.reached_sub = self.create_subscription(
            WaypointReached, '/mavros/mission/reached', self.reached_callback, 10
        )

        # GPS subscription (with BEST_EFFORT QoS to match MAVROS)
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        self.gps_sub = self.create_subscription(
            NavSatFix, '/mavros/global_position/global', self.gps_callback, qos_profile
        )

        # --------------------------
        # Timer
        # --------------------------
        self.stitch_timer = self.create_timer(self.stitch_interval_sec, self.timer_callback)

        # Helpers
        self.bridge = CvBridge()
        self.received_images = []

        self.get_logger().info(f"Subscribed to {image_topic}, stitch every {self.stitch_interval_sec}s")

    # --------------------------
    # Callbacks
    # --------------------------
    def rosimg_to_ndarray(self, msg: Image) -> np.ndarray:
        """Convert sensor_msgs/Image → NumPy array (H×W×C)."""
        enc = msg.encoding.lower()
        dtype = np.uint16 if enc.endswith('16') else np.uint8
        flat = np.frombuffer(msg.data, dtype=dtype)
        elems_per_row = msg.step // dtype().itemsize
        mat = flat.reshape((msg.height, elems_per_row))

        if enc in ('rgb8', 'bgr8', 'rgba8', 'bgra8'):
            nch = 3 if enc.endswith('8') else 4
            mat = mat[:, : msg.width * nch]
            mat = mat.reshape((msg.height, msg.width, nch))
            if nch == 4:
                mat = mat[:, :, :3]
        if enc == 'rgb8':
            mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        if mat.dtype == np.uint16:
            mat = cv2.convertScaleAbs(mat, alpha=(255.0 / 65535.0))
        return mat

    def image_callback(self, msg: Image):
        try:
            frame = self.rosimg_to_ndarray(msg)
            self.latest_frame = frame
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    def state_callback(self, msg: String):
        self.state = msg.data
        self.get_logger().info(f"Mission state: {self.state}")

    def gps_callback(self, msg: NavSatFix):
        self.current_lat = msg.latitude
        self.current_lon = msg.longitude
        self.current_alt = msg.altitude
        # self.get_logger().info(f"GPS: lat={self.current_lat}, lon={self.current_lon}, alt={self.current_alt}")

    def reached_callback(self, msg: WaypointReached):
        if self.transition_in_progress:
            return
        self.waypoint_current = msg.wp_seq
        self.get_logger().info(f"Reached waypoint {self.waypoint_current} (state={self.state})")
        if self.state == "stitching":
            self.reached = True
        else:
            self.reached = False

    def timer_callback(self):
        if (self.latest_frame is not None and
            self.reached and
            self.pic_counter < MAX_PICS and
            self.waypoint_current != self.waypoint_prev):

            self.waypoint_prev = self.waypoint_current
            self.received_images.append(self.latest_frame)
            self.pic_counter += 1

            # Save image with GPS metadata in filename
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            gps_str = f"_{self.current_lat:.6f}_{self.current_lon:.6f}_{self.current_alt:.1f}" if self.current_lat else ""
            out_file = os.path.join(os.path.dirname(self.output_path),
                                    f"pic_wp{self.waypoint_current}{gps_str}.jpg")
            cv2.imwrite(out_file, self.latest_frame)

            self.get_logger().info(f"Saved picture #{self.pic_counter} at {out_file}")

        elif self.pic_counter >= MAX_PICS:
            self.get_logger().info(f"Collected {self.pic_counter} images, stitching…")
            imgs = list(self.received_images)
            self.received_images.clear()
            self.pic_counter = 0

            status, pano = self.stitch_images(imgs)
            if status == cv2.Stitcher_OK:
                self.get_logger().info("Stitch successful")
                if self.crop:
                    pano = self.crop_image(pano)

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_file = os.path.join(os.path.dirname(self.output_path), f"stitched_{ts}.jpg")
                cv2.imwrite(out_file, pano)
                self.get_logger().info(f"Saved stitched panorama: {out_file}")

                cv2.imshow("Stitched", pano)
                cv2.waitKey(1)
            else:
                self.get_logger().error(f"Stitch failed (code={status})")

    # --------------------------
    # Image helpers
    # --------------------------
    def resize_images(self, images, widthThreshold=1500):
        resized = []
        for img in images:
            h, w = img.shape[:2]
            if w > widthThreshold:
                r = widthThreshold / w
                resized.append(cv2.resize(img, (widthThreshold, int(h*r))))
            else:
                resized.append(img)
        return resized

    def preprocess_images(self, images):
        if not self.preprocessing:
            return images
        out = []
        for img in images:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge((l,a,b))
            out.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
        return out

    def stitch_images(self, images):
        images = self.resize_images(images)
        images = self.preprocess_images(images)
        stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
        return stitcher.stitch(images)

    def crop_image(self, stitched):
        img = cv2.copyMakeBorder(stitched,10,10,10,10,cv2.BORDER_CONSTANT,value=(0,0,0))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            img = img[y:y+h, x:x+w]
        return img


def main(args=None):
    rclpy.init(args=args)
    node = ImageStitcherNode()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
