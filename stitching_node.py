#!/usr/bin/env python3
# ----------------------------------------------------
# Image Stitcher Node (ROS2 + OpenCV in Python)
# ----------------------------------------------------
# - Subscribes to camera images (sensor_msgs/Image)
# - Subscribes to mission state (String)
# - Subscribes to GPS (NavSatFix)
# - Stores images with GPS coords when waypoints are reached
# - After N images, stitches them with GPS + ORB keypoints
# - Saves and displays the stitched result
# ----------------------------------------------------

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, NavSatFix
from cv_bridge import CvBridge
from std_msgs.msg import String
import cv2
import numpy as np
import os
from mavros_msgs.msg import WaypointReached
from datetime import datetime
import math

MAX_PICS = 2   # number of images before stitching


class ImageStitcherNode(Node):
    def __init__(self):
        super().__init__('image_stitcher_node')

        # State variables
        self.transition_in_progress = False
        self.waypoint_current = 0
        self.waypoint_prev = -1
        self.latest_frame = None
        self.latest_gps = None
        self.state = "lap"
        self.reached = False
        self.pic_counter = 0

        # Parameters
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('output_path', '/home/bvorinnano/bv_ws/stitched_image')
        self.declare_parameter('crop', True)
        self.declare_parameter('preprocessing', False)
        self.declare_parameter('stitch_interval_sec', 5.0)

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.output_path = self.get_parameter('output_path').get_parameter_value().string_value
        self.crop = self.get_parameter('crop').get_parameter_value().bool_value
        self.preprocessing = self.get_parameter('preprocessing').get_parameter_value().bool_value
        self.stitch_interval_sec = self.get_parameter('stitch_interval_sec').get_parameter_value().double_value

        # Subscriptions
        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.mission_state_sub = self.create_subscription(String, '/mission_state', self.state_callback, 10)
        self.reached_sub = self.create_subscription(WaypointReached, '/mavros/mission/reached', self.reached_callback, 10)
        self.gps_sub = self.create_subscription(NavSatFix, '/mavros/global_position/global', self.gps_callback, 10)

        self.stitch_timer = self.create_timer(self.stitch_interval_sec, self.timer_callback)

        # Helpers
        self.bridge = CvBridge()
        self.received_images = []  # will store (image, (lat, lon, alt))

        self.get_logger().info(f"Subscribed to {image_topic}; stitch every {self.stitch_interval_sec}s")

    # -------------------------
    # Callbacks
    # -------------------------
    def rosimg_to_ndarray(self, msg: Image) -> np.ndarray:
        """Convert any sensor_msgs/Image to an H×W×C NumPy array."""
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
            self.get_logger().error(f"Failed to convert raw image: {e}")

    def gps_callback(self, msg: NavSatFix):
        self.latest_gps = (msg.latitude, msg.longitude, msg.altitude)

    def state_callback(self, msg: String):
        self.get_logger().info(f"Received State: state={msg.data}")
        self.state = msg.data

    def reached_callback(self, msg: WaypointReached):
        if self.transition_in_progress:
            return
        idx = msg.wp_seq
        self.get_logger().info(f'Reached waypoint {idx} (state={self.state})')
        self.waypoint_current = idx
        if self.state == "stitching":
            self.reached = True
            self.get_logger().info(f'Reached waypoint in {self.state}')
        else:
            self.reached = False
            self.get_logger().info(f'Reached waypoint, NOT in stitching')

    def timer_callback(self):
        if (self.latest_frame is not None and self.latest_gps is not None
            and self.reached and self.pic_counter < MAX_PICS
            and self.waypoint_current != self.waypoint_prev):

            self.get_logger().info(f"Took picture number: {self.pic_counter}")
            self.waypoint_prev = self.waypoint_current

            # Save (image, gps)
            self.received_images.append((self.latest_frame, self.latest_gps))
            self.get_logger().info(
                f"Stored image with GPS={self.latest_gps}"
            )

            out_file = os.path.join(os.path.dirname(self.output_path),
                                    f"pic_{self.waypoint_current}.jpg")
            cv2.imwrite(out_file, self.latest_frame)
            self.get_logger().info(f"Saved: {out_file}")
            self.pic_counter += 1

        elif self.pic_counter >= MAX_PICS:
            self.get_logger().info(f"Got all pictures: {self.pic_counter}, starting stitching")
            imgs = list(self.received_images)
            self.received_images.clear()

            status, pano = self.stitch_images(imgs)
            if status == cv2.Stitcher_OK:
                self.get_logger().info("Stitch successful")
                if self.crop:
                    pano = self.crop_image(pano)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_file = os.path.join(os.path.dirname(self.output_path), f"stitched_{ts}.jpg")
                cv2.imwrite(out_file, pano)
                self.get_logger().info(f"Saved stitched: {out_file}")
                cv2.imshow("Stitched", pano)
                cv2.waitKey(1)
            else:
                self.get_logger().error(f"Stitch failed (code={status})")

    # -------------------------
    # Utilities
    # -------------------------
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
        self.get_logger().info("Preprocessing (CLAHE)…")
        out = []
        for img in images:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge((l,a,b))
            out.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
        return out

    def deg2rad(self, d): 
        return d * math.pi / 180.0

    def gps_to_xy(self, lat0, lon0, lat, lon):
        """Convert GPS to local XY in meters relative to reference (lat0, lon0)."""
        R = 6378137.0
        dlat = self.deg2rad(lat - lat0)
        dlon = self.deg2rad(lon - lon0)
        x = dlon * R * math.cos(self.deg2rad(lat0))
        y = dlat * R
        return (x, y)

    def stitch_images(self, img_gps_list):
        """Stitch images using GPS for rough placement and ORB for refinement."""
        if len(img_gps_list) < 2:
            return (cv2.Stitcher_ERR_NEED_MORE_IMGS, None)

        # Reference image
        ref_img, (lat0, lon0, alt0) = img_gps_list[0]
        h, w = ref_img.shape[:2]
        pano_canvas = np.zeros((h*5, w*5, 3), dtype=np.uint8)
        offset_x, offset_y = 2*w, 2*h
        pano_canvas[offset_y:offset_y+h, offset_x:offset_x+w] = ref_img

        orb = cv2.ORB_create(5000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        kp1, des1 = orb.detectAndCompute(ref_img, None)

        for img, (lat, lon, alt) in img_gps_list[1:]:
            # Predict placement with GPS
            x, y = self.gps_to_xy(lat0, lon0, lat, lon)
            gsd = 0.2  # meters per pixel (tune this)
            tx = int(offset_x + x / gsd)
            ty = int(offset_y - y / gsd)

            # Refine with keypoints
            kp2, des2 = orb.detectAndCompute(img, None)
            if des1 is not None and des2 is not None:
                matches = bf.match(des2, des1)
                if len(matches) >= 4:
                    src_pts = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                    dst_pts = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if H is not None:
                        h2, w2 = img.shape[:2]
                        warped = cv2.warpPerspective(img, H, (pano_canvas.shape[1], pano_canvas.shape[0]))
                        mask = (warped > 0).astype(np.uint8)
                        pano_canvas = np.where(mask, warped, pano_canvas)
                        continue

            # Fallback: paste at GPS-estimated position
            h2, w2 = img.shape[:2]
            pano_canvas[ty:ty+h2, tx:tx+w2] = img

        return (cv2.Stitcher_OK, pano_canvas)

    def crop_image(self, stitched):
        self.get_logger().info("Cropping black borders…")
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
