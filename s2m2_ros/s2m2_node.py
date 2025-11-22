#!/usr/bin/env python3

import os
import math
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import torch
import torch.nn.functional as F
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from s2m2.s2m2 import load_model
from s2m2.config import S2M2_PRETRAINED_WEIGHTS_PATH


import torch._dynamo
torch._dynamo.config.verbose=True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException



def image_pad(img: torch.Tensor, factor: int):
    """
    Pad img so that its dimensions are a multiple of `factor`.

    img: torch.Tensor (B,C,H,W), img needing to be padded
    factor: factor by which the new img dimensions should be divisible

    Out: torch.Tensor (B,C,H_new,W_new), padded image with H_new and W_new divisible by factor 
    """
    with torch.no_grad():
        B, C, H, W = img.shape
        assert H > factor, W > factor

        H_new = math.ceil(H / factor) * factor
        W_new = math.ceil(W / factor) * factor
        
        # pad the original img
        pad_h = H_new - H
        pad_w = W_new - W
        pad = (
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2
        )
        padded = F.pad(img, pad, "constant", 0)

        # smooth the whole padded img
        down = F.adaptive_avg_pool2d(padded, (H // factor, W // factor))
        img_smooth_pad = F.interpolate(down, (H_new, W_new),
                            mode="bilinear", align_corners=False)

        # put original img back in non-padded area
        h_s, w_s = pad_h // 2, pad_w // 2
        img_smooth_pad[..., h_s:h_s+H, w_s:w_s+W] = img
        return img_smooth_pad


def image_crop(img: torch.Tensor, H_new: int, W_new: int) -> torch.Tensor:
    """
    Center crop the image to a new dimension
    """
    with torch.no_grad():
        H, W = img.shape[-2], img.shape[-1]
        if H_new > H or W_new > W:
            raise ValueError(f"Crop size {(H_new, W_new)} exceeds image size {(H, W)}")
        top  = (H - H_new) // 2
        left = (W - W_new) // 2
        return img[..., top:top + H_new, left:left + W_new]


def depth_to_pointcloud(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    """
    depth_mm: (H, W) depth in meters
    returns Nx3 float32 array (X,Y,Z) in meters
    """
    h, w = depth.shape

    # pixel coordinate grid
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))

    # back project into 3D
    X = (xs - cx) * depth / fx
    Y = (ys - cy) * depth / fy

    # flatten
    points = np.stack((X, Y, depth), axis=-1).reshape(-1, 3)
    return points



class S2m2Node(Node):
    def __init__(self):
        super().__init__('s2m2_node')

        # ----------------------------
        # Load s2m2 stereo depth model
        # ----------------------------
        model_type = "S"  # select model type: S,M,L,XL
        allow_negative = False  # TODO: figure out what this is
        num_refine = 3
        self.device = "cuda"
        self.model_s2m2 = load_model(
            S2M2_PRETRAINED_WEIGHTS_PATH,
            model_type,
            allow_negative,
            num_refine,
        ).to(self.device).eval()
        # self.model_s2m2 = torch.compile(self.model_s2m2)
        self.get_logger().info(f"Loaded s2m2 model '{model_type}' on device {self.device}")

        self.bridge = CvBridge()

        # -----------------------------
        # Subscribers (message_filters)
        # -----------------------------
        topic_left_image_rect = '/camera/camera/infra1/image_rect_raw'
        topic_left_info = '/camera/camera/infra1/camera_info'
        topic_right_image_rect = '/camera/camera/infra2/image_rect_raw'
        topic_right_info = '/camera/camera/infra2/camera_info'
        self.infra1_img_sub = Subscriber(self, Image, topic_left_image_rect)
        self.infra1_info_sub = Subscriber(self, CameraInfo, topic_left_info)
        self.infra2_img_sub = Subscriber(self, Image, topic_right_image_rect)
        self.infra2_info_sub = Subscriber(self, CameraInfo, topic_right_info)

        # tf needed to get the baseline
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # -----------------------------
        # Approximate Time Sync
        # -----------------------------
        sync_topics = [self.infra1_img_sub, self.infra1_info_sub, self.infra2_img_sub, self.infra2_info_sub]
        self.ts = ApproximateTimeSynchronizer(sync_topics, queue_size=1, slop=0.05)
        self.ts.registerCallback(self.sync_callback)

        # -----------------------------
        # Publishers
        # -----------------------------
        topic_depth_info = '/camera/camera/depth_s2m2/camera_info'
        topic_depth_image = '/camera/camera/depth_s2m2/depth'
        self.depth_info_pub = self.create_publisher(CameraInfo, topic_depth_info, 1)
        self.depth_pub = self.create_publisher(Image, topic_depth_image, 1)

        self.topic_s2m2_points = '/camera/camera/depth_s2m2/points'
        self.pc_pub = self.create_publisher(PointCloud2, self.topic_s2m2_points, 1)

        self.t0 = None

    def sync_callback(self,
            left_img_msg: Image,
            left_info_msg: CameraInfo,
            right_img_msg: Image,
            right_info_msg: CameraInfo
        ):
        assert np.allclose(left_info_msg.k, right_info_msg.k), "Left and right images should have identical intrinsics"

        if self.pc_pub.get_subscription_count() == 0 and self.depth_pub.get_subscription_count() == 0:
            self.get_logger().warn("No subscriber, return")
            return

        left_frame = left_img_msg.header.frame_id
        right_frame = right_img_msg.header.frame_id

        baseline = self.get_stereo_baseline(left_frame, right_frame, left_img_msg.header.stamp)
        if baseline is None:
            self.get_logger().warn("Baseline unavailable — skipping this frame")
            return

        try:
            cv_infra1 = self.bridge.imgmsg_to_cv2(left_img_msg, desired_encoding='mono8')
            cv_infra2 = self.bridge.imgmsg_to_cv2(right_img_msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        t1 = time.perf_counter()
        disp = self.get_disparity_map(cv_infra1, cv_infra2)
        fx = left_info_msg.k[0]
        depth = baseline * fx / disp 
        depth_np = depth.squeeze(0).cpu().numpy()

        # print some timings
        self.get_logger().info(f"dt_depth [ms]: {1e3*(time.perf_counter() - t1)}")

        # Convert to uint16, mm scaled, numpy depth map (e.g. like realsense depth topics)
        # Then convert to ROS Image msg
        depth_mm_uint16 = (depth_np*1000).astype(np.uint16)
        depth_msg = self.bridge.cv2_to_imgmsg(depth_mm_uint16, encoding='16UC1')
        depth_msg.header = left_img_msg.header  # time + frame sync
        depth_info = left_info_msg  # depth image intrinsics is identical to the left image

        # ---------------------------------------------------------
        # Publish depth messages
        # ---------------------------------------------------------
        self.depth_info_pub.publish(depth_info)
        self.depth_pub.publish(depth_msg)
        self.get_logger().info("Published depth_s2m2 camera_info + depth")

        if self.pc_pub.get_subscription_count() == 0:
            self.get_logger().warn(f"No {self.topic_s2m2_points}, return")
            return

        t1 = time.perf_counter()
        # Publish PointCloud2
        # -----------------------------------------
        k = left_info_msg.k  # stored in row-major order
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]
        points = depth_to_pointcloud(depth_np, fx, fy, cx, cy)

        # Remove invalid points (inf or NaN or zero depth)
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
        points = points[valid]

        # Create and publish PointCloud2 message
        pc_msg = pc2.create_cloud_xyz32(left_img_msg.header, points)
        # self.get_logger().info(f"dt after create_cloud_xyz32 [ms]: {1e3*(time.perf_counter() -t1)}")
        self.pc_pub.publish(pc_msg)
        self.get_logger().info(f"dt after publish(pc_msg) [ms]: {1e3*(time.perf_counter() -t1)}")

        self.get_logger().info("Published depth_s2m2 point cloud")

    def get_disparity_map(self, left: np.ndarray, right: np.ndarray):
        # convert to RGB
        if left.ndim == 2:
            left = cv2.cvtColor(left, cv2.COLOR_GRAY2RGB)
        if right.ndim == 2:
            right = cv2.cvtColor(right, cv2.COLOR_GRAY2RGB)

        left_torch = torch.tensor(left, device=self.device, dtype=torch.half).permute(2,0,1).unsqueeze(0)  # (H,W,3) f32 -> (1,3,H,W) f16
        right_torch = torch.tensor(right, device=self.device, dtype=torch.half).permute(2,0,1).unsqueeze(0)  # (H,W,3) f32 -> (1,3,H,W) f16

        # s2m2 model requires img dimensions divisible by 32 -> smooth pad the imgs
        left_torch_pad = image_pad(left_torch, 32)  # (1,3,H,W) -> (1,3,H_new,W_new)
        right_torch_pad = image_pad(right_torch, 32)  # (1,3,H,W) -> (1,3,H_new,W_new)

        # predict disparity map
        with torch.no_grad():
            with torch.amp.autocast(enabled=True, device_type=self.device, dtype=torch.float16):
                pred_disp, pred_occ, pred_conf = self.model_s2m2(left_torch_pad, right_torch_pad)  # (1,1,H,W)

        # Remove padding
        img_height, img_width = left.shape[:2]
        pred_disp = image_crop(pred_disp, img_height, img_width)  # (1,1,H_new,W_new) -> (1,1,H,W) 
        # pred_occ = image_crop(pred_occ, img_height, img_width)
        # pred_conf = image_crop(pred_conf, img_height, img_width)

        return pred_disp.squeeze(0).squeeze(0)  # (H,W)

    def get_stereo_baseline(self, left_frame: str, right_frame: str, img_time: rclpy.time.Time) -> float:
        """
        Returns the baseline (meters) as the absolute translation along x
        between left and right camera frames.
        """
        try:
            # lookup the transform from left → right
            tf = self.tf_buffer.lookup_transform(
                target_frame=left_frame,
                source_frame=right_frame,
                time=img_time,    # latest available
                timeout=rclpy.duration.Duration(seconds=0.2)
            )
            baseline = abs(tf.transform.translation.x)
            self.get_logger().info(
                f"TF stereo baseline between {left_frame} and {right_frame} = {baseline:.6f} m"
            )

            return baseline

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"Failed to lookup transform {left_frame} → {right_frame}: {e}")
            return None

def main(args=None):
    rclpy.init(args=args)
    node = S2m2Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
