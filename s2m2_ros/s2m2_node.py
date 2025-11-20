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
import open3d as o3d

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

from sensor_msgs.msg import CameraInfo, Image
from message_filters import Subscriber, ApproximateTimeSynchronizer



def image_pad(img, factor):
    with torch.no_grad():
        H,W = img.shape[-2:]

        H_new = math.ceil(H / factor) * factor
        W_new = math.ceil(W / factor) * factor

        pad_h = H_new - H
        pad_w = W_new - W

        p2d = (pad_w//2, pad_w-pad_w//2, 0, 0)
        img_pad = F.pad(img, p2d, "constant", 0)
        #
        p2d = (0,0, pad_h // 2, pad_h - pad_h // 2)
        img_pad = F.pad(img_pad, p2d, "constant", 0)

        img_pad_down = F.adaptive_avg_pool2d(img_pad, output_size=[H // factor, W // factor])
        img_pad = F.interpolate(img_pad_down, size=[H_new, W_new], mode='bilinear')

        h_s = pad_h // 2
        h_e = (pad_h - pad_h // 2)
        w_s = pad_w // 2
        w_e = (pad_w - pad_w // 2)
        if h_e==0 and w_e==0:
            img_pad[:, :, h_s:, w_s:] = img
        elif h_e==0:
            img_pad[:, :, h_s:, w_s:-w_e] = img
        elif w_e==0:
            img_pad[:, :, h_s:-h_e, w_s:] = img
        else:
            img_pad[:, :, h_s:-h_e, w_s:-w_e] = img

        return img_pad

def image_crop(img, img_shape):
    with torch.no_grad():
        H,W = img.shape[-2:]
        H_new, W_new = img_shape

        crop_h = H - H_new
        if crop_h > 0:
            crop_s = crop_h // 2
            crop_e = crop_h - crop_h // 2
            img = img[:,:,crop_s: -crop_e]

        crop_w = W - W_new
        if crop_w > 0:
            crop_s = crop_w // 2
            crop_e = crop_w - crop_w // 2
            img = img[:,:,:, crop_s: -crop_e]

        return img
    

class S2m2Node(Node):
    def __init__(self):
        super().__init__('s2m2_node')

        # -----------------------------
        # Subscribers (message_filters)
        # -----------------------------
        print("Load s2m2 model")
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

        # -----------------------------
        # Approximate Time Sync
        # -----------------------------
        sync_topics = [self.infra1_img_sub, self.infra1_info_sub, self.infra2_img_sub, self.infra2_info_sub]
        self.ts = ApproximateTimeSynchronizer(sync_topics, queue_size=10, slop=0.05)
        self.ts.registerCallback(self.sync_callback)

        # -----------------------------
        # Publishers
        # -----------------------------
        topic_depth_info = '/camera/camera/depth_s2m2/camera_info'
        topic_depth_image = '/camera/camera/depth_s2m2/depth'
        self.depth_info_pub = self.create_publisher(CameraInfo, topic_depth_info, 10)
        self.depth_pub = self.create_publisher(Image, topic_depth_image, 10)

    def get_disparity_map(self, left: np.ndarray, right: np.ndarray):
        # convert to RGB
        if left.ndim == 2:
            left = cv2.cvtColor(left, cv2.COLOR_GRAY2RGB)
        if right.ndim == 2:
            right = cv2.cvtColor(right, cv2.COLOR_GRAY2RGB)

        left_torch = (torch.from_numpy(left).permute(-1, 0, 1).unsqueeze(0)).half().to(self.device)  # (H,W,3) f32 -> (1,3,H,W) f16
        right_torch = (torch.from_numpy(right).permute(-1, 0, 1).unsqueeze(0)).half().to(self.device)  # (H,W,3) f32 -> (1,3,H,W) f16

        # pad images with smooth padding
        left_torch_pad = image_pad(left_torch, 32)
        right_torch_pad = image_pad(right_torch, 32)

        # predict disparity map
        with torch.no_grad():
            with torch.amp.autocast(enabled=True, device_type=self.device, dtype=torch.float16):
                pred_disp, pred_occ, pred_conf = self.model_s2m2(left_torch_pad, right_torch_pad)  # (1,1,H,W)

        # Remove padding
        img_height, img_width = left.shape[:2]
        pred_disp = image_crop(pred_disp, (img_height, img_width))
        # pred_occ = image_crop(pred_occ, (img_height, img_width))
        # pred_conf = image_crop(pred_conf, (img_height, img_width))

        return pred_disp.squeeze(0).squeeze(0)  # (H,W)

    def sync_callback(self,
                  left_img_msg: Image,
                  left_info_msg: CameraInfo,
                  right_img_msg: Image,
                  right_info_msg: CameraInfo):
        assert np.allclose(left_info_msg.k, right_info_msg.k), "Left and right images should have identical intrinsics"

        try:
            cv_infra1 = self.bridge.imgmsg_to_cv2(left_img_msg, desired_encoding='mono8')
            cv_infra2 = self.bridge.imgmsg_to_cv2(right_img_msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        t1 = time.perf_counter()
        disp = self.get_disparity_map(cv_infra1, cv_infra2)
        fx = left_info_msg.k[0]
        baseline = 0.014711925759911537  # TODO: get from /camera/camera/extrinsics/depth_to_color or read from config
        depth = baseline * fx / disp

        dt_disp = time.perf_counter() - t1

        # convert to uint6, mm scaled, numpy depth map (e.g. like realsense depth topics)
        depth_np_mm = (depth*1000).squeeze(0).cpu().numpy()
        depth_mm_uint16 = depth_np_mm.astype(np.uint16)

        dt_depth = time.perf_counter() - t1
        dt_conv = dt_depth - dt_disp

        self.get_logger().info(f"dt_disp [ms]: {1e3*dt_disp}")
        self.get_logger().info(f"dt_conv [ms]: {1e3*dt_conv}")
        self.get_logger().info(f"dt_depth [ms]: {1e3*dt_depth}")

        # Convert depth to ROS Image
        depth_msg = self.bridge.cv2_to_imgmsg(depth_mm_uint16, encoding='16UC1')
        depth_msg.header = left_img_msg.header  # time + frame sync
        depth_info = left_info_msg  # depth image intrinsics is identical to the left image

        # ---------------------------------------------------------
        # Publish messages
        # ---------------------------------------------------------
        self.depth_info_pub.publish(depth_info)
        self.depth_pub.publish(depth_msg)

        self.get_logger().info("Published depth_s2m2 camera_info + depth")

def main(args=None):
    rclpy.init(args=args)
    node = S2m2Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
