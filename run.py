import argparse
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from models.nets import vnect_model_bn_folded as vnect_model
import utils.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('--demo_type', default='image')
parser.add_argument('--device', default='cpu')
parser.add_argument('--model_file', default='models/weights/vnect_tf')
parser.add_argument('--test_img', default='test_imgs/yuniko.jpg')
parser.add_argument('--input_size', default=368)
parser.add_argument('--num_of_joints', default=21)
parser.add_argument('--pool_scale', default=8)
parser.add_argument('--plot_2d', default=True)
parser.add_argument('--plot_3d', default=True)
args = parser.parse_args()

joint_color_code = [[139, 53, 255],
                    [0, 56, 255],
                    [43, 140, 237],
                    [37, 168, 36],
                    [147, 147, 0],
                    [70, 17, 145]]

# Limb parents of each joint
limb_parents = [1, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13]

# input scales
scales = [1.0, 0.7]

# Use gpu or cpu
gpu_count = {'GPU':1} if args.device == 'gpu' else {'GPU':0}


def demo_tf():
    if args.plot_3d:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)
        plt.show()

    # Create model
    model_tf = vnect_model.VNect(args.input_size)

    # Create session
    sess_config = tf.ConfigProto(device_count=gpu_count)
    sess = tf.Session(config=sess_config)

    # Restore weights
    saver = tf.train.Saver()
    saver.restore(sess, args.model_file)

    # Joints placeholder
    joints_2d = np.zeros(shape=(args.num_of_joints, 2), dtype=np.int32)
    joints_3d = np.zeros(shape=(args.num_of_joints, 3), dtype=np.float32)

    # Establish stream of cameras
    if args.demo_type == 'webcam':
        cam = cv2.VideoCapture(0)
    elif args.demo_type == 'realsense':
        cam = cv2.VideoCapture(0)
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)

    while(True):
        # Get image for inference
        if args.demo_type == 'image':
            img_path = args.test_img
            cam_img = utils.read_square_image(img_path, '', args.input_size, 'image')
        elif args.demo_type == 'webcam':
            cam_img = utils.read_square_image('', cam, args.input_size, 'webcam')
        elif args.demo_type == 'realsense':
            frames = pipeline.wait_for_frames()
            cam_frames = frames.get_color_frame()
            cam_img = utils.read_square_image('', cam_frames, args.input_size, 'realsense')
    
        orig_size_input = cam_img.astype(np.float32)
        input_batch = []

        # Create multi-scale inputs
        for scale in scales:
            resized_img = utils.resize_pad_img(orig_size_input, scale, args.input_size)
            input_batch.append(resized_img)

        input_batch = np.asarray(input_batch, dtype=np.float32)
        input_batch /= 255.0
        input_batch -= 0.4

        # Inference
        inference_time = time.time()
        heatmaps = sess.run(
            [model_tf.heapmap, model_tf.x_heatmap, model_tf.y_heatmap, model_tf.z_heatmap],
            feed_dict={model_tf.input_holder: input_batch})
        print("Inference Time: {:>2.2f}".format((time.time() - inference_time)))
        
        # Average scale outputs
        [hm_avg, x_hm_avg, y_hm_avg, z_hm_avg] = average_scale_outputs(heatmaps)
        
        # Get 2d joints
        utils.extract_2d_joint_from_heatmap(hm_avg, args.input_size, joints_2d)

        # Get 3d joints
        utils.extract_3d_joints_from_heatmap(joints_2d, x_hm_avg, y_hm_avg, z_hm_avg, args.input_size, joints_3d)

        # Show results

        if args.plot_3d:
            plot_output(ax, ax2, plt, fig, joints_3d, joints_2d, cam_img)
        elif args.plot_2d:
            plot_output('', '', '', '', joints_3d, joints_2d, cam_img)

        if args.demo_type == 'image':
            return

def average_scale_outputs(heatmaps):
    [hm, x_hm, y_hm, z_hm] = heatmaps
    hm_size = args.input_size // args.pool_scale
    hm_avg = np.zeros(shape=(hm_size, hm_size, args.num_of_joints))
    x_hm_avg = np.zeros(shape=(hm_size, hm_size, args.num_of_joints))
    y_hm_avg = np.zeros(shape=(hm_size, hm_size, args.num_of_joints))
    z_hm_avg = np.zeros(shape=(hm_size, hm_size, args.num_of_joints))
    for i in range(len(scales)):
            rescale = 1.0 / scales[i]
            scaled_hm = cv2.resize(hm[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
            scaled_x_hm = cv2.resize(x_hm[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
            scaled_y_hm = cv2.resize(y_hm[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
            scaled_z_hm = cv2.resize(z_hm[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
            mid = [scaled_hm.shape[0] // 2, scaled_hm.shape[1] // 2]
            hm_avg += scaled_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
                      mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
            x_hm_avg += scaled_x_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
                        mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
            y_hm_avg += scaled_y_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
                        mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
            z_hm_avg += scaled_z_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
                        mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
    hm_avg /= len(scales)
    x_hm_avg /= len(scales)
    y_hm_avg /= len(scales)
    z_hm_avg /= len(scales)
    return [hm_avg, x_hm_avg, y_hm_avg, z_hm_avg]

def plot_output(ax, ax2, plt, fig, joints_3d, joints_2d, cam_img):
    if args.plot_2d:
        # Plot 2d joint location
        joint_map = np.zeros(shape=(args.input_size, args.input_size, 3))
        for joint_num in range(joints_2d.shape[0]):
            cv2.circle(joint_map, center=(joints_2d[joint_num][1], joints_2d[joint_num][0]), radius=3,
                       color=(255, 0, 0), thickness=-1)
        # Draw 2d limbs
        utils.draw_limbs_2d(cam_img, joints_2d, limb_parents)

    if args.plot_3d:
        ax.clear()
        ax.view_init(azim=-90, elev=-90)
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_zlim(-50, 50)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        utils.draw_limbs_3d(joints_3d, limb_parents, ax)

        if args.plot_2d:
            # Display 2d results
            concat_img = np.concatenate((cam_img[:, :, ::-1], joint_map), axis=1)
            ax2.imshow(concat_img.astype(np.uint8))
        if args.demo_type == 'image':
            plt.show(block=True)
        else:
            fig.canvas.draw()
            fig.canvas.flush_events()
    elif args.plot_2d:
        concat_img = np.concatenate((cam_img, joint_map), axis=1)
        cv2.imshow('2D img', concat_img.astype(np.uint8))
        if args.demo_type == 'image':
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)
        
if __name__ == '__main__':
    demo_tf()
