import caffe
import argparse
import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='gpu')
parser.add_argument('--model_dir', default='/media/tim_ho/HDD1/Projects/VNect-tensorflow/models')
parser.add_argument('--input_size', default=368)
parser.add_argument('--num_of_joints', default=21)
parser.add_argument('--pool_scale', default=8)
parser.add_argument('--plot_2d', default=False)
parser.add_argument('--plot_3d', default=False)
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


def demo():
    joints_2d = np.zeros(shape=(args.num_of_joints, 2), dtype=np.int32)
    joints_3d = np.zeros(shape=(args.num_of_joints, 3), dtype=np.float32)

    if args.plot_3d:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)
        plt.show()

    if args.device == 'cpu':
        caffe.set_mode_cpu()
    elif args.device == 'gpu':
        caffe.set_mode_gpu()
        caffe.set_device(0)
    else:
        raise ValueError('No such device')

    model_prototxt_path = os.path.join(args.model_dir, 'vnect_net.prototxt')
    model_weight_path = os.path.join(args.model_dir, 'vnect_model.caffemodel')

    # Load model
    model = caffe.Net(model_prototxt_path,
                      model_weight_path,
                      caffe.TEST)

    # Show network structure and shape
    for layer_name in model.params.keys():
        print(layer_name, model.params[layer_name][0].data.shape)
    print('')

    for i in model.blobs.keys():
        print(i, model.blobs[i].data.shape)

    cam = cv2.VideoCapture(0)
    is_tracking = False
    # for img_name in os.listdir('test_imgs'):
    while True:
        # if not is_tracking:

        img_path = 'test_imgs/{}'.format('dance.jpg')
        t1 = time.time()
        input_batch = []

        cam_img = utils.read_square_image('', cam, args.input_size, 'WEBCAM')
        # cam_img = utils.read_square_image(img_path, '', args.input_size, 'IMAGE')
        # cv2.imshow('', cam_img)
        # cv2.waitKey(0)
        orig_size_input = cam_img.astype(np.float32)

        for scale in scales:
            resized_img = utils.resize_pad_img(orig_size_input, scale, args.input_size)
            input_batch.append(resized_img)

        input_batch = np.asarray(input_batch, dtype=np.float32)
        input_batch = np.transpose(input_batch, (0, 3, 1, 2))
        input_batch /= 255.0
        input_batch -= 0.4

        model.blobs['data'].data[...] = input_batch

        # Forward
        model.forward()

        # Get output data
        x_hm = model.blobs['x_heatmap'].data
        y_hm = model.blobs['y_heatmap'].data
        z_hm = model.blobs['z_heatmap'].data
        hm = model.blobs['heatmap'].data

        # Trans coordinates
        x_hm = x_hm.transpose([0, 2, 3, 1])
        y_hm = y_hm.transpose([0, 2, 3, 1])
        z_hm = z_hm.transpose([0, 2, 3, 1])
        hm = hm.transpose([0, 2, 3, 1])

        # Average scale outputs
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

        t2 = time.time()
        # Get 2d joints
        joints_2d = utils.extract_2d_joint_from_heatmap(hm_avg, args.input_size, joints_2d)

        # Get 3d joints
        joints_3d = utils.extract_3d_joints_from_heatmap(joints_2d, x_hm_avg, y_hm_avg, z_hm_avg, args.input_size,
                                                         joints_3d)
        print('Post FPS', 1/(time.time()-t2))

        # Plot 2d location heatmap
        joint_map = np.zeros(shape=(args.input_size, args.input_size, 3))
        for joint_num in range(joints_2d.shape[0]):
            cv2.circle(joint_map, center=(joints_2d[joint_num][1], joints_2d[joint_num][0]), radius=3,
                       color=(255, 0, 0), thickness=-1)

        # Plot 2d limbs
        limb_img = utils.draw_limbs_2d(cam_img, joints_2d, limb_parents)

        # Plot 3d limbs
        if args.plot_3d:
            ax.clear()
            ax.view_init(azim=0, elev=90)
            ax.set_xlim(-700, 700)
            ax.set_ylim(-800, 800)
            ax.set_zlim(-700, 700)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            utils.draw_limbs_3d(joints_3d, limb_parents, ax)

        # draw heatmap
        # hm_img = utils.draw_predicted_heatmap(hm_avg*200, args.input_size)
        # cv2.imshow('hm', hm_img.astype(np.uint8))
        # cv2.waitKey(0)


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        utils.draw_limb_3d_gl(joints_3d, limb_parents)
        pygame.display.flip()
        pygame.time.wait(1)


        concat_img = np.concatenate((limb_img, joint_map), axis=1)

        # ax2.imshow(concat_img[..., ::-1].astype(np.uint8))
        cv2.imshow('2d', concat_img.astype(np.uint8))
        cv2.waitKey(1)
        # ax2.imshow(concat_img.astype(np.uint8))
        # plt.pause(0.0001)
        # plt.show(block=False)
        print('Forward FPS', 1 / (time.time() - t1))


if __name__ == '__main__':
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(70, (display[0] / display[1]), 0.1, 200.0)
    view_range = 800
    # glOrtho(-view_range, view_range,
    #         -view_range, view_range,
    #         -view_range, view_range)

    glTranslatef(0.0, 0.0, 100)

    demo()
