#!/bin/bash

python demo_tf_gl.py --device=gpu \
		     --demo_type=webcam \
		     --test_img=test_imgs/yuniko.jpg \
		     --model_file=models/weights/vnect_tf \
		     --plot_2d=True \
		     --plot_3d=True 
