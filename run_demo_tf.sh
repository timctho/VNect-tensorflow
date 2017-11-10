#!/bin/bash

python demo_tf.py --device=gpu \
		  --demo_type=image \
		  --test_img=test_imgs/yuniko.jpg \
		  --model_file=models/weights/vnect_tf \
		  --plot_2d=True \
		  --plot_3d=True 
