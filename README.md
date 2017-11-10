# VNect -- Tensorflow version
This project is focused on tensorflow implementation of [VNect: Real-time 3D Human Pose Estimation with a Single RGB Camera](http://gvv.mpi-inf.mpg.de/projects/VNect/), SIGGRAPH 2017.

## Environments
- Ubuntu 16.04
- Python 2.7
- Tensorflow 1.3.0
- OpenCV 3.3.0
- OpenGL (optional)

## Model file
I converted the original caffe model into [tensorflow model](https://drive.google.com/open?id=1ETbIKHulW0ZhMZIfb_nFcQr-0InqWGor)

## Inference
- 1.Download model, put them in folder `models/weights`
- 2.Edit demo settings in shell script, `--device` `--demo_type` `--model_file` `--test_img` `--plot_2d` `--plot_3d`
- 3.If you have OpenGL, you can run `run_demo_tf_gl.sh` for faster rendering of 3d joints. Otherwise, run `run_demo_tf.sh`

# TODO
Training part of your own model.


