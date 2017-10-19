import caffe
import numpy as np
import pickle
import argparse
from collections import OrderedDict


parser = argparse.ArgumentParser()
parser.add_argument('--prototxt',
                    default='/media/tim_ho/HDD1/Projects/VNect-tensorflow/models/vnect_net.prototxt')
parser.add_argument('--caffemodel',
                    default='/media/tim_ho/HDD1/Projects/VNect-tensorflow/models/vnect_model.caffemodel')
parser.add_argument('--output_file',
                    default='vnect.pkl')
args = parser.parse_args()

if __name__ == '__main__':

    pkl_weights = OrderedDict()

    net = caffe.Net(args.prototxt,
                    args.caffemodel,
                    caffe.TEST)

    for layer in net.params.keys():
        print(layer)

    print('======')
    cur_bn_name = ''
    for layer in net.params.keys():
        print(layer, len(net.params[layer]))

        for i in range(len(net.params[layer])):
            print(net.params[layer][i].data.shape)

        if layer.startswith('bn'):
            cur_bn_name = layer
            pkl_weights[layer+'/moving_mean'] = np.asarray(net.params[layer][0].data) / net.params[layer][2].data
            pkl_weights[layer+'/moving_variance'] = np.asarray(net.params[layer][1].data) / net.params[layer][2].data
        elif layer.startswith('scale'):
            pkl_weights[cur_bn_name+'/gamma'] = np.asarray(net.params[layer][0].data)
            pkl_weights[cur_bn_name+'/beta'] = np.asarray(net.params[layer][1].data)
        elif len(net.params[layer]) == 2:
            pkl_weights[layer+'/weights'] = np.asarray(net.params[layer][0].data).transpose((2,3,1,0))
            pkl_weights[layer+'/biases'] = np.asarray(net.params[layer][1].data)
        elif len(net.params[layer]) == 1:
            pkl_weights[layer+'/kernel'] = np.asarray(net.params[layer][0].data).transpose((2,3,1,0))

    for layer in pkl_weights.keys():
        print(layer, pkl_weights[layer].shape)

    with open(args.output_file, 'wb') as f:
        pickle.dump(pkl_weights, f)