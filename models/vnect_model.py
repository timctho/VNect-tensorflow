import tensorflow as tf
import tensorflow.contrib as tc

import pickle
import numpy as np


class VNect():
    def __init__(self, input_size):
        self.is_training = False
        self.input_holder = tf.placeholder(dtype=tf.float32,
                                           shape=(None, input_size, input_size, 3))
        self._create_network()

    def _create_network(self):
        # Conv
        self.conv1 = tc.layers.conv2d(self.input_holder, kernel_size=7, num_outputs=64, stride=2, scope='conv1')
        self.pool1 = tc.layers.max_pool2d(self.conv1, kernel_size=3, padding='same', scope='pool1')

        # Residual block 2a
        self.res2a_branch2a = tc.layers.conv2d(self.pool1, kernel_size=1, num_outputs=64, scope='res2a_branch2a')
        self.res2a_branch2b = tc.layers.conv2d(self.res2a_branch2a, kernel_size=3, num_outputs=64, scope='res2a_branch2b')
        self.res2a_branch2c = tc.layers.conv2d(self.res2a_branch2b, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2a_branch2c')
        self.res2a_branch1 = tc.layers.conv2d(self.pool1, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2a_branch1')
        self.res2a = tf.add(self.res2a_branch2c, self.res2a_branch1, name='res2a_add')
        self.res2a = tf.nn.relu(self.res2a, name='res2a')

        # Residual block 2b
        self.res2b_branch2a = tc.layers.conv2d(self.res2a, kernel_size=1, num_outputs=64, scope='res2b_branch2a')
        self.res2b_branch2b = tc.layers.conv2d(self.res2b_branch2a, kernel_size=3, num_outputs=64, scope='res2b_branch2b')
        self.res2b_branch2c = tc.layers.conv2d(self.res2b_branch2b, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2b_branch2c')
        self.res2b = tf.add(self.res2b_branch2c, self.res2a, name='res2b_add')
        self.res2b = tf.nn.relu(self.res2b, name='res2b')

        # Residual block 2c
        self.res2c_branch2a = tc.layers.conv2d(self.res2b, kernel_size=1, num_outputs=64, scope='res2c_branch2a')
        self.res2c_branch2b = tc.layers.conv2d(self.res2b_branch2a, kernel_size=3, num_outputs=64, scope='res2c_branch2b')
        self.res2c_branch2c = tc.layers.conv2d(self.res2b_branch2b, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2c_branch2c')
        self.res2c = tf.add(self.res2c_branch2c, self.res2b, name='res2c_add')
        self.res2c = tf.nn.relu(self.res2b, name='res2c')

        # Residual block 3a
        self.res3a_branch2a = tc.layers.conv2d(self.res2c, kernel_size=1, num_outputs=128, stride=2, scope='res3a_branch2a')
        self.res3a_branch2b = tc.layers.conv2d(self.res3a_branch2a, kernel_size=3, num_outputs=128, scope='res3a_branch2b')
        self.res3a_branch2c = tc.layers.conv2d(self.res3a_branch2b, kernel_size=1, num_outputs=512, activation_fn=None,scope='res3a_branch2c')
        self.res3a_branch1 = tc.layers.conv2d(self.res2c, kernel_size=1, num_outputs=512, activation_fn=None, stride=2, scope='res3a_branch1')
        self.res3a = tf.add(self.res3a_branch2c, self.res3a_branch1, name='res3a_add')
        self.res3a = tf.nn.relu(self.res3a, name='res3a')

        # Residual block 3b
        self.res3b_branch2a = tc.layers.conv2d(self.res3a, kernel_size=1, num_outputs=128, scope='res3b_branch2a')
        self.res3b_branch2b = tc.layers.conv2d(self.res3b_branch2a, kernel_size=3, num_outputs=128,scope='res3b_branch2b')
        self.res3b_branch2c = tc.layers.conv2d(self.res3b_branch2b, kernel_size=1, num_outputs=512, activation_fn=None,scope='res3b_branch2c')
        self.res3b = tf.add(self.res3b_branch2c, self.res3a, name='res3b_add')
        self.res3b = tf.nn.relu(self.res3b, name='res3b')

        # Residual block 3c
        self.res3c_branch2a = tc.layers.conv2d(self.res3b, kernel_size=1, num_outputs=128, scope='res3c_branch2a')
        self.res3c_branch2b = tc.layers.conv2d(self.res3c_branch2a, kernel_size=3, num_outputs=128,scope='res3c_branch2b')
        self.res3c_branch2c = tc.layers.conv2d(self.res3c_branch2b, kernel_size=1, num_outputs=512, activation_fn=None,scope='res3c_branch2c')
        self.res3c = tf.add(self.res3c_branch2c, self.res3b, name='res3c_add')
        self.res3c = tf.nn.relu(self.res3c, name='res3c')

        # Residual block 3d
        self.res3d_branch2a = tc.layers.conv2d(self.res3c, kernel_size=1, num_outputs=128, scope='res3d_branch2a')
        self.res3d_branch2b = tc.layers.conv2d(self.res3d_branch2a, kernel_size=3, num_outputs=128,scope='res3d_branch2b')
        self.res3d_branch2c = tc.layers.conv2d(self.res3d_branch2b, kernel_size=1, num_outputs=512, activation_fn=None,scope='res3d_branch2c')
        self.res3d = tf.add(self.res3d_branch2c, self.res3b, name='res3d_add')
        self.res3d = tf.nn.relu(self.res3d, name='res3d')

        # Residual block 4a
        self.res4a_branch2a = tc.layers.conv2d(self.res3d, kernel_size=1, num_outputs=256, stride=2, scope='res4a_branch2a')
        self.res4a_branch2b = tc.layers.conv2d(self.res4a_branch2a, kernel_size=3, num_outputs=256,scope='res4a_branch2b')
        self.res4a_branch2c = tc.layers.conv2d(self.res4a_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None,scope='res4a_branch2c')
        self.res4a_branch1 = tc.layers.conv2d(self.res3d, kernel_size=1, num_outputs=1024, activation_fn=None, stride=2, scope='res4a_branch1')
        self.res4a = tf.add(self.res4a_branch2c, self.res4a_branch1, name='res4a_add')
        self.res4a = tf.nn.relu(self.res4a, name='res4a')

        # Residual block 4b
        self.res4b_branch2a = tc.layers.conv2d(self.res4a, kernel_size=1, num_outputs=256, scope='res4b_branch2a')
        self.res4b_branch2b = tc.layers.conv2d(self.res4b_branch2a, kernel_size=3, num_outputs=256, scope='res4b_branch2b')
        self.res4b_branch2c = tc.layers.conv2d(self.res4b_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4b_branch2c')
        self.res4b = tf.add(self.res4b_branch2c, self.res4a, name='res4b_add')
        self.res4b = tf.nn.relu(self.res4b, name='res4b')

        # Residual block 4c
        self.res4c_branch2a = tc.layers.conv2d(self.res4b, kernel_size=1, num_outputs=256, scope='res4c_branch2a')
        self.res4c_branch2b = tc.layers.conv2d(self.res4c_branch2a, kernel_size=3, num_outputs=256, scope='res4c_branch2b')
        self.res4c_branch2c = tc.layers.conv2d(self.res4c_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4c_branch2c')
        self.res4c = tf.add(self.res4c_branch2c, self.res4b, name='res4c_add')
        self.res4c = tf.nn.relu(self.res4c, name='res4c')

        # Residual block 4d
        self.res4d_branch2a = tc.layers.conv2d(self.res4c, kernel_size=1, num_outputs=256, scope='res4d_branch2a')
        self.res4d_branch2b = tc.layers.conv2d(self.res4d_branch2a, kernel_size=3, num_outputs=256, scope='res4d_branch2b')
        self.res4d_branch2c = tc.layers.conv2d(self.res4d_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4d_branch2c')
        self.res4d = tf.add(self.res4d_branch2c, self.res4c, name='res4d_add')
        self.res4d = tf.nn.relu(self.res4d, name='res4d')

        # Residual block 4e
        self.res4e_branch2a = tc.layers.conv2d(self.res4d, kernel_size=1, num_outputs=256, scope='res4e_branch2a')
        self.res4e_branch2b = tc.layers.conv2d(self.res4e_branch2a, kernel_size=3, num_outputs=256, scope='res4e_branch2b')
        self.res4e_branch2c = tc.layers.conv2d(self.res4e_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4e_branch2c')
        self.res4e = tf.add(self.res4e_branch2c, self.res4d, name='res4e_add')
        self.res4e = tf.nn.relu(self.res4e, name='res4e')

        # Residual block 4f
        self.res4f_branch2a = tc.layers.conv2d(self.res4e, kernel_size=1, num_outputs=256, scope='res4f_branch2a')
        self.res4f_branch2b = tc.layers.conv2d(self.res4f_branch2a, kernel_size=3, num_outputs=256, scope='res4f_branch2b')
        self.res4f_branch2c = tc.layers.conv2d(self.res4f_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4f_branch2c')
        self.res4f = tf.add(self.res4f_branch2c, self.res4e, name='res4f_add')
        self.res4f = tf.nn.relu(self.res4f, name='res4f')

        # Residual block 5a
        self.res5a_branch2a_new = tc.layers.conv2d(self.res4f, kernel_size=1, num_outputs=512, scope='res5a_branch2a_new')
        self.res5a_branch2b_new = tc.layers.conv2d(self.res5a_branch2a_new, kernel_size=3, num_outputs=512, scope='res5a_branch2b_new')
        self.res5a_branch2c_new = tc.layers.conv2d(self.res5a_branch2b_new, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res5a_branch2c_new')
        self.res5a_branch1_new = tc.layers.conv2d(self.res4f, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res5a_branch1_new')
        self.res5a = tf.add(self.res5a_branch2c_new, self.res5a_branch1_new, name='res5a_add')
        self.res5a = tf.nn.relu(self.res5a, name='res5a')

        # Residual block 5b
        self.res5b_branch2a_new = tc.layers.conv2d(self.res5a, kernel_size=1, num_outputs=256, scope='res5b_branch2a_new')
        self.res5b_branch2b_new = tc.layers.conv2d(self.res5b_branch2a_new, kernel_size=3, num_outputs=128, scope='res5b_branch2b_new')
        self.res5b_branch2c_new = tc.layers.conv2d(self.res5b_branch2b_new, kernel_size=1, num_outputs=256, scope='res5b_branch2c_new')

        # Transpose Conv
        self.res5c_branch1a = tf.layers.conv2d_transpose(self.res5b_branch2c_new, kernel_size=4, filters=63, activation=None, strides=2, padding='same', use_bias=False, name='res5c_branch1a')
        self.res5c_branch2a = tf.layers.conv2d_transpose(self.res5b_branch2c_new, kernel_size=4, filters=128, activation=None, strides=2, padding='same', use_bias=False, name='res5c_branch2a')
        self.bn5c_branch2a = tc.layers.batch_norm(self.res5c_branch2a, scale=True, is_training=self.is_training, scope='bn5c_branch2a')
        self.bn5c_branch2a = tf.nn.relu(self.bn5c_branch2a)

        self.res5c_delta_x, self.res5c_delta_y, self.res5c_delta_z = tf.split(self.res5c_branch1a, num_or_size_splits=3, axis=3)
        self.res5c_branch1a_sqr = tf.multiply(self.res5c_branch1a, self.res5c_branch1a, name='res5c_branch1a_sqr')
        self.res5c_delta_x_sqr, self.res5c_delta_y_sqr, self.res5c_delta_z_sqr = tf.split(self.res5c_branch1a_sqr, num_or_size_splits=3, axis=3)
        self.res5c_bone_length_sqr = tf.add(tf.add(self.res5c_delta_x_sqr, self.res5c_delta_y_sqr), self.res5c_delta_z_sqr)
        self.res5c_bone_length = tf.sqrt(self.res5c_bone_length_sqr)

        self.res5c_branch2a_feat = tf.concat([self.bn5c_branch2a, self.res5c_delta_x, self.res5c_delta_y, self.res5c_delta_z, self.res5c_bone_length],
                                             axis=3, name='res5c_branch2a_feat')

        self.res5c_branch2b = tc.layers.conv2d(self.res5c_branch2a_feat, kernel_size=3, num_outputs=128, scope='res5c_branch2b')
        self.res5c_branch2c = tf.layers.conv2d(self.res5c_branch2b, kernel_size=1, filters=84, activation=None, use_bias=False, name='res5c_branch2c')
        self.heapmap, self.x_heatmap, self.y_heatmap, self.z_heatmap = tf.split(self.res5c_branch2c, num_or_size_splits=4, axis=3)


    @property
    def all_vars(self):
        return tf.global_variables()


    def load_weights(self, sess, weight_file):
        # Read pretrained model file
        model_weights = pickle.load(open(weight_file, 'rb'))

        # For each layer each var
        with tf.variable_scope('', reuse=True):
            for variable in tf.global_variables():
                var_name = variable.name.split(':')[0]
                self._assign_weights_from_dict(var_name, model_weights, sess)


    def _assign_weights_from_dict(self, var_name, model_weights, sess):
        with tf.variable_scope('', reuse=True):
            var_tf = tf.get_variable(var_name)
            # print(var_tf)
            sess.run(tf.assign(var_tf, model_weights[var_name]))
            np.testing.assert_allclose(var_tf.eval(sess), model_weights[var_name])




if __name__ == '__main__':
    model_file = 'vnect.pkl'
    model = VNect(368)


    with tf.Session() as sess:
        saver = tf.train.Saver()
        tf_writer = tf.summary.FileWriter(logdir='./', graph=sess.graph)

        sess.run(tf.global_variables_initializer())
        print(model.res5b_branch2c_new)
        print(model.heapmap, model.x_heatmap, model.y_heatmap, model.z_heatmap)


