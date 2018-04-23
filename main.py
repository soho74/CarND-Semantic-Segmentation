import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests



# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
tf.device('/gpu:2')

def load_vgg(sess, vgg_path):
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)


    return w1, keep, layer3, layer4, layer7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same', strides=(1, 1), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output1 = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, strides=(2, 2), padding='same', kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same', strides=(1, 1), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output2 = tf.add(output1, conv_1x1)
    output3 = tf.layers.conv2d_transpose(output2, num_classes, 4, strides=(2, 2), padding='same', kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same', strides=(1, 1), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output4 = tf.add(output3, conv_1x1)
    output5 = tf.layers.conv2d_transpose(output4, num_classes, 16, strides=(8, 8), padding='same', kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    return output5
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    #train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(loss = cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, kprob_rate, l_rate):
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        for image, label in get_batches_fn(batch_size):
            print("Batch", i)
            # Training
            sess.run(train_op, feed_dict={input_image: image, correct_label: label, keep_prob: kprob_rate, learning_rate: l_rate})

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    logs_dir = './logs'
    tests.test_for_kitti_dataset(data_dir)

    helper.maybe_download_pretrained_vgg(data_dir)
    
    epochs = 250
    batch_size = 16

    for l_rate in [0.0001, 0.0004]:

        for kprob_rate in [0.8, 0.9]:

            tf.reset_default_graph()            

            hparam = str(kprob_rate) +"_" + str(l_rate)
            print('Starting run for %s' % hparam)
            model_path = logs_dir + hparam + "/model"

            with tf.Session() as sess:
                vgg_path = os.path.join(data_dir, 'vgg')
                get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
                correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name= 'correct_label')
                learning_rate = tf.placeholder(tf.float32, name='learning_rate')

                input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

                layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

                logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

                saver = tf.train.Saver()

                sess.run(tf.global_variables_initializer())
                train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate, kprob_rate, l_rate)

                save_path = saver.save(sess, model_path)
                print("Model saved in file: %s" % save_path)

                helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


if __name__ == '__main__':
    run()
