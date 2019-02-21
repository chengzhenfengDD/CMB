# Author: Karl Sangwon Lee
# coding utf-8
# Last Updated: 02:11:2019

# Objective: This file fully translates 3D CNN model from the following paper into Tensorflow based Python 3.6
# Paper Reference: Automatic Detection of Cerebral Microbleeds From MR Images via 3D Convolutional Neural networks
# The general architecture will be used, but will be modified when neccessary

# imports
# 02:11:2019

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

import tensorflow as tf
# This one may be changed
from tensorflow.python.framework import ops
# Utility file will be made
# from cnn_utils import *



###################################################

# The following model is a basis model example in Python 3.6 Tensorflow

# Translation of the paper model to tensorflow architecture

def forward_propagation(X, parameters):

    """
    Paper Model:

    CONV1 -> RELU -> MAXPOOL1 -> CONV2 -> RELU -> FLATTEN -> FULLYCONNECTED

    *Question: why is the paper model not going through another maxpool?

    Arguments:

    X -- input dataset
    parameters -- python dictionary of Weights

    Returns:

    Z -- The output of last linear unit

    """

    # corresponds to W_L0
    W1 = parameters['W1']
    # corresponds to W_L1
    W2 = parameters['W2']

    # CONV3D: stride of 1, padding: 'VALID'
    # C1
    Z1 = tf.nn.conv3d(X, W1, strides = [1,1,1,1,1], padding = 'VALID', name = 'C1')
    # ReLU
    A1 = tf.nn.relu(Z1)

    # MAXPOOL3D: window 2x2x2, stride 2, padding 'VALID'
    P1 = tf.nn.max_pool3d(A1, ksize = [1,2,2,2,1], strides = [1,2,2,2,1], padding = 'VALID', name = 'M1')
    # Seems like we don't need to specify channel, prob because it's already in the filter Weight

    # CONV3D: stride of 1, padding: 'VALID'
    Z2 = tf.nn.conv3d(P1, W2, strides = [1,1,1,1,1], padding = 'VALID', name = 'C2')
    # ReLU
    A2 = tf.nn.relu(Z2)

    # FLATTEN
    A2 = tf.contrib.layers.flatten(A2)

    # FULLY-CONNECTED 1
    Z3 = tf.contrib.layers.fully_connected(A2,500)

    # FULLY-CONNECTED 2
    Z4 = tf.contrib.layers.fully_connected(Z3,100)

    # FULLY CONNECTED 3
    Z5 = tf.contrib.layers.fully_connected(Z4,2, activation_fn = )


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []                                        # To keep track of the cost

    # Create Placeholders of the correct shape
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters()
    ### END CODE HERE ###

    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z3 = forward_propagation(X, parameters)
    ### END CODE HERE ###

    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z3,Y)
    ### END CODE HERE ###

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    ### END CODE HERE ###

    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , temp_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X, Y:minibatch_Y})
                ### END CODE HERE ###

                minibatch_cost += temp_cost / num_minibatches


            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)


        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters


# Run the following cell to train your model for 100 epochs. Check if your cost after epoch 0 and 5 matches our output. If not, stop the cell and go back to your code!

# In[33]:

_, _, parameters = model(X_train, Y_train, X_test, Y_test)


###################################################################################

# The following is for translation.
# Translation: Python 2.7 - > 3.6


def evaluate_cnn3d(learning_rate = 0.03, n_epochs = 30, batch_size = 1):

    results_path = '../result/final_prediction/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Path for the pretrained model model parameters
    # This will come useful!
    # Update: The weights are in Theano Kernel. Must be converted to Tensorflow
    # Conversion Reference: https://github.com/keras-team/keras/wiki/Converting-convolution-kernels-from-Theano-to-TensorFlow-and-vice-versa

    model_path = '../model/fine_tuned_params_step2.pkl'
    f_param = open(model_path,'r')

    # It seems to load the parameter using pickle...
    # This may be a problem for the new python?
    # Update: pickle can be used instead.


    params = cPickle.load(f_param)
    print('params legnth:' + len(params))

    params_L0, params_L1, params_L2, params_L3, params_L4 = [param for param in params]

    W_L0 = params_L0[0].eval()
    b_L0 = params_L0[1].eval()

    W_L1 = params_L1[0].eval()
    b_L1 = params_L1[1].eval()

    W_L2 = params_L2[0].eval()
    b_L2 = params_L2[1].eval()

    W_L3 = params_L3[0].eval()
    b_L3 = params_L3[1].eval()

    W_L4 = params_L4[0].eval()
    b_L4 = params_L4[1].eval()

    f_param.close()
    print(params loaded!)

    print 'weights shape:', W_L0.shape, W_L1.shape, W_L2.shape, W_L3.shape, W_L4.shape
    cand_num = theano.shared(np.asarray(1,dtype='int32'))
    test_set_x = sharedata(data=numpy.ones([1,20*20*16]))
    print 'prepare data done'

    in_channels = 1
    in_time = 16
    in_width = 20
    in_height = 20

    x = T.matrix('x')
    y = T.ivector('y')
    batch_size = T.iscalar('batch_size')

    #define filter shape of the first layer
    flt_channels_L0 = 32
    flt_time_L0 = 5
    flt_width_L0 = 7
    flt_height_L0 = 7
    filter_shape_L0 = (flt_channels_L0,flt_time_L0,in_channels,flt_height_L0,flt_width_L0)

    #define filter shape of the second layer
    flt_channels_L1 = 64
    flt_time_L1 = 3
    flt_width_L1 = 5
    flt_height_L1 = 5
    in_channels_L1 = flt_channels_L0
    filter_shape_L1 = (flt_channels_L1,flt_time_L1,in_channels_L1,flt_height_L1,flt_width_L1)

    layer0_input = x.reshape((batch_size,in_channels,in_time,in_height,in_width)).dimshuffle(0,2,1,3,4)

    layer0 = ConvPool3dLayer(
        input = layer0_input,
        W = W_L0,
        b = b_L0,
        filter_shape = filter_shape_L0,
        poolsize=(2,2,2))

    layer1 = ConvPool3dLayer(
        input = layer0.output,
        W = W_L1,
        b = b_L1,
        filter_shape = filter_shape_L1,
        poolsize = (1,1,1))

    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        input = layer2_input,
        W = W_L2,
        b = b_L2)

    layer3 = HiddenLayer(
        input = layer2.output,
        W = W_L3,
        b = b_L3)

    layer4 = LogisticRegression(input = layer3.output, W=W_L4, b=b_L4)

    test_model = theano.function(
        inputs = [],
        outputs = [layer4.positive_prob,layer3.output],
        givens = {x: test_set_x, batch_size: cand_num})


    print '...testing...'
    datapath = '../result/test_set_cand/'
    files = os.listdir(datapath)
    n_cases = len(files)
    print 'n_cases:',files
    start_time = time.time()
    for cs in xrange(n_cases):
        case = cs + 1
        set_x = np.array(h5py.File(datapath + str(case) + '_patches.mat')['test_set_x'])
        set_x = np.transpose(set_x) - np.mean(set_x)
        print 'predicting {0} subject, contains {1} candidates...'.format(case, set_x.shape[0])
        cand_num.set_value(set_x.shape[0])
        test_set_x.set_value(set_x.astype(floatX))
        prediction, feature = test_model()
        sio.savemat(results_path + str(case)+'_prediction.mat',{'prediction':prediction})
    end_time = time.time()
    print 'time spent {} seconds.'.format((end_time-start_time)/n_cases)
