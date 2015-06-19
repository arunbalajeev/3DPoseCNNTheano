'''
Created on Mar 27, 2015

@author: arun
'''
from theano.tensor.basic import dmatrix

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.sandbox.cuda.dnn import *
import scipy.io #for old .mat
import h5py #for new .mat
import getopt#parsing command line


from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from mlp import ReturnValue

import cPickle
#from collections import namedtuple
#Weights = namedtuple("Weights", "field1 field2 field3 field4 field5 field6")

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), Wi=None, bi=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1];
        self.input = input;

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))



        if Wi is None:
            self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)
        else:
            self.W = theano.shared(value=numpy.asarray(Wi, dtype=theano.config.floatX),name='W', borrow=True)

        if bi is None:
            # the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            b_values = numpy.asarray(bi, dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)
        #conv_out = dnn_conv(input=input, filters=self.W,
        #        filter_shape=filter_shape, image_shape=image_shape)
        #conv_out = dnn_conv(img=input, kerns, border_mode, subsample, conv_mode, direction_hint, workmem)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)
        #pooled_out = dnn_pool(input=conv_out,
        #                                    ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]
    


def buildLayersD(layer0_input,batch_size, dim, nkerns, rng,TT=None):
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)

    W0 = None
    b0= None
    W1 = None
    b1= None
    W2 = None
    b2= None
    W3 = None
    b3= None
    W4 = None
    b4= None
    W5 = None
    b5 = None

    if TT != None:
        W0 = TT.Layer0_param.W.get_value(borrow=True)
        b0 = TT.Layer0_param.b.get_value(borrow=True)
        W1 = TT.Layer1_param.W.get_value(borrow=True)
        b1 = TT.Layer1_param.b.get_value(borrow=True)
        W2 = TT.Layer2_param.W.get_value(borrow=True)
        b2 = TT.Layer2_param.b.get_value(borrow=True)
        W3 = TT.Layer3_param.W.get_value(borrow=True)
        b3 = TT.Layer3_param.b.get_value(borrow=True)
        W4 = TT.Layer4_param.W.get_value(borrow=True)
        b4 = TT.Layer4_param.b.get_value(borrow=True)
        W5 = TT.Layer5_param.W.get_value(borrow=True)
        b5 = TT.Layer5_param.b.get_value(borrow=True)


    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, dim, 128, 128),
            filter_shape=(nkerns[0], dim, 5, 5), poolsize=(2, 2),Wi=W0,bi=b0)

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)



    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], 62, 62),
            filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2),Wi=W1,bi=b1)

    layer12 = LeNetConvPoolLayer(rng, input=layer1.output,
            image_shape=(batch_size, nkerns[1], 29, 29),
            filter_shape=(nkerns[2], nkerns[1], 6, 6), poolsize=(2, 2),Wi=W2,bi=b2)    #4,4

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer12.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[2] *12*12,             #* 6 * 13,(for 128X72)
                         n_out=1024,Wi=W3,bi=b3)
    layer3 = HiddenLayer(rng, input=layer2.output, n_in=1024,
                        n_out=2048,Wi=W4,bi=b4)
    layer4 = HiddenLayer(rng, input=layer3.output, n_in=2048,
                        n_out=1700,Wi=W5,bi=b5)
    #w = Weights(W0,b0,W1,b1,W2,b2);

    return [layer0, layer1, layer12, layer2, layer3, layer4, TT];
    #return [layer0, layer1, layer12, layer2];

def buildLayersR(layer0_input,batch_size, dim, nkerns, rng,TT=None,TR=None,par=None):
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    #if(TT==None):
    #    print 'Hello, it is None here......'

    W0 = None
    b0= None
    W1 = None
    b1= None
    W2 = None
    b2= None
    if par!=None:
        W0 = par[0]
        b0= par[1]
        W1 = par[2]
        b1= par[3]
        W2 = par[4]
        b2= par[5]
    W3 = None
    b3= None
    W4 = None
    b4= None
    W5 = None
    b5 = None

    if TT != None:
        W0 = TT.Layer0_param.W.get_value(borrow=True)
        b0 = TT.Layer0_param.b.get_value(borrow=True)
        W1 = TT.Layer1_param.W.get_value(borrow=True)
        b1 = TT.Layer1_param.b.get_value(borrow=True)
        W2 = TT.Layer2_param.W.get_value(borrow=True)
        b2 = TT.Layer2_param.b.get_value(borrow=True)
    if TR != None:
        W3 = TR.Layer3_param.W.get_value(borrow=True)
        b3 = TR.Layer3_param.b.get_value(borrow=True)
        W4 = TR.Layer4_param.W.get_value(borrow=True)
        b4 = TR.Layer4_param.b.get_value(borrow=True)
        W5 = TR.Layer5_param.W.get_value(borrow=True)
        b5 = TR.Layer5_param.b.get_value(borrow=True)


    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, dim, 128, 128),
            filter_shape=(nkerns[0], dim, 5, 5), poolsize=(2, 2),Wi=W0,bi=b0)

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)



    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], 62, 62),
            filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2),Wi=W1,bi=b1)

    layer12 = LeNetConvPoolLayer(rng, input=layer1.output,
            image_shape=(batch_size, nkerns[1], 29, 29),
            filter_shape=(nkerns[2], nkerns[1], 6, 6), poolsize=(2, 2),Wi=W2,bi=b2)

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer12.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[2] * 12 * 12,
                         n_out=1024,Wi=W3,bi=b3)
    layer3 = HiddenLayer(rng, input=layer2.output, n_in=1024,
                        n_out=2048,Wi=W4,bi=b4)
    layer4 = HiddenLayer(rng, input=layer3.output, n_in=2048,
                        n_out=51,Wi=W5,bi=b5)
    

    return [layer0, layer1, layer12, layer2, layer3, layer4];

def evaluate_lenet5(learning_rate=0.01, n_epochs=1,
                    pathDataset='path',
                    nameDataset='nameDataset',
                    nkerns=[10, 50, 500], batch_size=2, TT=None,TR=None):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    #dim = 1;
    #datasets = load_data(dataset)


    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y1 = T.matrix('y1')  # the labels are presented as 1D vector of
    y2 = T.matrix('y2')                    # [int] labels
    
    nameDataset=['Walking/','WalkingDog/','Eating/','Discussion/','TakingPhoto/','Greeting/']
    # useCustom = True;
    # if (useCustom):
    dim = 1;
    rang = 162000;
    #rang = 641;
    #mat = scipy.io.loadmat('data_'+nameDataset+'.mat')  #old .mat
    #mat = h5py.File(pathDataset+'data_'+nameDataset+'.mat','r');
    #a = numpy.concatenate((mat['train_x'],mat['validation_x'],mat['test_x']))
    #mat2 = h5py.File(pathDataset+'x_matrix.mat','r');
    #mat3 = h5py.File(pathDataset+'y_matrix.mat','r');
    #mat4 = h5py.File(pathDataset+'y_matrix_3D.mat','r');
    for i in range(0,1):
    #mat1 = h5py.File(pathDataset+'data_'+nameDataset+'.mat') 
        mat11 = h5py.File(pathDataset+nameDataset[i]+'data_36MS1.mat')
        mat12 = h5py.File(pathDataset+nameDataset[i]+'data_36MS5.mat')
        mat13 = h5py.File(pathDataset+nameDataset[i]+'data_36MS6.mat')
        mat14 = h5py.File(pathDataset+nameDataset[i]+'data_36MS7.mat')
        mat15 = h5py.File(pathDataset+nameDataset[i]+'data_36MS8.mat')
        mat16 = h5py.File(pathDataset+nameDataset[i]+'data_36MS9.mat')
        mat17 = h5py.File(pathDataset+nameDataset[i]+'data_36MS11.mat')
    
        mat2DPose11 = h5py.File(pathDataset+nameDataset[i]+'pose_2DS1.mat')
        mat2DPose12 = h5py.File(pathDataset+nameDataset[i]+'pose_2DS5.mat')
        mat2DPose13 = h5py.File(pathDataset+nameDataset[i]+'pose_2DS6.mat')
        mat2DPose14 = h5py.File(pathDataset+nameDataset[i]+'pose_2DS7.mat')
        mat2DPose15 = h5py.File(pathDataset+nameDataset[i]+'pose_2DS8.mat')
        mat2DPose16 = h5py.File(pathDataset+nameDataset[i]+'pose_2DS9.mat')
        mat2DPose17 = h5py.File(pathDataset+nameDataset[i]+'pose_2DS11.mat')
    
        matPose11 = h5py.File(pathDataset+nameDataset[i]+'pose_3DS1.mat')
        matPose12 = h5py.File(pathDataset+nameDataset[i]+'pose_3DS5.mat')
        matPose13 = h5py.File(pathDataset+nameDataset[i]+'pose_3DS6.mat')
        matPose14 = h5py.File(pathDataset+nameDataset[i]+'pose_3DS7.mat')
        matPose15 = h5py.File(pathDataset+nameDataset[i]+'pose_3DS8.mat')
        matPose16 = h5py.File(pathDataset+nameDataset[i]+'pose_3DS9.mat')
        matPose17 = h5py.File(pathDataset+nameDataset[i]+'pose_3DS11.mat')
        if(i==0):
            a = numpy.concatenate((mat11['datafinal'],mat12['datafinal'],mat13['datafinal'],mat14['datafinal'],mat15['datafinal'],mat16['datafinal'],mat17['datafinal']))
            b = numpy.concatenate((matPose11['posedata'],matPose12['posedata'],matPose13['posedata'],matPose14['posedata'],matPose15['posedata'],matPose16['posedata'],matPose17['posedata']))
            ab = numpy.concatenate((mat2DPose11['y_vector_new'],mat2DPose12['y_vector_new'],mat2DPose13['y_vector_new'],mat2DPose14['y_vector_new'],mat2DPose15['y_vector_new'],mat2DPose16['y_vector_new'],mat2DPose17['y_vector_new']))
        else:
            a = numpy.concatenate((a,mat11['datafinal'],mat12['datafinal'],mat13['datafinal'],mat14['datafinal'],mat15['datafinal'],mat16['datafinal'],mat17['datafinal']))
            b= numpy.concatenate((b,matPose11['posedata'],matPose12['posedata'],matPose13['posedata'],matPose14['posedata'],matPose15['posedata'],matPose16['posedata'],matPose17['posedata']))
            ab = numpy.concatenate((ab,mat2DPose11['y_vector_new'],mat2DPose12['y_vector_new'],mat2DPose13['y_vector_new'],mat2DPose14['y_vector_new'],mat2DPose15['y_vector_new'],mat2DPose16['y_vector_new'],mat2DPose17['y_vector_new']))
    #a = (mat2['x_vector'])
    #ab = (mat3['y_vector'])
    
    #b = numpy.concatenate((mat['train_y'],mat['validation_y'],mat['test_y']))
    #b = (mat4['y_vector_3D'])
    a = a[1:rang,:]
    ab = ab[1:rang,:]
    b = b[1:rang,:]
    joints = [1, 2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]
    
    for gh in reversed(range(0,96)):
        if((gh+1)%3==0):
            b[1:rang,gh] = b[1:rang,gh]-b[1:rang,2]
        if((gh+1)%3==1):
            b[1:rang,gh] = b[1:rang,gh]-b[1:rang,0]
        if((gh+1)%3==2):
            b[1:rang,gh] = b[1:rang,gh]-b[1:rang,1]
    print ab.shape
    
    c = []
    for j in joints:
        i=j-1;
        if(i==0):
            c= b[:,i*3:i*3+3]
        else:
            c = numpy.append(c,b[:,i*3:i*3+3],1)
    print c.shape
        
    '''for g in range(0,1):
        a = numpy.append(a,a,0)
        b = numpy.concatenate((b,b))'''
    print(a.shape)
    #print(b[2,:])
    #tx = numpy.asarray(numpy.concatenate((mat['train_x'],mat['validation_x'],mat['test_x'],mat['train_x'],mat['validation_x'],mat['test_x'])),dtype=theano.config.floatX);
    tx = numpy.asarray(a[0:141999,:],dtype=theano.config.floatX); #2819
    train_set_x =  theano.shared(tx,borrow=True);
    #ty = numpy.asarray(numpy.concatenate((mat['train_y'],mat['validation_y'],mat['test_y'],mat['train_y'],mat['validation_y'],mat['test_y'])),dtype=theano.config.floatX);
    ty1 = numpy.asarray(ab[0:141999,:],dtype=theano.config.floatX);
    train_set_y1 =  theano.shared(ty1,borrow=True);
    ty = numpy.asarray(c[0:141999,:],dtype=theano.config.floatX);
    train_set_y2 =  theano.shared(ty,borrow=True);
    #train_set_y = theano.shared(numpy.reshape(ty,ty.shape[0]),borrow=True);#mat['train_y'];
    #train_set_y =  theano.shared(numpy.squeeze(ty),borrow=True);
    #train_set_y = theano.shared(numpy.reshape(ty,(ty.shape[0],ty.shape[1])),borrow=True);#mat['train_y'];
    #train_set_y = theano.shared(numpy.reshape(ty,ty.shape[0]*ty.shape[1]),borrow=True);#mat
    #vx = numpy.asarray(mat['validation_x'],dtype=theano.config.floatX);
    vx = numpy.asarray(a[122000:141999,:],dtype=theano.config.floatX);#2819:3315
    valid_set_x =  theano.shared(vx,borrow=True);
    vy = numpy.asarray(c[122000:141999,:],dtype=theano.config.floatX);
    valid_set_y2 =  theano.shared(vy,borrow=True);
    #valid_set_y =  theano.shared(numpy.reshape(vy,vy.shape[0]),borrow=True);
    #valid_set_y =  theano.shared(numpy.squeeze(vy),borrow=True);
    #valid_set_y =  theano.shared(numpy.reshape(vy,(vy.shape[0],vy.shape[1])),borrow=True);#mat['train_y'];
    #valid_set_y =  theano.shared(numpy.reshape(vy,vy.shape[0]*vy.shape[1]),borrow=True);#mat
    ttx = numpy.asarray(a[142000:161899,:],dtype=theano.config.floatX); #3316:3851
    test_set_x =  theano.shared(ttx,borrow=True);
    vvy = numpy.asarray(c[142000:161899,:],dtype=theano.config.floatX);
    test_set_y2 =  theano.shared(vvy,borrow=True);
    #test_set_y = theano.shared(numpy.reshape(vvy,vvy.shape[0]),borrow=True);#mat['train_y'];
    #test_set_y =  theano.shared(numpy.squeeze(vvy),borrow=True);
    #test_set_y = theano.shared(numpy.reshape(vvy,(vvy.shape[0],vvy.shape[1])),borrow=True);#mat
    #test_set_y = theano.shared(numpy.reshape(vvy,vvy.shape[0]*vvy.shape[1]),borrow=True);#mat['train_y']; # n_train_batches = train_set_x.shape[0]
        #n_valid_batches = valid_set_x.shape[0]
        #n_test_batches = test_set_x.shape[0]
    # else:
    #     dim = 1;
    #     datasets = load_data(dataset)
    #     train_set_x, train_set_y = datasets[0]
    #     valid_set_x, valid_set_y = datasets[1]
    #     test_set_x, test_set_y = datasets[2]


    # datasets2 = load_data(dataset)
    # train_set_x2, train_set_y2 = datasets2[0]
    # valid_set_x2, valid_set_y2 = datasets2[1]
    # test_set_x2, test_set_y2 = datasets2[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]



    # compute number of minibatches for training, validation and testing

    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches  /= batch_size



    #ishape = (28, 28)  # this is the size of MNIST images

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

     # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, dim, 128, 128))

    [ layerD0, layerD1, layerD2, layerD3, layerD4, layerD5, TTD] = buildLayersD(layer0_input,batch_size, dim, nkerns,rng,TT);
    
    file = open('resultDetectionWalking_50samp_5_10_15.dat','rb')
    par = cPickle.load(file)
    #par=None
    #if(par==None):
    #    print 'par is None'

    [ layerR0, layerR1, layerR2, layerR3, layerR4, layerR5] = buildLayersR(layer0_input,batch_size, dim, nkerns,rng,TTD,TR,par);
    
    # the cost we minimize during training is the NLL of the model
    #cost = layer4.negative_log_likelihood(y)
    costD = layerD5.newerrorsD(y1)
    costDlast = layerD5.newerrorsDlast(y1,layerD0,layerD1,layerD2)
    costR = layerR5.newerrors(y2,layerR0,layerR1,layerR2,batch_size)
    #cost = T.mean(0.5*(y.reshape((batch_size,1,96,1)) - 64 / (1 + T.exp(-T.dot(layer4.output.flatten(2), layer4.W) - layer1.b)))**2)
    #print(cost)

    # create a function to compute the mistakes that are made by the model
    '''test_modelD = theano.function([index], layerD5.Nerrors(y1),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y1: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_modelD = theano.function([index], layerD5.Nerrors(y1),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y1: valid_set_y[index * batch_size: (index + 1) * batch_size]})
    
    train_modelD = theano.function([index], layerD5.Nerrors(y1),
             givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y1: train_set_y[index * batch_size: (index + 1) * batch_size]})'''
    
    
    
    test_modelR = theano.function([index], layerR5.Nerrors(y2,batch_size),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y2: test_set_y2[index * batch_size: (index + 1) * batch_size]})

    validate_modelR = theano.function([index], layerR5.Nerrors(y2,batch_size),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y2: valid_set_y2[index * batch_size: (index + 1) * batch_size]})

    # create a list of all model parameters to be fit by gradient descent
    paramsD = layerD5.params + layerD4.params + layerD3.params + layerD2.params + layerD1.params + layerD0.params
    paramsR = layerR5.params + layerR4.params + layerR3.params + layerR2.params + layerR1.params + layerR0.params
    #params = layer3.params + layer2.params + layer1.params + layer0.params


    # create a list of gradients for all model parameters
    gradsD = T.grad(costD, paramsD)
    gradsDlast = T.grad(costDlast[0], paramsD)
    gradsR = T.grad(costR[0], paramsR)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    #momentum =0.1
    #updates = gradient_updates_momentum(cost, params, learning_rate, momentum)
    updatesD = []
    for param_i, grad_i in zip(paramsD, gradsD):
        updatesD.append((param_i, param_i - learning_rate * grad_i))
    
    updatesDlast = []
    for param_i, grad_i in zip(paramsD, gradsDlast):
        updatesDlast.append((param_i, param_i - learning_rate * grad_i))

    updatesR = []
    momentum=0.9
    #for param_i, grad_i in zip(paramsR, gradsR):
    #    updatesR.append((param_i, param_i - learning_rate * grad_i))
    weight_decay=0.0005
    for param_i, grad_i in zip(paramsR, gradsR):
        param_update = theano.shared(param_i.get_value()*0., broadcastable=param_i.broadcastable)
        updatesR.append((param_i, param_i + param_update))
        updatesR.append((param_update, momentum*param_update -learning_rate *grad_i-weight_decay*learning_rate*param_i))

    train_modelD = theano.function([index], costD, updates=updatesD,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y1: train_set_y1[index * batch_size: (index + 1) * batch_size]})
    train_modelDlast = theano.function([index], costDlast, updates=updatesDlast,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y1: train_set_y1[index * batch_size: (index + 1) * batch_size]})


    '''model_probD = theano.function([index], layerD5.err,
             givens={
                #x: test_set_x[index * batch_size: (index + 1) * batch_size]})
                x: valid_set_x[index * batch_size: (index + 1) * batch_size]})
    model_yD = theano.function([index], y1,
             givens={
                #y: test_set_y[index * batch_size: (index + 1) * batch_size]})
                y1: valid_set_y[index * batch_size: (index + 1) * batch_size]})'''


    train_modelR = theano.function([index], costR, updates=updatesR,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y2: train_set_y2[index * batch_size: (index + 1) * batch_size]})

    model_probR = theano.function([index], layerR5.err,
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size]})
    model_yR = theano.function([index], y2,
             givens={
                y2: test_set_y2[index * batch_size: (index + 1) * batch_size]})
    model_probvalidR = theano.function([index], layerR5.err,
             givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size]})
    model_yvalidR = theano.function([index], y2,
             givens={
                y2: valid_set_y2[index * batch_size: (index + 1) * batch_size]})

    # what you want: create a function to predict labels that are made by the model
   # model_predict = theano.function([index], layer4.y_pred,
   #          givens={
    #            x: test_set_x[index * batch_size: (index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 100000000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_train_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = True
    while (epoch <50) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index
            #print(minibatch_index)

            if iter % 1000 == 0:
                print 'trainingD @ iter = ', iter
            cost_ij = train_modelD(minibatch_index)
    
            if (iter + 1) % validation_frequency == 0:

                train_losses = [train_modelD(i) for i
                                in xrange(n_train_batches)]
                #train_losses = [normWeights[i][0] for i
                #                in xrange(n_train_batches)]
                normWeights = train_modelDlast(n_train_batches-1)
                print train_losses
                this_train_loss = numpy.mean(train_losses)
                print('epoch %i, minibatch %i/%i, Train error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       (this_train_loss)/17))
                if this_train_loss < 0.1:
                #    best_train_loss = this_train_loss
                    done_looping = True
    
    '''weight0 = normWeights[1]
    bias0=normWeights[2]
    weight1 = normWeights[3]
    bias1=normWeights[4]
    weight2 = normWeights[5]
    bias2=normWeights[6]
    #print weight0
    param=[weight0,bias0,weight1,bias1,weight2,bias2]
    save_file = open('resultDetectionWalking_50samp_5_10_15.dat', 'wb')
    cPickle.dump(param, save_file, protocol=cPickle.HIGHEST_PROTOCOL)
    save_file.close()'''
    
    print('Let us use the weights of Detection for Regression')
    epoch = 0    
    done_looping = False
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index
            #print(minibatch_index)

            if iter % 500 == 0:
                print 'trainingR @ iter = ', iter
            cost_ijR = train_modelR(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                normtrainweights = [train_modelR(i) for i
                                in xrange(n_train_batches)]
                train_losses = [normtrainweights[i][0] for i
                                in xrange(n_train_batches)]
                this_train_loss = numpy.mean(train_losses)
                print('epoch %i, minibatch %i/%i, Train error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       (this_train_loss)))
		weight0 = normtrainweights[n_train_batches-1][1]
                bias0=normtrainweights[n_train_batches-1][2]
                weight1 = normtrainweights[n_train_batches-1][3]
                bias1=normtrainweights[n_train_batches-1][4]
                weight2 = normtrainweights[n_train_batches-1][5]
                bias2=normtrainweights[n_train_batches-1][6]
                param=[weight0,bias0,weight1,bias1,weight2,bias2]
                save_file = open('./Regression/resultRegressionWalking50samp_batch128_nosqrt_mom_net3_wt.dat', 'wb')
                cPickle.dump(param, save_file, protocol=cPickle.HIGHEST_PROTOCOL)
                save_file.close()
                
                # compute zero-one loss on validation set
                validation_losses = [validate_modelR(i) for i
                                     in xrange(n_valid_batches)]

                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_valid_batches, \
                       this_validation_loss))
		with open("./Regression/OutputValid1Pbatch128_nosqrt_mom_net3_wt.txt","a") as text_file:
                    text_file.write("\n")
                    text_file.write("{0}".format(epoch))
                    text_file.write("\n\n")
                with open("./Regression/OutputValid1Gbatch128_nosqrt_mom_net3_wt.txt","a") as text_file:
                    text_file.write("\n")
                    text_file.write("{0}".format(epoch))
                    text_file.write("\n\n")
                gen1 = (x1 for x1 in xrange(n_valid_batches) if x1%100==0)
                for i in gen1:
                    a = numpy.array(model_probvalidR(i)).tolist()
                    c = numpy.array(model_yvalidR(i)).tolist()
                    with open("./Regression/OutputValid1Pbatch128_nosqrt_mom_net3_wt.txt","a") as text_file:
                        text_file.write("{0}".format(a))
                        text_file.write("\n")
                    with open("./Regression/OutputValid1Gbatch128_nosqrt_mom_net3_wt.txt","a") as text_file:
                        text_file.write("{0}".format(c))
                        text_file.write("\n\n")
                text_file.close()
                
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    
                        

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    #print [model_prob(i) for i in xrange(n_test_batches)]
                    
                    #print [model_predict(i) for i in xrange(n_test_batches)]
                    #pathDataset+'data_'+ nameDataset+'.mat'
                    #a = numpy.array(model_prob(numpy.argmin(validation_losses))).tolist()
                    with open("./Regression/OutputTest1Pbatch128_nosqrt_mom_net3_wt.txt","a") as text_file:
                        text_file.write("\n")
                        text_file.write("{0}".format(epoch))
                        text_file.write("\n\n")
                    with open("./Regression/OutputTest1Gbatch128_nosqrt_mom_net3_wt.txt","a") as text_file:
                        text_file.write("\n")
                        text_file.write("{0}".format(epoch))
                        text_file.write("\n\n")
                    gen1 = (x1 for x1 in xrange(n_test_batches) if x1%100==0)
                    for i in gen1:
                        a = numpy.array(model_probR(i)).tolist()
                        c = numpy.array(model_yR(i)).tolist()
                        with open("./Regression/OutputTest1Pbatch128_nosqrt_mom_net3_wt.txt","a") as text_file:
                            text_file.write("{0}".format(a))
                            text_file.write("\n")
                        with open("./Regression/OutputTest1Gbatch128_nosqrt_mom_net3_wt.txt","a") as text_file:
                            text_file.write("{0}".format(c))
                            text_file.write("\n\n")
                    
                    print("Wrote in the file")
                    
                    #cPickle.dump(layer0, save_file, protocol=cPickle.HIGHEST_PROTOCOL)
                    #cPickle.dump(layer1, save_file, protocol=cPickle.HIGHEST_PROTOCOL)
                    #cPickle.dump(layer2, save_file, protocol=cPickle.HIGHEST_PROTOCOL)
                    #cPickle.dump(layer3, save_file, protocol=cPickle.HIGHEST_PROTOCOL)
                    #cPickle.dump(layer4, save_file, protocol=cPickle.HIGHEST_PROTOCOL)
                    #save_file.close()
                    #h5f = h5py.File('data.h5', 'w');
                    #h5f.create_dataset('dataset_1', data=T.vector(layer4.err,dmatrix));
                    #h5f.close();
                    
                    text_file.close()

                    # test it on the test set
                    test_losses = [test_modelR(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('epoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') %
                          (epoch, minibatch_index + 1, n_test_batches,
                           test_score ))

            if patience <= iter:
                done_looping = False
                #break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))





def gradient_updates_momentum(cost, params, learning_rate, momentum):
    updates = []
    grads = T.grad(cost, params)
    for param_i in params:
        param_update = theano.shared(param_i.get_value()*0., broadcastable=param_i.broadcastable)
        updates.append((param_i, param_i - learning_rate * param_update))
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param_i)))
    return updates

#data_<name>.mat
if __name__ == '__main__':
    learning_rate=0.001
    n_epochs=100
    pathDataset = './'
    nameDataset=''  #no .mat
    nkerns=[5, 10, 15]
    batch_size=128

    default = False;
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"n:p:",["nfile=","pfile="])
    except getopt.GetoptError:
        print 'use default parameter, otherwise -n <name Dataset> -p <path Dataset>'
        default = True;

    if not default:
        for opt, arg in opts:
            if opt in ("-n", "--nfile"):
                nameDataset = arg
            elif opt in ("-p", "--pfile"):
                pathDataset = arg




    print "Train for: "+nameDataset

    evaluate_lenet5(learning_rate, n_epochs, pathDataset, nameDataset,nkerns, batch_size)


