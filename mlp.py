
__docformat__ = 'restructedtext en'


import cPickle
import gzip
import os
import sys
import time
import math
import numpy

import theano
import theano.tensor as T


from logistic_sgd import LogisticRegression, load_data
class ReturnValue(object):
  def __init__(self,norm,W0,b0,W1,b1,W2,b2):
     self.norm=norm
     self.W0 = W0
     self.b0 = b0
     self.W1 = W1
     self.b1 = b1
     self.W2 = W2
     self.b2 = b2

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, Wi=None, bi=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if Wi is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)
        else:
            W_values = numpy.asarray(Wi, dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if bi is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        else:
            b_values = numpy.asarray(bi, dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]
        self.err = T.dot(input, self.W) + self.b
    
    def newerrorsD(self,y):
        q = self.err
        norm = 0
        #epsi = numpy.finfo(numpy.float32).eps
        #i=1699
        #for i in range(0,1700,1):
        #    norm =norm+(-1* y[0,i]*T.log(1/(1+T.exp(-q[0,i]))) - (1-y[0,i])*T.log(1-(1/(1+T.exp(-q[0,i])))))
        #if(q[0,i]==1):
        #    norm = -1* y[0,i]*T.log(epsi+q[0,i]) - (1-y[0,i])*T.log(1-q[0,i]);
        #else:
        #norm =  - (1-y[0,i])*T.log(1-(1/(1+T.exp(-q[0,i])))) - y[0,i]*T.log(1/(1+T.exp(-q[0,i])))
        norm =T.sum( - y[0,:]*T.log(1/(1+T.exp(-q[0,:]))) - (1-y[0,:])*T.log(1-(1/(1+T.exp(-q[0,:])))))
            #m = y[0,i:i+100]
            #n = q[0,i:i+100]
            
            #for j in range(0,100,1):
            #    print i+j
                #norm = norm - m[0,j]*math.log(n[0,j]) - (1-n[0,j])*math.log(m[0,j])
            #    norm = - y[0,i+j]*T.log(q[0,i+j]) - (1-q[0,i+j])*T.log(y[0,i+j])
        #norm = T.sum(norm1)
        #return ReturnValue(norm,W0,b0,W1,b1,W2,b2)
        #return [norm, W0,b0,W1,b1,W2,b2]
        return norm
    
    def newerrorsDlast(self,y,D0,D1,D2):
        q = self.err
        W0=D0.W
        b0=D0.b
        W1=D1.W
        b1=D1.b
        W2=D2.W
        b2=D2.b
        norm = 0
        norm =T.sum( - y[0,:]*T.log(1/(1+T.exp(-q[0,:]))) - (1-y[0,:])*T.log(1-(1/(1+T.exp(-q[0,:])))))
        return [norm, W0,b0,W1,b1,W2,b2]
        
    def newerrors(self,y,D0,D1,D2,batch_size):
        #if y.dtype.startswith('float'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            #return T.mean(T.neq(self.y_pred, y))
        '''y_r = y
        for i in range(0,95):
            if((i+1)%3!=0):
                y_r[i] = y[i]*10
            else:
                y_r[i] = y[i]
        o = T.as_tensor_variable([y_r])
        f = theano.function([y],o)'''
        #return T.mean(0.5*(y- self.err)**2)
        #p =y
        q = self.err
        W0=D0.W
        b0=D0.b
        W1=D1.W
        b1=D1.b
        W2=D2.W
        b2=D2.b
        #norm = T.lscalar() 
        norm=0
        '''index = T.lscalar() 
        y_l = T.dmatrix('y_l')
        q_l = T.dmatrix('q_l')
        norm = T.sqrt(T.sum((y_l-q_l)**2))
        func = theano.function([index],norm,
                givens={
                    q_l: q[index * 3: (index + 1) * 3],
                    y_l: p[index * 3: (index + 1) * 3]})'''
        #y_split = theano.function([index],y_l,givens={y_l: y[index * 3: (index + 1) * 3]})
        #for j in range(0,10,1):
        for i in range(0,50,3):
            norm = norm + T.sum(T.sqrt((y[:,i] - q[:,i])**2+ (y[:,i+1] - q[:,i+1])**2 + (y[:,i+2] - q[:,i+2])**2))#T.sum((y[1,i:i+3] - q[1,i:i+3]).norm(L=2))#T.sqrt(T.sum((y[0,i:i+2] - q[0,i:i+2])**2))#+ (y[0,i+1] - q[0,i+1])**2 + (y[0,i+2] - q[0,i+2])**2
        #print y.dtype
        #losses = [func(i) for i
        #                in range(0,95,3)]
        #this_train_loss = numpy.mean(train_losses)
        #i = 0
        '''results, updates = theano.scan(lambda y_i, q_i: T.sqrt(((y_i-q_i) ** 2).sum()), sequences=[y.T,q.T])
        compute_norm_cols = theano.function(inputs=[y,q], outputs=[results])'''
        #return T.sqrt(T.sum((y - q)**2))
        #return numpy.mean(losses)
        return [norm/(17*batch_size), W0,b0,W1,b1,W2,b2]
        #return T.sum((y[:,[0,2]] - q[:,[0,2]]).norm(L=2))
        #return T.sum(([(y[i,i+2] - q[i:i+2]).norm(L=2) for i in range(0,95,3)]))
        #return compute_norm_cols(y,q)
        #return T.mean(([T.sqrt(T.sum((y[i:i+2] - q[i:i+2])**2)) for i in range(0,95,3)]))
        #return T.mean(abs((y.reshape((batch_size,1,96,1)) - 64 / (1 + T.exp(-T.dot(self.output.flatten(2), self.W) - self.b)))))

    
    def Nerrors(self,y,batch_size):
    #    return T.numpy.sqrt(T.mean(abs(y - self.err)))
        m = self.err
        norm1 = 0;
        for i in range(0,50,3):
            norm1 = norm1 + T.sum(T.sqrt((y[:,i] -m[:,i])**2+ (y[:,i+1] - m[:,i+1])**2 + (y[:,i+2] - m[:,i+2])**2))#(y[0,j:j+3] - m[0,j:j+3]).norm(L=2)
        return (norm1/(17*batch_size))
        #return T.sqrt(T.sum((y - self.err)**2))
     
    def mapfunc(self):
        return T.dot(self,10)
        

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(rng=rng, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.hiddenLayer.W).sum() \
                + abs(self.logRegressionLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
                    + (self.logRegressionLayer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(rng=rng, input=x, n_in=28 * 28,
                     n_hidden=n_hidden, n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = classifier.negative_log_likelihood(y) \
         + L1_reg * classifier.L1 \
         + L2_reg * classifier.L2_sqr

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})

    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = []
    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    for param, gparam in zip(classifier.params, gparams):
        updates.append((param, param - learning_rate * gparam))

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index], outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
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
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                     (epoch, minibatch_index + 1, n_train_batches,
                      this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                    done_looping = True
                    break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    test_mlp()
