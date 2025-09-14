"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    
    # According to the file format description on the website
    # TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    # [offset] [type]          [value]          [description]
    # 0000     32 bit integer  0x00000803(2051) magic number
    # 0004     32 bit integer  60000            number of images
    # 0008     32 bit integer  28               number of rows
    # 0012     32 bit integer  28               number of columns
    # 0016     unsigned byte   ??               pixel
    # 0017     unsigned byte   ??               pixel
    # ........
    # xxxx     unsigned byte   ??               pixel
    
    # Read the image file
    with gzip.open(image_filename, 'rb') as f:
        # Useless
        magic_number = struct.unpack('>I', f.read(4))[0]
        n_images = struct.unpack('>I', f.read(4))[0]
        n_rows = struct.unpack('>I', f.read(4))[0]
        n_cols = struct.unpack('>I', f.read(4))[0]
        X = np.frombuffer(f.read(), dtype=np.uint8).reshape(n_images, n_rows * n_cols)
        X = X.astype(np.float32) / 255.0
        
    # According to the file format description on the website
    # TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    # [offset] [type]          [value]          [description]
    # 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    # 0004     32 bit integer  60000            number of items
    # 0008     unsigned byte   ??               label
    # 0009     unsigned byte   ??               label
    # ........
    # xxxx     unsigned byte   ??               label
    # The labels values are 0 to 9.

    # Read the label file
    with gzip.open(label_filename, 'rb') as f:
        # Useless
        magic_number = struct.unpack('>I', f.read(4))[0]
        n_labels = struct.unpack('>I', f.read(4))[0]
        
        y = np.frombuffer(f.read(), dtype=np.uint8)
        y = y.astype(np.uint8)

    return X, y
    ### END YOUR CODE


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    return ndl.summation(ndl.log(ndl.summation(ndl.exp(Z), axes=(1,))) - ndl.summation(Z * y_one_hot, axes=(1,))) / Z.shape[0]
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
