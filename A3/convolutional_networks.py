"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from a3_helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, adam, ReLU


def hello_convolutional_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modfiy the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the convolutional forward pass.                  #
        # Hint: you can use function torch.nn.functional.pad for padding.  #
        # You are NOT allowed to use anything in torch.nn in other places. #
        ####################################################################
        pad, stride = conv_param["pad"], conv_param["stride"]

        x_padded = torch.nn.functional.pad(
            x,
            pad=(pad,) * 4,
            mode="constant",
            value=0
        )

        # First attempt at convolving: nested loops
        H_in, W_in = x.shape[-2:]
        H_filter, W_filter = w.shape[-2:]
        H_prime = int(1 + (H_in + 2 * pad - H_filter) / stride)
        W_prime = int(1 + (W_in + 2 * pad - W_filter) / stride)

        out = x.new_empty((x.shape[0], w.shape[0], H_prime, W_prime))
        for h in range(x.shape[0]): # Looping over samples in batch
            for i in range(w.shape[0]): # Looping over filters
                for j in range(H_prime): # Vertical scan
                    for k in range(W_prime): # Horizontal scan
                        H_o = j * stride # Vertical offset
                        W_o = k * stride # Horizontal offset

                        out[h, i, j, k] = \
                            (x_padded[h, :, H_o:H_o + H_filter, W_o:W_o + W_filter] * \
                                w[i]).sum() + b[i]

        #####################################################################
        #                          END OF YOUR CODE                         #
        #####################################################################
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None
        ###############################################################
        # TODO: Implement the convolutional backward pass.            #
        ###############################################################
        x, w, b, conv_param = cache

        pad, stride = conv_param["pad"], conv_param["stride"]

        x_padded = torch.nn.functional.pad(
            x,
            pad=(pad,) * 4,
            mode="constant",
            value=0
        )

        # First attempt at convolving: nested loops
        H_in, W_in = x.shape[-2:]
        H_filter, W_filter = w.shape[-2:]
        H_prime = int(1 + (H_in + 2 * pad - H_filter) / stride)
        W_prime = int(1 + (W_in + 2 * pad - W_filter) / stride)

        # out = x.new_empty((x.shape[0], w.shape[0], H_prime, W_prime))
        dw = torch.zeros_like(w)
        dx_padded = torch.zeros_like(x_padded)
        for h in range(x.shape[0]): # Looping over samples in batch
            for i in range(w.shape[0]): # Looping over filters
                for j in range(H_prime): # Vertical scan
                    for k in range(W_prime): # Horizontal scan
                        H_o = j * stride # Vertical offset
                        W_o = k * stride # Horizontal offset

                        # One one to understand this is that we are accumulating
                        # the input "patches" that the filter scanned over when
                        # convolving over the input (additionally multiplying
                        # with the upstream gradient)
                        dw[i] += dout[h, i, j, k] * \
                            x_padded[h, :, H_o:H_o + H_filter, W_o:W_o + W_filter]

                        # One way to understand this is that we are accumulating,
                        # for each "patch" of the input, the filter weights that
                        # scanned over that "patch" (and additionally multiplying
                        # with the upstream gradient)
                        dx_padded[h, :, H_o:H_o + H_filter, W_o:W_o + W_filter] \
                            += dout[h, i, j, k] * w[i]

        # "Unpadding" the input
        dx = dx_padded[:, :, pad:pad + x.shape[-2], pad:pad + x.shape[-1]]

        # Gradient of the bias
        db = dout.sum(dim=[0, -1, -2])
        ###############################################################
        #                       END OF YOUR CODE                      #
        ###############################################################
        return dx, dw, db


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the max-pooling forward pass                     #
        ####################################################################
        height = pool_param["pool_height"]
        width = pool_param["pool_width"]
        stride = pool_param["stride"]

        H, W = x.shape[-2:]
        H_out = int(1 + (H - height) / stride)
        W_out = int(1 + (W - width) / stride)

        out = x.new_zeros((*x.shape[:2], H_out, W_out))
        for j in range(H_out): # Vertical scan
            for k in range(W_out): # Horizontal scan
                H_origin = j * stride # Vertical offset
                W_origin = k * stride # Horizontal offset

                H_slice = slice(H_origin, H_origin + height)
                W_slice = slice(W_origin, W_origin + width)


                out[:, :, j, k] = x[:, :, H_slice, W_slice].amax(dim=(-2, -1))
        ####################################################################
        #                         END OF YOUR CODE                         #
        ####################################################################
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        #####################################################################
        # TODO: Implement the max-pooling backward pass                     #
        #####################################################################
        x, pool_param = cache

        height = pool_param["pool_height"]
        width = pool_param["pool_width"]
        stride = pool_param["stride"]

        dx = torch.zeros_like(x)
        for j in range(dout.shape[-2]): # Vertical scan
            for k in range(dout.shape[-1]): # Horizontal scan
                H_origin = j * stride # Vertical offset
                W_origin = k * stride # Horizontal offset

                H_slice = slice(H_origin, H_origin + height)
                W_slice = slice(W_origin, W_origin + width)

                x_slice = x[:, :, H_slice, W_slice]
                maxes = x_slice.amax(dim=(-2, -1), keepdim=True)

                # This mask allows "routing" upstream gradients to the correct 
                # positions (the ones containing the max values)
                gradient_mask = x_slice == maxes

                dx[:, :, H_slice, W_slice] += gradient_mask * dout[:, :, j, k] \
                    .unsqueeze(dim=-1).unsqueeze(dim=-1)
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################
        return dx


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ######################################################################
        # TODO: Initialize weights，biases for the three-layer convolutional #
        # network. Weights should be initialized from a Gaussian             #
        # centered at 0.0 with standard deviation equal to weight_scale;     #
        # biases should be initialized to zero. All weights and biases       #
        # should be stored in thedictionary self.params. Store weights and   #
        # biases for the convolutional layer using the keys 'W1' and 'b1';   #
        # use keys 'W2' and 'b2' for the weights and biases of the hidden    #
        # linear layer, and key 'W3' and 'b3' for the weights and biases of  #
        # the output linear layer                                            #
        #                                                                    #
        # IMPORTANT: For this assignment, you can assume that the padding    #
        # and stride of the first convolutional layer are chosen so that     #
        # **the width and height of the input are preserved**. Take a        #
        # look at the start of the loss() function to see how that happens.  #
        ######################################################################
        C, H, W = input_dims
        
        weights = {
            "W1": weight_scale * torch.randn(
                (num_filters, C, *(filter_size,) * 2),
                device=device,
                dtype=dtype
            ),
            "W2": weight_scale * torch.randn(
                # Divide each spatial dimension by two due to the max pooling operation
                (torch.tensor([num_filters, H // 2, W // 2]).prod().item(), hidden_dim),
                device=device,
                dtype=dtype
            ),
            "W3": weight_scale * torch.randn(
                (hidden_dim, num_classes),
                device=device,
                dtype=dtype
            )
        }

        biases = {
            "b1": torch.zeros(num_filters, device=device, dtype=dtype),
            "b2": torch.zeros(hidden_dim, device=device, dtype=dtype),
            "b3": torch.zeros(num_classes, device=device, dtype=dtype)
        }

        self.params = {
            **weights,
            **biases
        }
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet.
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for three-layer convolutional     #
        # net, computing the class scores for X and storing them in the      #
        # scores variable.                                                   #
        #                                                                    #
        # Remember you can use functions defined in your implementation      #
        # above                                                              #
        ######################################################################
        h1, conv_cache = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
        h2, hidden_cache = Linear_ReLU.forward(h1, W2, b2)
        scores, out_cache = Linear.forward(h2, W3, b3)
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################

        if y is None:
            return scores

        loss, grads = 0.0, {}
        ####################################################################
        # TODO: Implement backward pass for three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables.  #
        # Compute data loss using softmax, and make sure that grads[k]     #
        # holds the gradients for self.params[k]. Don't forget to add      #
        # L2 regularization!                                               #
        #                                                                  #
        # NOTE: To ensure that your implementation matches ours and you    #
        # pass the automated tests, make sure that your L2 regularization  #
        # does not include a factor of 0.5                                 #
        ####################################################################
        loss, dscores = softmax_loss(scores, y)

        # Adding regulatization contribution to overall loss
        for W in (W1, W2, W3):
            loss += self.reg * torch.sum(W * W)

        # Computing parameter gradients
        dh2, dW3, db3 = Linear.backward(dscores, out_cache)
        dh1, dW2, db2 = Linear_ReLU.backward(dh2, hidden_cache)
        _, dW1, db1 = Conv_ReLU_Pool.backward(dh1, conv_cache)

        grads = {
            "W1": dW1 + 2 * self.reg * W1,
            "b1": db1,
            "W2": dW2 + 2 * self.reg * W2,
            "b2": db2,
            "W3": dW3 + 2 * self.reg * W3,
            "b3": db3
        }
        ###################################################################
        #                             END OF YOUR CODE                    #
        ###################################################################

        return loss, grads


class DeepConvNet(object):
    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:

    {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    Each {...} structure is a "macro layer" consisting of a convolution layer,
    an optional batch normalization layer, a ReLU nonlinearity, and an optional
    pooling layer. After L-1 such macro layers, a single fully-connected layer
    is used to predict the class scores.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 batchnorm=False,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 weight_initializer=None,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.

        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: List of length (L - 1) giving the number of
          convolutional filters to use in each macro layer.
        - max_pools: List of integers giving the indices of the macro
          layers that should have max pooling (zero-indexed).
        - batchnorm: Whether to include batch normalization in each macro layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights, or the string "kaiming" to use Kaiming
          initialization instead
        - reg: Scalar giving L2 regularization strength. L2 regularization
          should only be applied to convolutional and fully-connected weight
          matrices; it should not be applied to biases or to batchnorm scale
          and shifts.
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you should
          use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.num_layers = len(num_filters)+1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

        #####################################################################
        # TODO: Initialize the parameters for the DeepConvNet. All weights, #
        # biases, and batchnorm scale and shift parameters should be        #
        # stored in the dictionary self.params.                             #
        #                                                                   #
        # Weights for conv and fully-connected layers should be initialized #
        # according to weight_scale. Biases should be initialized to zero.  #
        # Batchnorm scale (gamma) and shift (beta) parameters should be     #
        # initilized to ones and zeros respectively.                        #
        #####################################################################
        def naive_initializer(
            Din, Dout, K=None, relu=True, device="cpu", dtype=torch.float32
        ):
            if K is None:
                shape = (Din, Dout)
            else:
                shape = (Dout, Din, K, K)

            return weight_scale * torch.randn(
                shape,
                device=device,
                dtype=dtype
            )

        # Select weight initializer function
        weight_init = kaiming_initializer if weight_scale == "kaiming" else naive_initializer
        
        conv_weights = {
            f"W{i+1}": weight_init(
                Din=in_depth,
                Dout=num_filter,
                K=3,
                relu=True,
                device=device,
                dtype=dtype
            )

            for i, (num_filter, in_depth) in enumerate(
                zip(num_filters, [input_dims[0]] + num_filters[:-1])
            )
        }

        # To instantiate the weights of the final linear layer, the dimensions
        # of the final feature map must be known. These are a function the
        # number filters in the last conv layer, the input spatial dimensions,
        # and the number of max pooling layers (because each halves the spatial
        # dimensions)
        n_max_pooling = len(max_pools)
        final_layer_inp_dim = num_filters[-1] \
            * (input_dims[1] // 2**n_max_pooling) \
            * (input_dims[2] // 2**n_max_pooling)

        weights = {
            **conv_weights,
            # f"W{self.num_layers}": weight_scale * torch.randn(
            #     (final_layer_inp_dim, num_classes),
            #     device=device,
            #     dtype=dtype
            # )

            f"W{self.num_layers}": weight_init(
                Din=final_layer_inp_dim,
                Dout=num_classes,
                K=None,
                relu=False,
                device=device,
                dtype=dtype
            )
        }

        biases = {
            f"b{i+1}": torch.zeros(s, device=device, dtype=dtype)

            for i, s in enumerate(num_filters + [num_classes])
        }

        # Parameters for the optional batch normalization layers
        if batchnorm:
            gammas = {
                f"g{i+1}": torch.ones(dim, dtype=dtype, device=device)
                for i, dim in enumerate(num_filters)
            }

            betas = {
                f"beta{i+1}": torch.zeros(dim, device=device, dtype=dtype)
                for i, dim in enumerate(num_filters)
            }
        else:
            gammas, betas = {}, {}

        self.params = {
            **weights,
            **biases,

            **gammas,
            **betas
        }
        ################################################################
        #                      END OF YOUR CODE                        #
        ################################################################

        # With batch normalization we need to keep track of running
        # means and variances, so we need to pass a special bn_param
        # object to each batch normalization layer. You should pass
        # self.bn_params[0] to the forward pass of the first batch
        # normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for _ in range(len(num_filters))]

        # Check that we got the right number of parameters
        if not self.batchnorm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 4  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of ' \
              'elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        assert len(self.params) == num_params, msg

        # Check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' \
                  % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' \
                  % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'max_pools': self.max_pools,
          'batchnorm': self.batchnorm,
          'bn_params': self.bn_params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = \
                self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = \
                    self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the deep convolutional
        network.
        Input / output: Same API as ThreeLayerConvNet.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        # pass conv_param to the forward pass for the
        # convolutional layer
        # Padding and stride chosen to preserve the input
        # spatial size
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        #########################################################
        # TODO: Implement the forward pass for the DeepConvNet, #
        # computing the class scores for X and storing them in  #
        # the scores variable.                                  #
        #                                                       #
        # You should use the fast versions of convolution and   #
        # max pooling layers, or the convolutional sandwich     #
        # layers, to simplify your implementation.              #
        #########################################################
        h = X
        forward_caches = []

        # Apply convolution layers
        for i in range(self.num_layers - 1):
            W = self.params[f"W{i+1}"]
            b = self.params[f"b{i+1}"]

            # Checking whether max pooling and batch norm has to be applied seems
            # inefficient
            if i in self.max_pools:
                if self.batchnorm:
                    h, cache = Conv_BatchNorm_ReLU_Pool.forward(
                        h,
                        W,
                        b,
                        self.params[f"g{i+1}"],
                        self.params[f"beta{i+1}"],
                        conv_param,
                        self.bn_params[i],
                        pool_param
                    )
                else:
                    h, cache = Conv_ReLU_Pool.forward(h, W, b, conv_param, pool_param)
            else:
                if self.batchnorm:
                    h, cache = Conv_BatchNorm_ReLU.forward(
                        h,
                        W,
                        b,
                        self.params[f"g{i+1}"],
                        self.params[f"beta{i+1}"],
                        conv_param,
                        self.bn_params[i]
                    )
                else:
                    h, cache = Conv_ReLU.forward(h, W, b, conv_param)

            forward_caches.append(cache)

        # Apply final linear layer
        scores, cache = Linear.forward(
            h,
            self.params[f"W{self.num_layers}"],
            self.params[f"b{self.num_layers}"]
        )

        forward_caches.append(cache)
        #####################################################
        #                 END OF YOUR CODE                  #
        #####################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ###################################################################
        # TODO: Implement the backward pass for the DeepConvNet,          #
        # storing the loss and gradients in the loss and grads variables. #
        # Compute data loss using softmax, and make sure that grads[k]    #
        # holds the gradients for self.params[k]. Don't forget to add     #
        # L2 regularization!                                              #
        #                                                                 #
        # NOTE: To ensure that your implementation matches ours and you   #
        # pass the automated tests, make sure that your L2 regularization #
        # does not include a factor of 0.5                                #
        ###################################################################
        loss, dscores = softmax_loss(scores, y)

        # Adding regularization contribution to the overall loss
        for param_name, param_value in self.params.items():
            if 'W' in param_name:
                loss += self.reg * torch.sum(param_value * param_value)

        # Computing gradients
        # Starting with the final linear layer
        i = self.num_layers
        dh, grads[f"W{i}"], grads[f"b{i}"] = Linear.backward(
            dscores,
            forward_caches[i - 1]
        )

        grads[f"W{i}"] += 2 * self.reg * self.params[f"W{i}"]

        while i != 1:
            i -= 1

            if (i - 1) in self.max_pools:
                if self.batchnorm:
                    (
                        dh,
                        grads[f"W{i}"],
                        grads[f"b{i}"],
                        grads[f"g{i}"],
                        grads[f"beta{i}"]
                    ) = Conv_BatchNorm_ReLU_Pool.backward(
                        dh,
                        forward_caches[i - 1]
                    )
                else:
                    dh, grads[f"W{i}"], grads[f"b{i}"] = Conv_ReLU_Pool.backward(
                        dh,
                        forward_caches[i - 1]
                    )
            else:
                if self.batchnorm:
                    (
                        dh,
                        grads[f"W{i}"],
                        grads[f"b{i}"],
                        grads[f"g{i}"],
                        grads[f"beta{i}"]
                    ) = Conv_BatchNorm_ReLU.backward(
                        dh,
                        forward_caches[i - 1]
                    )
                else:
                    dh, grads[f"W{i}"], grads[f"b{i}"] = Conv_ReLU.backward(
                        dh,
                        forward_caches[i - 1]
                    )

            grads[f"W{i}"] += 2 * self.reg * self.params[f"W{i}"]
        #############################################################
        #                       END OF YOUR CODE                    #
        #############################################################

        return loss, grads


def find_overfit_parameters():
    weight_scale = 2e-3   # Experiment with this!
    learning_rate = 1e-5  # Experiment with this!
    ###########################################################
    # TODO: Change weight_scale and learning_rate so your     #
    # model achieves 100% training accuracy within 30 epochs. #
    ###########################################################
    weight_scale = 0.1
    learning_rate = 2e-3
    ###########################################################
    #                       END OF YOUR CODE                  #
    ###########################################################
    return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict, dtype, device):
    model = None
    solver = None
    #########################################################
    # TODO: Train the best DeepConvNet that you can on      #
    # CIFAR-10 within 60 seconds.                           #
    #########################################################
    model = DeepConvNet(
        input_dims=data_dict["X_train"].shape[1:],
        num_classes=10,

        num_filters=([8] * 3) + ([16] * 3) + ([32] * 3) + ([64] * 3),
        max_pools=[5, 8, 11],

        batchnorm=False,
        weight_scale="kaiming",
        reg=3e-4,
        dtype=dtype,
        device=device
    )

    solver = Solver(
        model,
        data_dict,
        num_epochs=30,
        batch_size=256,
        update_rule=adam,
        optim_config={
            "learning_rate": 2e-3
        },
        print_every=50,
        device=device
    )
    #########################################################
    #                  END OF YOUR CODE                     #
    #########################################################
    return solver


def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initializaiton); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, Dout); for a
      convolution layer it should have shape (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.
    weight = None
    if K is None:
        ###################################################################
        # TODO: Implement Kaiming initialization for linear layer.        #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din).                           #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        weight = torch.randn(
            (Din, Dout),
            dtype=dtype,
            device=device
        ) * torch.sqrt(torch.tensor(gain) / Din)
        ###################################################################
        #                            END OF YOUR CODE                     #
        ###################################################################
    else:
        ###################################################################
        # TODO: Implement Kaiming initialization for convolutional layer. #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din) * K * K                    #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        weight = torch.randn(
            (Dout, Din, K, K),
            dtype=dtype,
            device=device
        ) * torch.sqrt(torch.tensor(gain) / (Din * K * K))
        ###################################################################
        #                         END OF YOUR CODE                        #
        ###################################################################
    return weight


class BatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization.

        During training the sample mean and (uncorrected) sample variance
        are computed from minibatch statistics and used to normalize the
        incoming data. During training we also keep an exponentially decaying
        running mean of the mean and variance of each feature, and these
        averages are used to normalize data at test-time.

        At each timestep we update the running averages for mean and
        variance using an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        Note that the batch normalization paper suggests a different
        test-time behavior: they compute sample mean and variance for
        each feature using a large number of training images rather than
        using a running average. For this implementation we have chosen to use
        running averages instead since they do not require an additional
        estimation step; the PyTorch implementation of batch normalization
        also uses running averages.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift paremeter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean
            of features
          - running_var Array of shape (D,) giving running variance
            of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_param.get('running_mean',
                                    torch.zeros(D,
                                                dtype=x.dtype,
                                                device=x.device))
        running_var = bn_param.get('running_var',
                                   torch.zeros(D,
                                               dtype=x.dtype,
                                               device=x.device))

        out, cache = None, None
        if mode == 'train':
            ##################################################################
            # TODO: Implement the training-time forward pass for batch norm. #
            # Use minibatch statistics to compute the mean and variance, use #
            # these statistics to normalize the incoming data, and scale and #
            # shift the normalized data using gamma and beta.                #
            #                                                                #
            # You should store the output in the variable out.               #
            # Any intermediates that you need for the backward pass should   #
            # be stored in the cache variable.                               #
            #                                                                #
            # You should also use your computed sample mean and variance     #
            # together with the momentum variable to update the running mean #
            # and running variance, storing your result in the running_mean  #
            # and running_var variables.                                     #
            #                                                                #
            # Note that though you should be keeping track of the running    #
            # variance, you should normalize the data based on the standard  #
            # deviation (square root of variance) instead!                   #
            # Referencing the original paper                                 #
            # (https://arxiv.org/abs/1502.03167) might prove to be helpful.  #
            ##################################################################
            # Compute batch mean and variance
            x_mean = x.mean(dim=0)
            x_centered = x - x_mean
            x_centered_squared = x_centered ** 2
            x_var = x_centered_squared.mean(dim=0)
            sqrt_x_var = torch.sqrt(x_var + eps)
            inv_sqrt_x_var = 1 / sqrt_x_var
            x_normed = x_centered * inv_sqrt_x_var

            # x_var, x_mean = torch.var_mean(x, dim=0, correction=0)

            # Update running mean and variance
            running_mean = momentum * running_mean + (1 - momentum) * x_mean
            running_var = momentum * running_var + (1 - momentum) * x_var

            # Compute normalized input

            # Compute out, which is the normalized input shifted and scaled
            out = gamma * x_normed + beta

            # TODO: What is it that I need in the cache?
            cache = (
                x,
                x_normed,
                x_centered,
                x_mean,
                x_var,
                sqrt_x_var,
                inv_sqrt_x_var,
                gamma,
                eps
            )

            ################################################################
            #                           END OF YOUR CODE                   #
            ################################################################
        elif mode == 'test':
            ################################################################
            # TODO: Implement the test-time forward pass for               #
            # batch normalization. Use the running mean and variance to    #
            # normalize the incoming data, then scale and shift the        #
            # normalized data using gamma and beta. Store the result       #
            # in the out variable.                                         #
            ################################################################
            x_normed = (x - running_mean) / torch.sqrt(running_var + eps)

            out = gamma * x_normed + beta
            ################################################################
            #                      END OF YOUR CODE                        #
            ################################################################
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for batch normalization.

        For this implementation, you should write out a
        computation graph for batch normalization on paper and
        propagate gradients backward through intermediate nodes.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma,
          of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta,
          of shape (D,)
        """
        dx, dgamma, dbeta = None, None, None
        #####################################################################
        # TODO: Implement the backward pass for batch normalization.        #
        # Store the results in the dx, dgamma, and dbeta variables.         #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167) #
        # might prove to be helpful.                                        #
        # Don't forget to implement train and test mode separately.         #
        #####################################################################
        # NOTICE: the backward pass as implemented here assumes that the 
        # batch norm layer is in "train" mode, where the input batch is used
        # to compute sample mean and variance. If we are operating in "test"
        # mode, the backpropagation would be different (and simpler)
        # Unpack cache
        (
            x,
            x_normed,
            x_centered,
            x_mean,
            x_var,
            sqrt_x_var,
            inv_sqrt_x_var,
            gamma,
            eps
        ) = cache

        # Derivatives of the shift and scale params
        dbeta = dout.sum(dim=0)
        dgamma = (x_normed * dout).sum(dim=0)

        # Building blocks for the derivative of the input
        N = x.shape[0]

        dx_normed = dout * gamma
        dinv_sqrt_x_var = (dx_normed * x_centered).sum(dim=0)
        dsqrt_x_var = - dinv_sqrt_x_var / sqrt_x_var ** 2
        dx_var = dsqrt_x_var / 2 / sqrt_x_var
        dx_centered_squared = dx_var * torch.ones_like(x) / N
        dx_centered = dx_normed * inv_sqrt_x_var + dx_centered_squared * 2 * \
            x_centered
        dx_mean = - torch.sum(dx_centered, dim=0)
        dx = dx_centered + dx_mean * torch.ones_like(x) / N

        #################################################################
        #                      END OF YOUR CODE                         #
        #################################################################

        return dx, dgamma, dbeta

    @staticmethod
    def backward_alt(dout, cache):
        """
        Alternative backward pass for batch normalization.
        For this implementation you should work out the derivatives
        for the batch normalizaton backward pass on paper and simplify
        as much as possible. You should be able to derive a simple expression
        for the backward pass. See the jupyter notebook for more hints.

        Note: This implementation should expect to receive the same
        cache variable as batchnorm_backward, but might not use all of
        the values in the cache.

        Inputs / outputs: Same as batchnorm_backward
        """
        dx, dgamma, dbeta = None, None, None
        ###################################################################
        # TODO: Implement the backward pass for batch normalization.      #
        # Store the results in the dx, dgamma, and dbeta variables.       #
        #                                                                 #
        # After computing the gradient with respect to the centered       #
        # inputs, you should be able to compute gradients with respect to #
        # the inputs in a single statement; our implementation fits on a  #
        # single 80-character line.                                       #
        ###################################################################
        # Unpack cache
        (
            x,
            x_normed,
            x_centered,
            x_mean,
            x_var,
            sqrt_x_var,
            inv_sqrt_x_var,
            gamma,
            eps
        ) = cache

        # Derivatives of the shift and scale params
        dbeta = dout.sum(dim=0)
        dgamma = (x_normed * dout).sum(dim=0)

        # Building blocks for the derivative of the input
        N = x.shape[0]

        # dx = gamma * torch.sum(dout * x_centered, dim=0) / sqrt_x_var / N

        dx = (1 / N) * gamma * (1 / sqrt_x_var) * (
            N * dout - dout.sum(dim=0) - x_centered / (x_var + eps) * \
                torch.sum(dout * x_centered, dim=0)
        )
        #################################################################
        #                        END OF YOUR CODE                       #
        #################################################################

        return dx, dgamma, dbeta


class SpatialBatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Computes the forward pass for spatial batch normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance. momentum=0
            means that old information is discarded completely at every
            time step, while momentum=1 means that new information is never
            incorporated. The default of momentum=0.9 should work well
            in most situations.
          - running_mean: Array of shape (C,) giving running mean of
            features
          - running_var Array of shape (C,) giving running variance
            of features

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        out, cache = None, None

        ################################################################
        # TODO: Implement the forward pass for spatial batch           #
        # normalization.                                               #
        #                                                              #
        # HINT: You can implement spatial batch normalization by       #
        # calling the vanilla version of batch normalization you       #
        # implemented above. Your implementation should be very short; #
        # ours is less than five lines.                                #
        ################################################################
        N, C, H, W = x.shape

        out, cache = BatchNorm.forward(
            (x
                .movedim(1, 3) # Move channel dimension to the end
                .flatten(end_dim=2)), # Flatten batch and spatial dimensions together
            gamma,
            beta,
            bn_param
        )

        # Undoing the dimension transforms
        out = out.reshape((N, H, W, C)).movedim(3, 1)
        ################################################################
        #                       END OF YOUR CODE                       #
        ################################################################

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for spatial batch normalization.
        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H, W)
        - cache: Values from the forward pass
        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        dx, dgamma, dbeta = None, None, None

        #################################################################
        # TODO: Implement the backward pass for spatial batch           #
        # normalization.                                                #
        #                                                               #
        # HINT: You can implement spatial batch normalization by        #
        # calling the vanilla version of batch normalization you        #
        # implemented above. Your implementation should be very short;  #
        # ours is less than five lines.                                 #
        #################################################################
        N, C, H, W = dout.shape

        dx, dgamma, dbeta = BatchNorm.backward(
            dout.movedim(1, 3).flatten(end_dim=2),
            cache
        )

        # Undoing dimension transforms
        dx = dx.reshape((N, H, W, C)).movedim(3, 1)
        ##################################################################
        #                       END OF YOUR CODE                         #
        ##################################################################

        return dx, dgamma, dbeta

##################################################################
#           Fast Implementations and Sandwich Layers             #
##################################################################


class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = torch.zeros_like(tx), \
                         torch.zeros_like(layer.weight), \
                         torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = \
            pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A convenience layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool
        convenience layer
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs an linear transform,
        batch normalization, and ReLU.
        Inputs:
        - x: Array of shape (N, D1); input to the linear layer
        - w, b: Arrays of shape (D2, D2) and (D2,) giving the
          weight and bias for the linear transform.
        - gamma, beta: Arrays of shape (D2,) and (D2,) giving
          scale and shift parameters for batch normalization.
        - bn_param: Dictionary of parameters for batch
          normalization.
        Returns:
        - out: Output from ReLU, of shape (N, D2)
        - cache: Object to give to the backward pass.
        """
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu
        convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma,
                                                beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta
