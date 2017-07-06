#Backporting faster layer normalization from chainer v2

from chainer import cuda
from chainer import function
from chainer.utils import type_check
import chainer.links as L
import math
import logging
logging.basicConfig()
log = logging.getLogger("rnns:lnks")
log.setLevel(logging.INFO)

use_chainer_layer_normalization = True

def turn_on_own_layer_normalization():
    global use_chainer_layer_normalization
    use_chainer_layer_normalization = False

def get_layer_normalization_class():
    global use_chainer_layer_normalization
    if use_chainer_layer_normalization:
        return L.LayerNormalization
    else:
        log.info("using faster LayerNormalization")
        return LayerNormalizationLink
    
    
def _broadcast_to(xp, x, shape):
    if hasattr(xp, 'broadcast_to'):
        return xp.broadcast_to(x, shape)
    else:
        # numpy 1.9 doesn't support broadcast_to method
        dummy = xp.empty(shape)
        bx, _ = xp.broadcast_arrays(x, dummy)
        return bx


try:
    import cupy as cp

    inv_norm_comp = cp.ReductionKernel(
        'T x',  # input params
        'T y',  # output params
        'x * x',  # map
        'a + b',  # reduce
        'y = 1.0/sqrt(a + 1e-5)',  # post-reduction map
        '0',  # identity value
        'inv_norm_comp'  # kernel name
    )
    
    
    scale_output = cp.ElementwiseKernel(
         'T x, T inv_norm, T gamma, T beta',
         'T normalized, T scaled',
          '''
              normalized = x * inv_norm;
              scaled = normalized * gamma + beta;
         ''',
         'scale_output')
    
    backprop_scale = cp.ElementwiseKernel(
         'T inv_norm, T gy_centered, T normalized, T sc_prod',
         'T z',
          '''
              z = inv_norm *(gy_centered - normalized * sc_prod);
         ''',
         'backprop_scale')
except ImportError:
    inv_norm_comp = None
    scale_output = None
    backprop_scale = None
        
class LayerNormalization(function.Function):
    def __init__(self, eps=1e-5, gpu_optim=True):
        self.eps = eps
        self.gpu_optim = gpu_optim
        
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        x_type, gamma_type, beta_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim == 2,
            gamma_type.ndim == 2,
            beta_type.ndim == 2,
            gamma_type.dtype == x_type.dtype,
            beta_type.dtype == x_type.dtype,
            gamma_type.shape == beta_type.shape,
            gamma_type.shape[0] == 1
        )
        
    def forward_cpu(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, gamma, beta = inputs
        a = x - xp.mean(x, axis=1, keepdims=True)
        assert len(a.shape) == 2
        inv_norm = 1.0/xp.sqrt(xp.mean(xp.square(a), axis=1, keepdims=True) + self.eps)
        self.inv_norm = inv_norm
        normalized = a * inv_norm
        self.normalized = normalized
        scaled = normalized * gamma + beta
        return scaled,
    
    def forward_gpu(self, inputs):
        if not self.gpu_optim:
            return self.forward_cpu(inputs)
        xp = cuda.get_array_module(*inputs)
        x, gamma, beta = inputs
        a = x - xp.mean(x, axis=1, keepdims=True)
        assert len(a.shape) == 2
        H = a.shape[1]
        
#         inv_norm = inv_norm_comp(a/math.sqrt(H), axis=1, keepdims=True) # 1.0/xp.sqrt(xp.sum(a*a, axis=1, keepdims=True) + self.eps)
        
        inv_norm = cp.ReductionKernel(
        'T x',  # input params
        'T y',  # output params
        'x * x',  # map
        'a + b',  # reduce
        'y = 1.0/sqrt(a/%f + %f)'%(H, self.eps),  # post-reduction map
        '0',  # identity value
        'inv_norm_comp'  # kernel name
        )(a, axis=1, keepdims=True)
        
        self.inv_norm = inv_norm
        
        normalized, scaled = scale_output(a, inv_norm, gamma, beta)
        self.normalized = normalized
        return scaled,

    def backward_gpu(self, inputs, gys):
        if not self.gpu_optim:
            return self.backward_cpu(inputs,  gys)
        xp = cuda.get_array_module(*inputs)
        x, gamma, beta = inputs
        gy, = gys
        g_beta = xp.sum(gy, axis=0, keepdims=True)
        g_gamma = xp.sum(gy*self.normalized, axis=0, keepdims=True)
        
        gy2 = gy*gamma
        gy_centered = gy2 - xp.mean(gy2, axis=1, keepdims=True)
        sc_prod = xp.sum(gy_centered * self.normalized, axis = 1, keepdims=True)
        
        H = x.shape[1]
#         ga = backprop_scale(self.inv_norm, gy_centered, self.normalized, sc_prod/H)
        ga = cp.ElementwiseKernel(
         'T inv_norm, T gy_centered, T normalized, T sc_prod',
         'T z',
          '''
              z = inv_norm *(gy_centered - normalized * (sc_prod/%f));
         '''%H,
         'backprop_scale')(self.inv_norm, gy_centered, self.normalized, sc_prod)
        
        return ga, g_gamma, g_beta 
   
    def backward_cpu(self, inputs, gys):
        xp = cuda.get_array_module(*inputs)
        x, gamma, beta = inputs
        gy, = gys
        g_beta = xp.sum(gy, axis=0, keepdims=True)
        g_gamma = xp.sum(gy*self.normalized, axis=0, keepdims=True)
        
        gy2 = gy*gamma
        gy_centered = gy2 - xp.mean(gy2, axis=1, keepdims=True)
        
        H = x.shape[1]
        sc_prod = xp.sum(gy_centered * self.normalized, axis = 1, keepdims=True)/H
        
        ga = self.inv_norm *(gy_centered - self.normalized * sc_prod)
        return ga, g_gamma, g_beta 

class LayerNormalizationOther(function.Function):

    """Layer normalization"""

    def __init__(self, eps=1e-5):
        self.eps = eps

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        x_type, gamma_type, beta_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim == 2,
            gamma_type.ndim == 1,
            beta_type.ndim == 1,
            gamma_type.dtype == x_type.dtype,
            beta_type.dtype == x_type.dtype,
            gamma_type.shape == beta_type.shape,
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, gamma, beta = inputs
        mu = xp.mean(x, axis=1, keepdims=True)
        self.x_mu = x - mu
        self.squ_x_mu = xp.square(self.x_mu)
        self.var = xp.mean(self.squ_x_mu, axis=1, keepdims=True)
        std = xp.sqrt(self.var + self.eps)
        self.inv_std = 1. / std
        self.x_hat = self.x_mu * self.inv_std
        scaled_x = self.x_hat * gamma[None, ]
        shifted_x = scaled_x + beta[None, ]
        return shifted_x,

    def backward(self, inputs, gy):
        xp = cuda.get_array_module(*inputs)
        x, gamma, beta = inputs
        gy = gy[0]

        g_beta = gy.sum(axis=0)
        g_scaled_x = gy

        g_gamma = xp.sum(g_scaled_x * self.x_hat, axis=0)
        g_x_hat = g_scaled_x * gamma[None, ]

        g_inv_std = xp.sum(g_x_hat * self.x_mu, axis=1, keepdims=True)
        g_x_mu_1 = g_x_hat * self.inv_std

        g_std = g_inv_std * (- 1. / self.var)
        g_var = g_std * 0.5 * self.inv_std

        n_units = x.shape[1]
        g_squ_x_mu = _broadcast_to(xp, g_var * 1. / n_units, x.shape)
        g_x_mu_2 = g_squ_x_mu * 2 * self.x_mu

        g_x_1 = g_x_mu_1 + g_x_mu_2
        g_mu = xp.sum(g_x_1, axis=1, keepdims=True) * (- 1.)

        g_x_2 = _broadcast_to(xp, g_mu * 1. / n_units, x.shape)

        g_x = g_x_1 + g_x_2

        return g_x, g_gamma, g_beta,


def layer_normalization(x, gamma, beta, eps=1e-5):
    """Layer normalization.
    This function implements a "layer normalization"
    which normalizes the input units by statistics
    that are computed along the second axis,
    scales and shifts them.
    Args:
        x (~chainer.Variable): Batch vectors.
            Shape of this value must be `(batch_size, unit_size)`,
            e.g., the output of :func:`~chainer.functions.linear`.
        gamma (~chainer.Variable): Scaling vectors.
        beta (~chainer.Variable): Shifting vectors.
    Returns:
        ~chainer.Variable: The output variable which has the same shape
        as :math:`x`.
    See: `Layer Normalization <https://arxiv.org/abs/1607.06450>`_
    """
    return LayerNormalization(eps)(x, gamma, beta)

from chainer import link
from chainer import utils
from chainer import variable
from chainer import initializers

class LayerNormalizationLink(link.Link):

    """Layer normalization layer on outputs of linear functions.
    This link implements a "layer normalization" layer
    which normalizes the input units by statistics
    that are computed along the second axis,
    scales and shifts them.
    Parameter initialization will be deferred until
    the first forward data pass at which time the size will be determined.
    Args:
        size (int): Size of input units. If ``None``, parameter initialization
            will be deferred until the first forward data pass at which time
            the size will be determined.
        eps (float): Epsilon value for numerical stability of normalization.
        initial_gamma (~chainer.Initializer): Initializer for scaling vector.
            If ``None``, then the vector is filled by 1.
            If a scalar, the vector is filled by it.
            If ``numpy.ndarray``, the vector is set by it.
        initial_beta (~chainer.Initializer): Initializer for shifting vector.
            If ``None``, then the vector is filled by 0.
            If a scalar, the vector is filled by it.
            If ``numpy.ndarray``, the vector is set by it.
    Attributes:
        gamma (~chainer.Parameter): Scaling parameter.
        beta (~chainer.Parameter): Shifting parameter.
        eps (float): Epsilon value for numerical stability.
    See: `Layer Normalization <https://arxiv.org/abs/1607.06450>`_
    """

    def __init__(self, size=None, eps=1e-6, initial_gamma=None,
                 initial_beta=None):
        super(LayerNormalizationLink, self).__init__()
        self.add_uninitialized_param('gamma')
        self.add_uninitialized_param('beta')
        if initial_gamma is None:
            initial_gamma = initializers.One()
        self._gamma_initializer = initial_gamma
        if initial_beta is None:
            initial_beta = initializers.Zero()
        self._beta_initializer = initial_beta
        self.eps = eps

        if size is not None:
            self._initialize_params(size)

    def _initialize_params(self, size):
        self.add_param('gamma', (1,size))
        initializers.init_weight(self.gamma.data, self._gamma_initializer)
        self.add_param('beta', (1,size))
        initializers.init_weight(self.beta.data, self._beta_initializer)

    def __call__(self, x):
        """Apply layer normalization to given input.
        Args:
            x (~chainer.Variable): Batch vectors.
                Shape of this value must be `(batch_size, unit_size)`,
                e.g., the output of :func:`~chainer.functions.linear`.
        Returns:
            ~chainer.Variable: Output of the layer normalization.
        """
#         if self.gamma.data is None:
#             self._initialize_params(x.size // x.shape[0])

        if self.has_uninitialized_params:
            with cuda.get_device_from_id(self._device_id):
                self._initialize_params(x.size // x.shape[0])

        return layer_normalization(
                x, self.gamma, self.beta, self.eps)

