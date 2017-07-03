#Backporting faster layer normalization from chainer v2

from chainer import cuda
from chainer import function
from chainer.utils import type_check
import chainer.links as L

import logging
logging.basicConfig()
log = logging.getLogger("rnns:lnks")
log.setLevel(logging.INFO)

use_chainer_layer_normalization = True

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


class LayerNormalization(function.Function):

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
        self.add_param('gamma', size)
        initializers.init_weight(self.gamma.data, self._gamma_initializer)
        self.add_param('beta', size)
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

