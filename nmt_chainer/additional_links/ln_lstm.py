import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _extract_gates(x):
    r = x.reshape((len(x), x.shape[1] // 4, 4) + x.shape[2:])
    return [r[:, :, i] for i in six.moves.range(4)]


def _sigmoid(x):
    half = x.dtype.type(0.5)
    return numpy.tanh(x * half) * half + half


def _grad_sigmoid(x):
    return x * (1 - x)


def _grad_tanh(x):
    return 1 - x * x


_preamble = '''
template <typename T> __device__ T sigmoid(T x) {
    const T half = 0.5;
    return tanh(x * half) * half + half;
}
template <typename T> __device__ T grad_sigmoid(T y) { return y * (1 - y); }
template <typename T> __device__ T grad_tanh(T y) { return 1 - y * y; }

#define COMMON_ROUTINE \
    T aa = tanh(a); \
    T ai = sigmoid(i_); \
    T af = sigmoid(f); \
    T ao = sigmoid(o);
'''


class LSTMWithUngatedOutput(function.Function):

    """Long short-term memory unit with forget gate.

    It has two inputs (c, x) and two outputs (c, h), where c indicates the cell
    state. x must have four times channels compared to the number of units.

    """

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        c_type, x_type = in_types

        type_check.expect(
            c_type.dtype.kind == 'f',
            x_type.dtype == c_type.dtype,

            c_type.ndim >= 2,
            x_type.ndim >= 2,
            c_type.ndim == x_type.ndim,

            x_type.shape[0] <= c_type.shape[0],
            x_type.shape[1] == 4 * c_type.shape[1],
        )
        for i in six.moves.range(2, c_type.ndim.eval()):
            type_check.expect(x_type.shape[i] == c_type.shape[i])

    def forward(self, inputs):
        c_prev, x = inputs
        a, i, f, o = _extract_gates(x)
        batch = len(x)

        if isinstance(x, numpy.ndarray):
            self.a = numpy.tanh(a)
            self.i = _sigmoid(i)
            self.f = _sigmoid(f)
            self.o = _sigmoid(o)

            c_next = numpy.empty_like(c_prev)
            c_next[:batch] = self.a * self.i + self.f * c_prev[:batch]
            ungated_h = numpy.tanh(c_next[:batch])
            o_gate = self.o
        else:
            c_next = cuda.cupy.empty_like(c_prev)
            ungated_h = cuda.cupy.empty_like(c_next[:batch])
            o_gate = cuda.cupy.empty_like(c_next[:batch])
            cuda.elementwise(
                'T c_prev, T a, T i_, T f, T o', 'T c, T ungated_h, T o_gate',
                '''
                    COMMON_ROUTINE;
                    c = aa * ai + af * c_prev;
                    ungated_h = tanh(c);
                    o_gate = ao;
                ''',
                'lstm_fwd', preamble=_preamble)(
                    c_prev[:batch], a, i, f, o, c_next[:batch], ungated_h, o_gate)

        c_next[batch:] = c_prev[batch:]
        self.c = c_next[:batch]
        return c_next, ungated_h, o_gate

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        c_prev, x = inputs
        batch = len(x)
        gc, gh , go_gate = grad_outputs

        gx = xp.empty_like(x)
        ga, gi, gf, go = _extract_gates(gx)

        # Consider the case that either gradient is not given
        if gc is None:
            gc_update = 0
            gc_rest = 0
        else:
            gc_update = gc[:batch]
            gc_rest = gc[batch:]
        if gh is None:
            gh = 0
        if go_gate is None:
            go_gate = 0

        if xp is numpy:
            co = numpy.tanh(self.c)
            gc_prev = numpy.empty_like(c_prev)
            # multiply f later
            gc_prev[:batch] = gh * _grad_tanh(co) + gc_update #  self.o *
            gc = gc_prev[:batch]
            ga[:] = gc * self.i * _grad_tanh(self.a)
            gi[:] = gc * self.a * _grad_sigmoid(self.i)
            gf[:] = gc * c_prev[:batch] * _grad_sigmoid(self.f)
            go[:] = _grad_sigmoid(self.o) * go_gate # gh * co * 
            gc_prev[:batch] *= self.f  # multiply f here
            gc_prev[batch:] = gc_rest
        else:
            a, i, f, o = _extract_gates(x)
            gc_prev = xp.empty_like(c_prev)
            cuda.elementwise(
                'T c_prev, T c, T gc, T gh, T a, T i_, T f, T o, T go_gate',
                'T gc_prev, T ga, T gi, T gf, T go',
                '''
                    COMMON_ROUTINE;
                    T co = tanh(c);
                    T temp = gh * grad_tanh(co) + gc;
                    ga = temp * ai * grad_tanh(aa);
                    gi = temp * aa * grad_sigmoid(ai);
                    gf = temp * c_prev * grad_sigmoid(af);
                    go = go_gate * grad_sigmoid(ao);
                    gc_prev = temp * af;
                ''',
                'lstm_bwd', preamble=_preamble)(
                    c_prev[:batch], self.c, gc_update, gh, a, i, f, o, go_gate,
                    gc_prev[:batch], ga, gi, gf, go)
            gc_prev[batch:] = gc_rest

        return gc_prev, gx


def lstm_with_ungated_output(c_prev, x):
    """Long Short-Term Memory units as an activation function.

    This function implements LSTM units with forget gates. Let the previous
    cell state ``c_prev`` and the input array ``x``.

    First, the input array ``x`` is split into four arrays
    :math:`a, i, f, o` of the same shapes along the second axis. It means that
    ``x`` 's second axis must have 4 times the ``c_prev`` 's second axis.

    The split input arrays are corresponding to:

        - :math:`a` : sources of cell input
        - :math:`i` : sources of input gate
        - :math:`f` : sources of forget gate
        - :math:`o` : sources of output gate

    Second, it computes the updated cell state ``c`` and the outgoing signal
    ``h`` as:

    .. math::

        c &= \\tanh(a) \\sigma(i)
           + c_{\\text{prev}} \\sigma(f), \\\\
        h &= \\tanh(c) \\sigma(o),

    where :math:`\\sigma` is the elementwise sigmoid function.
    These are returned as a tuple of two variables.

    This function supports variable length inputs. The mini-batch size of
    the current input must be equal to or smaller than that of the previous
    one. When mini-batch size of ``x`` is smaller than that of ``c``, this
    function only updates ``c[0:len(x)]`` and doesn't change the rest of ``c``,
    ``c[len(x):]``.
    So, please sort input sequences in descending order of lengths before
    applying the function.

    Args:
        c_prev (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Variable that holds the previous cell state. The cell state
            should be a zero array or the output of the previous call of LSTM.
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Variable that holds the sources of cell input, input gate, forget
            gate and output gate. It must have the second dimension whose size
            is four times of that of the cell state.

    Returns:
        tuple: Two :class:`~chainer.Variable` objects ``c`` and ``h``.
        ``c`` is the updated cell state. ``h`` indicates the outgoing signal.

    See the original paper proposing LSTM with forget gates:
    `Long Short-Term Memory in Recurrent Neural Networks \
    <http://www.felixgers.de/papers/phd.pdf>`_.

    .. seealso::
        :class:`~chainer.links.LSTM`

    .. admonition:: Example

        Assuming ``y`` is the current incoming signal, ``c`` is the previous
        cell state, and ``h`` is the previous outgoing signal from an ``lstm``
        function. Each of ``y``, ``c`` and ``h`` has ``n_units`` channels.
        Most typical preparation of ``x`` is:

        >>> n_units = 100
        >>> y = chainer.Variable(np.zeros((1, n_units), 'f'))
        >>> h = chainer.Variable(np.zeros((1, n_units), 'f'))
        >>> c = chainer.Variable(np.zeros((1, n_units), 'f'))
        >>> model = chainer.Chain(w=L.Linear(n_units, 4 * n_units),
        ...                       v=L.Linear(n_units, 4 * n_units),)
        >>> x = model.w(y) + model.v(h)
        >>> c, h = F.lstm(c, x)

        It corresponds to calculate the input array ``x``, or the input
        sources :math:`a, i, f, o`, from the current incoming signal ``y`` and
        the previous outgoing signal ``h``. Different parameters are used for
        different kind of input sources.

    .. note::

        We use the naming rule below.

        - incoming signal
            The formal input of the formulation of LSTM (e.g. in NLP, word
            vector or output of lower RNN layer). The input of
            :class:`chainer.links.LSTM` is the *incoming signal*.
        - input array
            The array which is linear transformed from *incoming signal* and
            the previous outgoing signal. The *input array* contains four
            sources, the sources of cell input, input gate, forget gate and
            output gate. The input of :class:`chainer.functions.LSTM` is the
            *input array*.

    """
    return LSTMWithUngatedOutput()(c_prev, x)


import numpy
import six

import chainer
from chainer import cuda
from chainer.functions.activation import lstm
from chainer.functions.array import concat
from chainer.functions.array import split_axis
from chainer import initializers
from chainer import link
from chainer.links.connection import linear
from chainer import variable

from nmt_chainer.additional_links.layer_normalization import LayerNormalizationLink as LayerNormalization

class LNStatelessLSTM(link.Chain):

    """Stateless LSTM layer.

    This is a fully-connected LSTM layer as a chain. Unlike the
    :func:`~chainer.functions.lstm` function, this chain holds upward and
    lateral connections as child links. This link doesn't keep cell and
    hidden states.

    Args:
        in_size (int or None): Dimension of input vectors. If ``None``,
            parameter initialization will be deferred until the first forward
            data pass at which time the size will be determined.
        out_size (int): Dimensionality of output vectors.

    Attributes:
        upward (chainer.links.Linear): Linear layer of upward connections.
        lateral (chainer.links.Linear): Linear layer of lateral connections.

    """

    def __init__(self, in_size, out_size,
                 lateral_init=None, upward_init=None,
                 bias_init=0, forget_bias_init=0):
        super(LNStatelessLSTM, self).__init__(
            upward=linear.Linear(in_size, 4 * out_size, initialW=0),
            lateral=linear.Linear(out_size, 4 * out_size,
                                  initialW=0, nobias=True),
            upward_ln = LayerNormalization(),
            lateral_ln = LayerNormalization(),
            output_ln = LayerNormalization()
        )
        self.state_size = out_size
        self.lateral_init = lateral_init
        self.upward_init = upward_init
        self.bias_init = bias_init
        self.forget_bias_init = forget_bias_init

        if in_size is not None:
            self._initialize_params()

    def _initialize_params(self):
        for i in six.moves.range(0, 4 * self.state_size, self.state_size):
            initializers.init_weight(
                self.lateral.W.data[i:i + self.state_size, :],
                self.lateral_init)
            initializers.init_weight(
                self.upward.W.data[i:i + self.state_size, :], self.upward_init)

        a, i, f, o = lstm._extract_gates(
            self.upward.b.data.reshape(1, 4 * self.state_size, 1))
        initializers.init_weight(a, self.bias_init)
        initializers.init_weight(i, self.bias_init)
        initializers.init_weight(f, self.forget_bias_init)
        initializers.init_weight(o, self.bias_init)

    def __call__(self, c, h, x):
        """Returns new cell state and updated output of LSTM.

        Args:
            c (~chainer.Variable): Cell states of LSTM units.
            h (~chainer.Variable): Output at the previous time step.
            x (~chainer.Variable): A new batch from the input sequence.

        Returns:
            tuple of ~chainer.Variable: Returns ``(c_new, h_new)``, where
                ``c_new`` represents new cell state, and ``h_new`` is updated
                output of LSTM units.

        """
        if self.upward.has_uninitialized_params:
            in_size = x.size // x.shape[0]
            with cuda.get_device_from_id(self._device_id):
                self.upward._initialize_params(in_size)
                self._initialize_params()

        lstm_in = self.upward_ln(self.upward(x))
        if h is not None:
            lstm_in += self.lateral_ln(self.lateral(h))
        if c is None:
            xp = self.xp
            with cuda.get_device_from_id(self._device_id):
                c = variable.Variable(
                    xp.zeros((x.shape[0], self.state_size), dtype=x.dtype),
                    volatile='auto')
        c_next, ungated_h, o_gate = lstm_with_ungated_output(c, lstm_in)
        h = o_gate * self.output_ln(ungated_h)
        return c_next, h
    
    