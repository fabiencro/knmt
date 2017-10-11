import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import array
from chainer.utils import type_check

from chainer.functions.math.matmul import _check_ndim, _get_check_index, _matmul

def _get_size(typ, index, vector_ndim):
    if type_check.eval(typ.ndim) == vector_ndim and \
       type_check.eval(index) == vector_ndim:
        return 1
    else:
        return typ.shape[index]


def _get_batch_mat_shape(shape):
    s = 1
    for x in shape[2:]:
        s *= x
    return shape[:2] + (s,)

def _batch_matmul(a, b, transa=False, transb=False, transout=False):
    a = a.reshape(a.shape[:2] + (-1,))
    b = b.reshape(b.shape[:2] + (-1,))
    trans_axis = (0, 2, 1)
    if transout:
        transa, transb = not transb, not transa
        a, b = b, a
    if transa:
        a = a.transpose(trans_axis)
    if transb:
        b = b.transpose(trans_axis)
    xp = cuda.get_array_module(a)
    if xp is numpy:
        ret = numpy.empty(a.shape[:2] + b.shape[2:], dtype=a.dtype)
        for i in six.moves.range(len(a)):
            ret[i] = numpy.dot(a[i], b[i])
        return ret
    return xp.matmul(a, b)

class MatMulConstant(function.Function):

    def __init__(self, b, transa=False, transb=False):
        self.b = b
        self.transa = transa
        self.transb = transb

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        a_type, = in_types

        type_check.expect(
            a_type.dtype.kind == 'f',
        )

        _check_ndim(a_type)

        a_idx = _get_check_index(self.transa, False)
        a_size = _get_size(a_type, a_idx, 1)

    def forward(self, x):
        a, = x
        return _matmul(a, self.b, transa=self.transa, transb=self.transb),

    def backward(self, x, gy):
        a, = x
        ga = _matmul(
            gy[0], self.b, transb=not self.transb, transout=self.transa
        ).reshape(a.shape)
        return ga,

def matmul_constant(a, b, transa=False, transb=False):
    """Computes the matrix multiplication of two arrays.
    Args:
        a (Variable): The left operand of the matrix multiplication.
            A 1-D array of shape ``(N,)`` is considered as an
            :math:`N \\times 1` matrix.
            A 2-D array of shape ``(M, N)`` is considered as an
            :math:`M \\times N` matrix.
        b (Variable): The right operand of the matrix multiplication.
            Its array is treated as a matrix in the same way as ``a``'s array.
        transa (bool): If ``True``, transpose a.
        transb (bool): If ``True``, transpose b.
    Returns:
        ~chainer.Variable: The result of the matrix multiplication as a 2-D
            array.
    """
    return MatMulConstant(b, transa=transa, transb=transb)(a)


class BatchMatMulConstant(function.Function):

    def __init__(self, b, transa=False, transb=False):
        self.b = b
        self.transa = transa
        self.transb = transb

    def _output_shape(self, a, b):
        batch_size = len(a)
        m = _get_batch_mat_shape(a.shape)[2 if self.transa else 1]
        n = _get_batch_mat_shape(self.b.shape)[1 if self.transb else 2]
        return batch_size, m, n

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        a_type, = in_types

        type_check.expect(
            a_type.dtype == numpy.float32,
        )

        _check_ndim(a_type, lower=2, upper=3)

        a_idx = _get_check_index(self.transa, False, row_idx=1, col_idx=2)

    def forward(self, x):
        a, = x
        return _batch_matmul(a, self.b, self.transa, self.transb),

    def backward(self, x, gy):
        a, = x
        ga = _batch_matmul(gy[0], self.b, transb=not self.transb,
                           transout=self.transa).reshape(a.shape)
        return ga,

def batch_matmul_constant(a, b, transa=False, transb=False):
    """Computes the batch matrix multiplications of two sets of arrays.

    Args:
        a (Variable): The left operand of the batch matrix multiplications.
            A 2-D array of shape ``(B, N)`` is considered as B
            :math:`N \\times 1` matrices.
            A 3-D array of shape ``(B, M, N)`` is considered as B
            :math:`M \\times N` matrices.
        b (Variable): The right operand of the batch matrix multiplications.
            Its array is treated as matrices in the same way as ``a``'s array.
        transa (bool): If ``True``, transpose each matrix in a.
        transb (bool): If ``True``, transpose each matrix in b.

    Returns:
        ~chainer.Variable: The result of the batch matrix multiplications as a
            3-D array.
    """
    return BatchMatMulConstant(b, transa=transa, transb=transb)(a)

