import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import array
from chainer.utils import type_check

from chainer.functions.math.matmul import _get_batch_mat_shape, _check_ndim, _convert_type, _get_check_index, _matmul

try:
    from chainer.functions.math.matmul import _batch_matmul_gpu
except ImportError:
    def _batch_matmul_gpu(*args, **kwds):
        # TO DO: fix this for recent chainer
        print "Sorry, use of lexical probabilities is broken with this version of chainer"
        raise NotImplemented


class MatMulConstant(function.Function):

    def __init__(self, b, transa=False, transb=False):
        self.transa = transa
        self.transb = transb
        self.b = b

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        a_type, = in_types

        type_check.expect(
            a_type.dtype.kind == 'f',
        )

        _check_ndim(a_type)

        a_type = _convert_type(a_type)
        a_idx = _get_check_index(self.transa, False)

    def forward(self, x):
        a, = x
        return _matmul(a, self.b, transa=self.transa, transb=self.transb),

    def backward(self, x, gy):
        gx0 = _matmul(
            gy[0], self.b, transb=not self.transb, transout=self.transa
        ).reshape(x[0].shape)
        return gx0,


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
        self.transa = transa
        self.transb = transb
        self.b = b

    def _output_shape(self, a, b):
        batch_size = len(a)
        m = _get_batch_mat_shape(a.shape)[2 if self.transa else 1]
        n = _get_batch_mat_shape(b.shape)[1 if self.transb else 2]
        return batch_size, m, n

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        a_type, = in_types

        type_check.expect(
            a_type.dtype == numpy.float32
            #             ,
            #             self.b.dtype == numpy.float32
        )

        _check_ndim(a_type, lower=2, upper=3)
#         _check_ndim(b_type, lower=2, upper=3)

        a_type = _convert_type(a_type, vector_ndim=2)
#         b_type = _convert_type(b_type, vector_ndim=2)
        a_idx = _get_check_index(self.transa, False, row_idx=1, col_idx=2)
#         b_idx = _get_check_index(self.transb, True, row_idx=1, col_idx=2)
#         type_check.expect(
#             a_type.shape[a_idx] == b_type.shape[b_idx]
#         )

    def forward_cpu(self, x):
        a, = x
        b = self.b
        batch_size = a.shape[0]
        shape = self._output_shape(a, b)
        ret_dtype = numpy.find_common_type([a.dtype, b.dtype], [])
        ret = numpy.empty(shape, dtype=ret_dtype)
        for i in six.moves.range(batch_size):
            ret[i] = _matmul(
                a[i], b[i], transa=self.transa, transb=self.transb)
        return ret,

    def backward_cpu(self, x, gy):
        a, = x
        b = self.b
        ga = numpy.empty_like(a)
        a0shape = a[0].shape
        for i in six.moves.range(len(a)):
            ga[i] = _matmul(
                gy[0][i], b[i], transb=not self.transb,
                transout=self.transa).reshape(a0shape)
        return ga,

    def forward_gpu(self, x):
        a, = x
        b = self.b
        shape = self._output_shape(a, b)
        ret = cuda.cupy.zeros(shape, dtype=a.dtype)
        _batch_matmul_gpu(
            a, b, transa=self.transa, transb=self.transb, out=ret)
        return ret,

    def backward_gpu(self, x, gy):
        a, = x
        b = self.b
        ga = cuda.cupy.empty(_get_batch_mat_shape(a.shape), a.dtype)
        _batch_matmul_gpu(
            gy[0], b, transb=not self.transb, transout=self.transa, out=ga)
        ga = ga.reshape(a.shape)
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
