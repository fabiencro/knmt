import numpy

import chainer
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer.functions import split_axis
from chainer import link
from chainer.links.connection import linear

from chainer import cuda
from chainer import function
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _mode = libcudnn.CUDNN_ACTIVATION_SIGMOID


class SigmoidAPLusBByH(function.Function):

    """SigmoidAPLusBByH function."""

#     def __init__(self):
#         self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        type_check.expect(in_types[0].dtype == numpy.float32)
        type_check.expect(in_types[1].dtype == numpy.float32)
        type_check.expect(in_types[2].dtype == numpy.float32)

    def forward_cpu(self, x):
        self.sigma_a_plus_b = (numpy.tanh((x[0] + x[1]) * 0.5) * 0.5 + 0.5)
        return x[2] * self.sigma_a_plus_b,

    def forward_gpu(self, x):
        self.sigma_a_plus_b, y = cuda.elementwise(
            'T x1, T x2, T x3', 'T sigma_a_plus_b, T y',
            '''
                sigma_a_plus_b = tanh((x1 + x2) * 0.5) * 0.5 + 0.5;// 1 / (1 + exp(-(x1 + x2)));
                y = x3 * sigma_a_plus_b;
                ''',
            'sigmoid_a_plus_b_by_h_fwd')(x[0], x[1], x[2])
        return y,

    def backward_cpu(self, x, gy):
        gy_by_sigma_a_plus_b = gy[0] * self.sigma_a_plus_b
        deriv1 = x[2] * gy_by_sigma_a_plus_b * (1 - self.sigma_a_plus_b)
        return deriv1, deriv1, gy_by_sigma_a_plus_b

    def backward_gpu(self, x, gy):
        gx1, gx2, gx3 = cuda.elementwise(
            'T sigma_a_plus_b, T h, T gy', 'T gx1, T gx2, T gx3',
            '''
            gx3 = gy * sigma_a_plus_b;
            gx1 = h * gx3 * (1-sigma_a_plus_b);
            gx2 = gx1;
            ''',
            'sigmoid_bwd')(self.sigma_a_plus_b, x[2], gy[0])
        return gx1, gx2, gx3,


def sigm_a_plus_b_by_h_fast(x1, x2, x3):
    """ sigm_a_plus_b_by_h_fast
    """
    return SigmoidAPLusBByH()(x1, x2, x3)


class ComputeOutputGRU(function.Function):

    """SigmoidAPLusBByH function."""

#     def __init__(self):
#         self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 5)
        type_check.expect(in_types[0].dtype == numpy.float32)
        type_check.expect(in_types[1].dtype == numpy.float32)
        type_check.expect(in_types[2].dtype == numpy.float32)
        type_check.expect(in_types[3].dtype == numpy.float32)
        type_check.expect(in_types[4].dtype == numpy.float32)

    def forward_cpu(self, x):
        z_x, z_h, h_x, h, hh = x
        z = 1.0 / (1 + numpy.exp(- (z_x + z_h)))
        h_bar = numpy.tanh(h_x + hh)
        h_new = (1 - z) * h + z * h_bar

        self.z = z
        self.h_bar = h_bar
        return h_new,

    def forward_gpu(self, x):
        z_x, z_h, h_x, h, hh = x
        self.z, self.h_bar, h_new = cuda.elementwise(
            'T z_x, T z_h, T h_x, T h, T hh',
            'T z, T h_bar, T h_new',
            '''
                z = tanh((z_x + z_h) * 0.5) * 0.5 + 0.5;
                //z = 1.0/ ( 1 + exp(- (z_x + z_h)));
                h_bar = tanh(h_x + hh);
                h_new = (1 - z) * h + z * h_bar;
                ''',
            'compute_output_gru_fwd')(z_x, z_h, h_x, h, hh)
        return h_new,

    def backward_cpu(self, x, gy):
        z_x, z_h, h_x, h, hh = x
        g_h = (1 - self.z) * gy[0]
        g_hh = self.z * (1 - self.h_bar * self.h_bar) * gy[0]
        g_h_x = g_hh
        g_z_x = g_h * self.z * (self.h_bar - h)
        g_z_h = g_z_x
        return g_z_x, g_z_h, g_h_x, g_h, g_hh

    def backward_gpu(self, x, gy):
        z_x, z_h, h_x, h, hh = x
        g_z_x, g_z_h, g_h_x, g_h, g_hh = cuda.elementwise(
            'T z, T h_bar, T h, T gy', 'T g_z_x, T g_z_h, T g_h_x, T g_h, T g_hh', '''
            g_h = (1 - z) * gy;
            g_hh = z * (1 - h_bar * h_bar) * gy;
            g_h_x = g_hh;
            g_z_x = g_h * z * (h_bar - h);
            g_z_h = g_z_x;
            ''', 'compute_output_gru_bwd')(
            self.z, self.h_bar, h, gy[0])
        return g_z_x, g_z_h, g_h_x, g_h, g_hh,


def compute_output_GRU(z_x, z_h, h_x, h, hh):
    """ compute_output_GRU
    """
    return ComputeOutputGRU()(z_x, z_h, h_x, h, hh)


class ComputeOutputGRU2(function.Function):

    """SigmoidAPLusBByH function."""

#     def __init__(self):
#         self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        type_check.expect(in_types[0].dtype == numpy.float32)
        type_check.expect(in_types[1].dtype == numpy.float32)
        type_check.expect(in_types[2].dtype == numpy.float32)

    def forward_cpu(self, x):
        z, h, hh = x
        self.h_bar = numpy.tanh(hh)
        h_new = (1 - z) * h + z * self.h_bar
        return h_new,

    def forward_gpu(self, x):
        z, h, hh = x
        self.h_bar, h_new = cuda.elementwise(
            'T z, T h, T hh',
            'T h_bar, T h_new',
            '''
                h_bar = tanh(hh);
                h_new = (1 - z) * h + z * h_bar;
                ''',
            'compute_output_gru_fwd')(z, h, hh)
        return h_new,

    def backward_cpu(self, x, gy):
        z, h, hh = x
        g_h = (1 - z) * gy[0]
        g_z = (self.h_bar - h) * gy[0]
        g_hh = z * (1 - self.h_bar * self.h_bar) * gy[0]
        return g_z, g_h, g_hh

    def backward_gpu(self, x, gy):
        z, h, hh = x
        g_z, g_h, g_hh = cuda.elementwise(
            'T z, T h_bar, T h, T gy', 'T g_z, T g_h, T g_hh',
            '''
            g_h = (1 - z) * gy;
            g_hh = z * (1 - h_bar * h_bar) * gy;
            g_z = g_h * (h_bar - h);
            ''',
            'compute_output_gru_bwd')(z, self.h_bar, h, gy[0])
        return g_z, g_h, g_hh,


def compute_output_GRU2(z, h, hh):
    """ compute_output_GRU
    """
    return ComputeOutputGRU2()(z, h, hh)


def sigm_a_plus_b_by_h(r_x, r_h, h):
    return h * (sigmoid.sigmoid(r_x + r_h))


def compute_output(z_x, z_h, h_x, h, hh):
    z = sigmoid.sigmoid(z_x + z_h)
    h_bar = tanh.tanh(h_x + hh)
    h_new = (1 - z) * h + z * h_bar
    return h_new


class GRUBase(link.Chain):

    def __init__(self, n_units, n_inputs=None, init=None, bias_init=None):
        if n_inputs is None:
            n_inputs = n_units
        super(GRUBase, self).__init__(
            W_r_z_h=linear.Linear(n_inputs, n_units * 3, initialW=init, initial_bias=bias_init),
            U_r_z=linear.Linear(n_units, n_units * 2, initialW=init, initial_bias=bias_init),
            #             W_r=linear.Linear(n_inputs, n_units),
            #             U_r = linear.Linear(n_units, n_units),
            #             W_z=linear.Linear(n_inputs, n_units),
            #             U_z=linear.Linear(n_units, n_units),
            #             W=linear.Linear(n_inputs, n_units),
            U=linear.Linear(n_units, n_units),
        )
        self.n_units = n_units

    def initialize_with_classic_implementation(self, gru):
        assert isinstance(gru, chainer.links.GRU)
        self.W_r_z_h.W.data[:self.n_units] = gru.W_r.W.data
        self.W_r_z_h.W.data[self.n_units: 2 * self.n_units] = gru.W_z.W.data
        self.W_r_z_h.W.data[2 * self.n_units:] = gru.W.W.data
        self.W_r_z_h.b.data[:self.n_units] = gru.W_r.b.data
        self.W_r_z_h.b.data[self.n_units: 2 * self.n_units] = gru.W_z.b.data
        self.W_r_z_h.b.data[2 * self.n_units:] = gru.W.b.data

        self.U_r_z.W.data[:self.n_units] = gru.U_r.W.data
        self.U_r_z.W.data[self.n_units:] = gru.U_z.W.data
        self.U_r_z.b.data[:self.n_units] = gru.U_r.b.data
        self.U_r_z.b.data[self.n_units:] = gru.U_z.b.data

        self.U.W.data[...] = gru.U.W.data
        self.U.b.data[...] = gru.U.b.data


class GRUBase2(link.Chain):
    def __init__(self, n_units, n_inputs=None):
        if n_inputs is None:
            n_inputs = n_units
        super(GRUBase2, self).__init__(
            W_r_z=linear.Linear(n_inputs + n_units, n_units * 2),
            W_h=linear.Linear(n_inputs + n_units, n_units),
        )
        self.n_units = n_units


import chainer.functions as F


def compute_GRU_out_2(z, h, hh):
    h_bar = F.tanh(hh)
    h_new = (1 - z) * h + z * h_bar
    return h_new


class GRU2(GRUBase2):

    """Stateless Gated Recurrent Unit function (GRU).
       """

    def faster_call(self, h, x):
        h_x = F.concat((h, x), axis=1)
        z_r = F.sigmoid(self.W_r_z(h_x))

        z, r = F.split_axis(z_r, (self.n_units,), axis=1)

        hr_x = F.concat((h * r, x), axis=1)
        h_bar = F.tanh(self.W_h(hr_x))

        h_new = (1 - z) * h + z * h_bar
        return h_new

    def faster_call2(self, h, x):
        h_x = F.concat((h, x), axis=1)
        z_r = F.sigmoid(self.W_r_z(h_x))

        z, r = F.split_axis(z_r, (self.n_units,), axis=1)

        hr_x = F.concat((h * r, x), axis=1)

        h_new = compute_GRU_out_2(z, h, self.W_h(hr_x))
        return h_new

    def faster_call3(self, h, x):
        h_x = F.concat((h, x), axis=1)
        z_r = F.sigmoid(self.W_r_z(h_x))

        z, r = F.split_axis(z_r, (self.n_units,), axis=1)

        hr_x = F.concat((h * r, x), axis=1)

        h_new = compute_output_GRU2(z, h, self.W_h(hr_x))
        return h_new


class GRU(GRUBase):

    """Stateless Gated Recurrent Unit function (GRU).
       """

    def classic_call(self, h, x):
        r = sigmoid.sigmoid(self.W_r(x) + self.U_r(h))
        z = sigmoid.sigmoid(self.W_z(x) + self.U_z(h))
        h_bar = tanh.tanh(self.W(x) + self.U(r * h))
        h_new = (1 - z) * h + z * h_bar
        return h_new

    def faster_call(self, h, x):
        r_z_h_x = self.W_r_z_h(x)
        r_x, z_x, h_x = split_axis(r_z_h_x, (self.n_units, self.n_units * 2), axis=1)
        assert r_x.data.shape[1] == self.n_units
        assert z_x.data.shape[1] == self.n_units
        assert h_x.data.shape[1] == self.n_units

        r_z_h = self.U_r_z(h)
        r_h, z_h = split_axis(r_z_h, (self.n_units,), axis=1)

        r = sigmoid.sigmoid(r_x + r_h)
        z = sigmoid.sigmoid(z_x + z_h)
        h_bar = tanh.tanh(h_x + self.U(r * h))
        h_new = (1 - z) * h + z * h_bar
        return h_new

    def faster_call2(self, h, x):
        r_z_h_x = self.W_r_z_h(x)

        r_z_h = self.U_r_z(h)

        r_x, z_x, h_x = split_axis(r_z_h_x, (self.n_units, self.n_units * 2), axis=1)
        assert r_x.data.shape[1] == self.n_units
        assert z_x.data.shape[1] == self.n_units
        assert h_x.data.shape[1] == self.n_units

        r_h, z_h = split_axis(r_z_h, (self.n_units,), axis=1)
#         r = sigmoid.sigmoid(r_x + r_h)
#         z = sigmoid.sigmoid(z_x + z_h)
#         h_bar = tanh.tanh(h_x + self.U(sigm_a_plus_b_by_h(r_x, r_h, h)))
#         h_new = (1 - z) * h + z * h_bar
#         return h_new

        return compute_output_GRU(z_x, z_h, h_x, h, self.U(sigm_a_plus_b_by_h_fast(r_x, r_h, h)))

    def faster_call3(self, h, x):
        device = chainer.cuda.get_device(h.data)
        with device:
            stream1 = chainer.cuda.Stream(1)
            stream2 = chainer.cuda.Stream(2)

            chainer.cuda.cublas.setStream(device.cublas_handle, stream1.ptr)
            r_z_h_x = self.W_r_z_h(x)

            chainer.cuda.cublas.setStream(device.cublas_handle, stream2.ptr)
            r_z_h = self.U_r_z(h)

            stream1.synchronize()
            stream2.synchronize()

            r_x, z_x, h_x = split_axis(r_z_h_x, (self.n_units, self.n_units * 2), axis=1)
            assert r_x.data.shape[1] == self.n_units
            assert z_x.data.shape[1] == self.n_units
            assert h_x.data.shape[1] == self.n_units

            r_h, z_h = split_axis(r_z_h, (self.n_units,), axis=1)
    #         r = sigmoid.sigmoid(r_x + r_h)
    #         z = sigmoid.sigmoid(z_x + z_h)
    #         h_bar = tanh.tanh(h_x + self.U(sigm_a_plus_b_by_h(r_x, r_h, h)))
    #         h_new = (1 - z) * h + z * h_bar
    #         return h_new

            return compute_output_GRU(z_x, z_h, h_x, h, self.U(sigm_a_plus_b_by_h_fast(r_x, r_h, h)))

    __call__ = faster_call2

    def initialize_with_classic_implementation(self, gru):
        """ Initialize with the parameters of a non optimized GRU. Useful for testing / model transfer. """
        assert isinstance(gru, chainer.links.connection.GRU)
        self.W_r_z_h.W.data[:self.n_units] = gru.W_r.W.data
        self.W_r_z_h.W.data[self.n_units: 2 * self.n_units] = gru.W_z.W.data
        self.W_r_z_h.W.data[2 * self.n_units:] = gru.W.W.data
        self.W_r_z_h.b.data[:self.n_units] = gru.W_r.b.data
        self.W_r_z_h.b.data[self.n_units: 2 * self.n_units] = gru.W_z.b.data
        self.W_r_z_h.b.data[2 * self.n_units:] = gru.W.b.data

        self.U_r_z.W.data[:self.n_units] = gru.U_r.W.data
        self.U_r_z.W.data[self.n_units:] = gru.U_z.W.data
        self.U_r_z.b.data[:self.n_units] = gru.U_r.b.data
        self.U_r_z.b.data[self.n_units:] = gru.U_z.b.data

        self.U.W.data[...] = gru.U.W.data
        self.U.b.data[...] = gru.U.b.data


import numpy as np
from chainer import Variable, cuda
from chainer import gradient_check
# import chainer.function_hooks
import operator


def commandline():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu")
    parser.add_argument("--fast", type=int)
    parser.add_argument("--fast2", type=int)
    parser.add_argument("--test_perf", default=False, action="store_true")
    parser.add_argument("--test_correctness", default=False, action="store_true")
    parser.add_argument("--mb_size", type=int, default=64)
    parser.add_argument("--h_size", type=int, default=512)
    parser.add_argument("--i_size", type=int, default=512)
    args = parser.parse_args()

    if args.test_perf:
        test_perf(args)

    if args.test_correctness:
        test_correctness(args)


def test_correctness(args):
    ni = 100
    no = 100
    mb_size = 8

    gru_classic = chainer.links.GRU(ni, no)
    my_gru = GRU(ni, no)
    my_gru.initialize_with_classic_implementation(gru_classic)

    x = np.random.randn(mb_size, ni).astype(np.float32)
    st = np.random.randn(mb_size, no).astype(np.float32)

    grad_output = np.random.randn(mb_size, no).astype(np.float32)

    if args.gpu is not None:
        my_gru = my_gru.to_gpu(args.gpu)
        gru_classic = gru_classic.to_gpu(args.gpu)
        x = cuda.to_gpu(x, args.gpu)
        st = cuda.to_gpu(st, args.gpu)
        grad_output = cuda.to_gpu(grad_output, args.gpu)

    x_v = Variable(x)
    x_v2 = Variable(x)

    st_v = Variable(st)
    st_v2 = Variable(st)

    with chainer.function_hooks.PrintHook():
        with chainer.function_hooks.TimerHook() as m:
            st_classic = gru_classic(st_v, x_v)
            print m
            print(m.total_time())
            print sorted(m.call_history, key=operator.itemgetter(1))

        with chainer.function_hooks.TimerHook() as m:
            my_st = my_gru.faster_call2(st_v2, x_v2)
            print m
            print(m.total_time())
            print sorted(m.call_history, key=operator.itemgetter(1))

    print np.max(np.abs(cuda.to_cpu(my_st.data - st_classic.data)))
    gradient_check.assert_allclose(my_st.data, st_classic.data)

    my_st.grad = grad_output
    st_classic.grad = grad_output

    my_gru.zerograds()
    gru_classic.zerograds()

    my_st.backward()
    st_classic.backward()

    gradient_check.assert_allclose(x_v.grad, x_v2.grad)
    gradient_check.assert_allclose(st_v.grad, st_v2.grad)

    gradient_check.assert_allclose(gru_classic.U.W.grad, my_gru.U.W.grad)
    gradient_check.assert_allclose(gru_classic.U.b.grad, my_gru.U.b.grad)


def test_perf(args):
    ni = args.i_size
    no = args.h_size
    mb_size = args.mb_size

    x = np.random.randn(mb_size, ni).astype(np.float32)
    st = np.random.randn(mb_size, no).astype(np.float32)

    if args.fast2 is not None:
        gru_model = GRU2(no, ni)
        if args.gpu is not None:
            gru_model = gru_model.to_gpu(args.gpu)

        if args.fast2 == 1:
            gru = gru_model.faster_call
        elif args.fast2 == 2:
            gru = gru_model.faster_call2
        elif args.fast2 == 3:
            gru = gru_model.faster_call3

    elif args.fast is not None:
        gru_model = GRU(no, ni)
        if args.gpu is not None:
            gru_model = gru_model.to_gpu(args.gpu)
        if args.fast == 0:
            gru = gru_model.classic_call
        elif args.fast == 1:
            gru = gru_model.faster_call
        elif args.fast == 2:
            gru = gru_model.faster_call2
        elif args.fast == 3:
            gru = gru_model.faster_call3
    else:
        gru = chainer.links.GRU(no, ni)
        if args.gpu is not None:
            gru = gru.to_gpu(args.gpu)
#     gru = GRU(no, ni)

    if args.gpu is not None:
        x = cuda.to_gpu(x, args.gpu)
        st = cuda.to_gpu(st, args.gpu)

    print "start"
    for _ in xrange(300):
        x_v = Variable(x)
        st_v = Variable(st)
        for _ in xrange(20):
            st_v = gru(st_v, x_v)
        loss = chainer.functions.sum(st_v)
        loss.backward()
    print "done"


def test2():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu")
    args = parser.parse_args()

    ni = 10
    no = 10
    mb_size = 32
    import numpy as np
    from chainer import Variable, cuda

    x = np.random.randn(mb_size, ni).astype(np.float32)
    st = np.random.randn(mb_size, no).astype(np.float32)

    gru_model = GRU(no, ni)
#     gru = GRU(no, ni)

    if args.gpu is not None:
        x = cuda.to_gpu(x, args.gpu)
        st = cuda.to_gpu(st, args.gpu)
        gru_model = gru_model.to_gpu(args.gpu)

    x_v = Variable(x)
    st_v = Variable(st)

    gru_model.zerograds()
#     r1 = gru_model.classic_call(st_v, x_v)
    r2 = gru_model.faster_call(st_v, x_v)
    r3 = gru_model.faster_call2(st_v, x_v)

    print r2.data - r2.data
    print r3.data - r3.data

    from chainer import gradient_check

    r3 = r2
    r3.grad = np.random.randn(mb_size, no).astype(np.float32)
    r3.backward()

    def f():
        return (gru_model.faster_call(st_v, x_v).data,)

    g_st, g_x = gradient_check.numerical_grad(f, (st_v.data, x_v.data), (r3.grad,))
    print np.max(np.abs(g_st - st_v.grad))
    print np.max(np.abs(g_x - x_v.grad))
    gradient_check.assert_allclose(g_st, st_v.grad, atol=1e-3)
    gradient_check.assert_allclose(g_x, x_v.grad, atol=1e-3)


if __name__ == '__main__':
    commandline()
