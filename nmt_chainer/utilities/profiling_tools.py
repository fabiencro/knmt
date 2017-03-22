import chainer
import chainer.function_hooks
from chainer import function
from chainer import cuda
from collections import defaultdict
import time
import numpy


def function_namer(function, in_data):
    in_shapes = []
    for in_elem in in_data:
        if hasattr(in_elem, "shape"):
            in_shapes.append("@%r" % (in_elem.shape,))
        elif isinstance(in_elem, (int, float, str)):
            in_shapes.append(in_elem)
        else:
            in_shapes.append("OTHER")
    if isinstance(function, chainer.functions.array.split_axis.SplitAxis):
        in_shapes.append(("s/a", function.indices_or_sections, function.axis))

    in_shapes = tuple(in_shapes)
    return (function.__class__, in_shapes)


class TimerElem(object):
    def __init__(self):
        self.fwd = 0
        self.bwd = 0
        self.total = 0
        self.nb_fwd = 0
        self.nb_bwd = 0

    def add_fwd(self, fwd):
        self.fwd += fwd
        self.total += fwd
        self.nb_fwd += 1

    def add_bwd(self, bwd):
        self.bwd += bwd
        self.total += bwd
        self.nb_bwd += 1

    def __repr__(self):
        return "<T:%f F[%i]:%f B[%i]:%f>" % (self.total, self.nb_fwd, self.fwd, self.nb_bwd, self.bwd)
    __str__ = __repr__


class MyTimerHook(function.FunctionHook):
    """Function hook for measuring elapsed time of functions.

    Attributes:
        call_history: List of measurement results. It consists of pairs of
            the function that calls this hook and the elapsed time
            the function consumes.
    """

    name = 'TimerHook'

    def __init__(self):
        self.call_times_per_classes = defaultdict(TimerElem)

    def __exit__(self, *_):
        print self
        self.print_sorted()
        print "total time:"
        print(self.total_time())
        super(MyTimerHook, self).__exit__(*_)

    def _preprocess(self):
        if self.xp == numpy:
            self.start = time.time()
        else:
            self.start = cuda.Event()
            self.stop = cuda.Event()
            self.start.record()

    def forward_preprocess(self, function, in_data):
        self.xp = cuda.get_array_module(*in_data)
        self._preprocess()

    def backward_preprocess(self, function, in_data, out_grad):
        self.xp = cuda.get_array_module(*(in_data + out_grad))
        self._preprocess()

    def _postprocess(self, function_repr, bwd=False):
        if self.xp == numpy:
            self.stop = time.time()
            elapsed_time = self.stop - self.start
        else:
            self.stop.record()
            self.stop.synchronize()
            # Note that `get_elapsed_time` returns result in milliseconds
            elapsed_time = cuda.cupy.cuda.get_elapsed_time(
                self.start, self.stop) / 1000.0
        if bwd:
            self.call_times_per_classes[function_repr].add_bwd(elapsed_time)
        else:
            self.call_times_per_classes[function_repr].add_fwd(elapsed_time)
#         self.call_history.append((function, elapsed_time))

    def forward_postprocess(self, function, in_data):
        xp = cuda.get_array_module(*in_data)
        assert xp == self.xp
        self._postprocess(function_namer(function, in_data))

    def backward_postprocess(self, function, in_data, out_grad):
        xp = cuda.get_array_module(*(in_data + out_grad))
        assert xp == self.xp
        self._postprocess(function_namer(function, in_data), bwd=True)

    def total_time(self):
        """Returns total elapsed time in seconds."""
        return sum(
            t.total for (
                _,
                t) in self.call_times_per_classes.iteritems())

    def print_sorted(self):
        for name, time in sorted(self.call_times_per_classes.items(), key=lambda x: x[1].total):
            print name, time
