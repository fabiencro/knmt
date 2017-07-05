import math

import numpy

from chainer import cuda
from chainer import optimizer


class ScheduledAdam(optimizer.GradientMethod):

    """Adam optimization algorithm scheduled as in https://arxiv.org/abs/1706.03762

    """

    def __init__(self, d_model=512, beta1=0.9, beta2=0.98, eps=1e-9, warmup_steps=4000):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.num_step = 1

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            state['m'] = xp.zeros_like(param.data)
            state['v'] = xp.zeros_like(param.data)

    def update_one_cpu(self, param, state):
        m, v = state['m'], state['v']
        grad = param.grad

        m += (1 - self.beta1) * (grad - m)
        v += (1 - self.beta2) * (grad * grad - v)
        param.data -= self.lr * m / (numpy.sqrt(v) + self.eps)
        self.num_step += 1

    def update_one_gpu(self, param, state):
        cuda.elementwise(
            'T grad, T lr, T one_minus_beta1, T one_minus_beta2, T eps',
            'T param, T m, T v',
            '''m += one_minus_beta1 * (grad - m);
               v += one_minus_beta2 * (grad * grad - v);
               param -= lr * m / (sqrt(v) + eps);''',
            'adam')(param.grad, self.lr, 1 - self.beta1, 1 - self.beta2,
                    self.eps, param.data, state['m'], state['v'])
        self.num_step += 1

    def compute_alpha(self):
        return min(1.0/math.sqrt(self.num_step), self.num_step*math.pow(self.warmup_steps, -1.5)) / math.sqrt(self.d_model)

    @property
    def lr(self):
        alpha = self.compute_alpha()
        fix1 = 1. - self.beta1 ** self.t
        fix2 = 1. - self.beta2 ** self.t
        return alpha * math.sqrt(fix2) / fix1
    
