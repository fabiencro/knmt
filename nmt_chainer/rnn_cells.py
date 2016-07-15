#!/usr/bin/env python
"""rnn_cells.py: Wrappers around various RNN Cell types"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

from _collections import defaultdict
import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import math, random

from utils import ortho_init

import logging
logging.basicConfig()
log = logging.getLogger("rnns:cells")
log.setLevel(logging.INFO)

# L.GRU = L.FastGRU
import faster_gru

class LSTMCell(Chain):
    def __init__(self, in_size, out_size):
        log.info("Creating LSTMCell(%i, %i)"%(in_size, out_size))
        super(LSTMCell, self).__init__(
            lstm = L.StatelessLSTM(in_size, out_size),
        )
        self.add_param("initial_state", (1, out_size))
        self.initial_state.data[...] = self.xp.random.randn(out_size)
        self.add_persistent("initial_cell", self.xp.zeros((1, out_size), dtype = self.xp.float32))
        self.out_size = out_size
        self.in_size = in_size
        
    def get_initial_states(self, mb_size):
        mb_initial_state = F.broadcast_to(F.reshape(self.initial_state, (1, self.out_size)), (mb_size, self.out_size))
        mb_initial_cell = Variable(self.xp.broadcast_to(self.initial_cell, (mb_size, self.out_size)), volatile = "auto")
        return (mb_initial_cell, mb_initial_state)
        
    def __call__(self, prev_states, x_in, mode = "test"):
        assert mode in "test train".split()
        prev_cell, prev_state = prev_states
        new_cell, new_state = self.lstm(prev_cell, prev_state, x_in)
        return new_cell, new_state
    
    def get_nb_states(self):
        return 2 # state + cell
    
class StackedCell(ChainList):
    def __init__(self, in_size, out_size, cell_type = LSTMCell, nb_stacks = 2, dropout = 0.5):
        log.info("Creating StackedCell(%i, %i) X %i"%(in_size, out_size, nb_stacks))
        super(StackedCell, self).__init__()
        self.nb_of_states = []
        
        cell0 = cell_type(in_size, out_size)
        self.add_link(cell0)
        self.nb_of_states.append(cell0.get_nb_states())
        
        for i in xrange(1, nb_stacks):
            cell = cell_type(out_size, out_size)
            self.add_link(cell)
            self.nb_of_states.append(cell.get_nb_states())
        assert len(self) == nb_stacks
        
        self.dropout = dropout
    
    def get_initial_states(self, mb_size):
        res = []
        for i in xrange(len(self)):
            res += list(self[i].get_initial_states(mb_size))
        return tuple(res)
            
    def __call__(self, prev_states, x_in, mode = "test"):
        assert mode in "test train".split()
        input_below = x_in
        states_cursor = 0
        res = []
        for i in xrange(len(self)):
            if self.dropout is not None:
                input_below = F.dropout(input_below, ratio = self.dropout, train = (mode == "train"))
            new_states = self[i](prev_states[states_cursor:states_cursor + self.nb_of_states[i]], input_below,
                                 mode = mode)
            states_cursor += self.nb_of_states[i]
            input_below = new_states[-1]
            res += list(new_states)
        return res
    
    
        
            
# class DoubleGRU(Chain):
#     def __init__(self, H, I):
#         log.info("using double GRU")
#         self.H1 = H/2
#         self.H2 = H - self.H1
#         super(DoubleGRU, self).__init__(
#             gru1 = faster_gru.GRU(self.H1, I),
#             gru2 = faster_gru.GRU(self.H2, self.H1)
#         )
#         
#     def __call__(self, prev_state, inpt):
#         prev_state1, prev_state2 = F.split_axis(prev_state, (self.H1,), axis = 1)
#         
#         prev_state1 = self.gru1(prev_state1, inpt)
#         prev_state2 = self.gru2(prev_state2, prev_state1)
#         
#         return F.concat((prev_state1, prev_state2), axis = 1)
    
cell_dict = {
             "lstm": LSTMCell,
             "dlstm": StackedCell
             }
        
        