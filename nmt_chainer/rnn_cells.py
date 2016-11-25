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


class GRUCell(Chain):
    def __init__(self, in_size, out_size):
        log.info("Creating GRUCell(%i, %i)"%(in_size, out_size))
        super(GRUCell, self).__init__(
            gru = L.GRU(out_size, in_size),
        )
        self.add_param("initial_state", (1, out_size))
        self.initial_state.data[...] = self.xp.random.randn(out_size)
        self.out_size = out_size
        self.in_size = in_size
          
    def get_initial_states(self, mb_size):
        mb_initial_state = F.broadcast_to(F.reshape(self.initial_state, (1, self.out_size)), (mb_size, self.out_size))
        return (mb_initial_state,)
    
    def __call__(self, prev_states, x_in, mode = "test"):      
        assert mode in "test train".split()
        assert len(prev_states) == 1
        prev_state = prev_states[0]
        new_state = self.gru(prev_state, x_in)
        return (new_state,)
    
    def get_nb_states(self):
        return 1
    
class FastGRUCell(Chain):
    def __init__(self, in_size, out_size):
        log.info("Creating GRUCell(%i, %i)"%(in_size, out_size))
        super(FastGRUCell, self).__init__(
            gru = faster_gru.GRU(out_size, in_size),
        )
        self.add_param("initial_state", (1, out_size))
        self.initial_state.data[...] = self.xp.random.randn(out_size)
        self.out_size = out_size
        self.in_size = in_size
          
    def get_initial_states(self, mb_size):
        mb_initial_state = F.broadcast_to(F.reshape(self.initial_state, (1, self.out_size)), (mb_size, self.out_size))
        return (mb_initial_state,)
    
    def __call__(self, prev_states, x_in, mode = "test"):      
        assert mode in "test train".split()
        assert len(prev_states) == 1
        prev_state = prev_states[0]
        new_state = self.gru(prev_state, x_in)
        return (new_state,)
    
    def get_nb_states(self):
        return 1

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
    
    
class GatedLSTMCell(Chain):
    def __init__(self, in_size, out_size):
        log.info("Creating GatedLSTMCell(%i, %i)"%(in_size, out_size))
        assert in_size == out_size
        
        super(GatedLSTMCell, self).__init__(
            lstm = L.StatelessLSTM(in_size, out_size),
            gate_w = L.Linear(in_size, in_size)
        )
        self.add_param("initial_state", (1, out_size))
        self.initial_state.data[...] = self.xp.random.randn(out_size)
        self.add_persistent("initial_cell", self.xp.zeros((1, out_size), dtype = self.xp.float32))
        self.add_persistent("initial_output", self.xp.zeros((1, out_size), dtype = self.xp.float32))
        self.out_size = out_size
        self.in_size = in_size
        
    def get_initial_states(self, mb_size):
        mb_initial_state = F.broadcast_to(F.reshape(self.initial_state, (1, self.out_size)), (mb_size, self.out_size))
        mb_initial_cell = Variable(self.xp.broadcast_to(self.initial_cell, (mb_size, self.out_size)), volatile = "auto")
        mb_initial_output = Variable(self.xp.broadcast_to(self.initial_output, (mb_size, self.out_size)), volatile = "auto")
        return (mb_initial_cell, mb_initial_state, mb_initial_output)
        
    def __call__(self, prev_states, x_in, mode = "test"):
        assert mode in "test train".split()
        prev_cell, prev_state, old_output = prev_states
        new_cell, new_state = self.lstm(prev_cell, prev_state, x_in)
        
        passthrough_gate_state = F.sigmoid(self.gate_w(x_in))
        output = passthrough_gate_state * x_in + (1-passthrough_gate_state) * new_state
        
        return new_cell, new_state, output
    
    def get_nb_states(self):
        return 3 # state + cell + gated_output
    
class StackedCell(ChainList):
    def __init__(self, in_size, out_size, cell_type = LSTMCell, nb_stacks = 2, 
                 dropout = 0.5, residual_connection = False, no_dropout_on_input = False,
                 no_residual_connection_on_output = False, no_residual_connection_on_input = False):
        nb_stacks = nb_stacks or 2
        cell_type = cell_type or LSTMCell
        
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
        self.residual_connection = residual_connection
        self.no_residual_connection_on_output = no_residual_connection_on_output
        self.no_residual_connection_on_input = no_residual_connection_on_input
        self.no_dropout_on_input = no_dropout_on_input
        
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
            if self.dropout is not None and not (self.no_dropout_on_input and i == 0):
                input_below = F.dropout(input_below, ratio = self.dropout, train = (mode == "train"))
            new_states = self[i](prev_states[states_cursor:states_cursor + self.nb_of_states[i]], input_below,
                                 mode = mode)
            states_cursor += self.nb_of_states[i]
            
            if (self.residual_connection 
                    and not (i == len(self) -1 and self.no_residual_connection_on_output)
                    and not (i == 0 and self.no_residual_connection_on_input)):
                input_below = new_states[-1] + input_below
            else:
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
             "dlstm": StackedCell, # for backward compatibility
             "stack": StackedCell,
             "slow_gru": GRUCell,
             "gru": FastGRUCell,
             "glstm": GatedLSTMCell
             }

# has_dropout = set(["dlno_dropout_on_input = Falsestm"])


cell_description_keywords = {
    "dropout": float,
    "nb_stacks": int,
    "sub_cell_type": lambda k:cell_dict[k],
    "residual_connection": int,
    "no_dropout_on_input": int,
    "no_residual_connection_on_output": int,
    "no_residual_connection_on_input": int,
    }

def create_cell_model_from_string(model_str):
    components = model_str.split(",")
    type_str = components[0]
    
    keywords = {}
    for comp in components[1:]:
        assert ":" in  comp
        key, val = comp.split(":")
        if key in cell_description_keywords:
            keywords[key] = cell_description_keywords[key](val)
        else:
            raise ValueError("bad cell parameter: %s (possible parameters: %s)"%
                             (comp, " ".join(cell_description_keywords.keys())))
    return create_cell_model(type_str, **keywords)

def create_cell_model(type_str, **cell_parameters):
    if type_str not in cell_dict:
        raise ValueError("bad cell type: %s (possible types: %s)"%
                             (type_str, " ".join(cell_dict.keys())))
    cell_type = cell_dict[type_str]
    if type_str == "dlstm" or type_str == "stack":
        def instantiate(in_size, out_size):
            return cell_type(in_size, out_size, **cell_parameters)        
    else:
        if len(cell_parameters) != 0:
            raise ValueError("unexpected cell parameters: %r" % cell_parameters)
        def instantiate(in_size, out_size):
            return cell_type(in_size, out_size)
    return instantiate
        
        
        
        