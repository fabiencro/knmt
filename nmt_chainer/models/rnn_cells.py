#!/usr/bin/env python
"""rnn_cells.py: Wrappers around various RNN Cell types"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

from collections import Counter
import chainer
from chainer import cuda, Variable
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer import initializers

import logging
logging.basicConfig()
log = logging.getLogger("rnns:cells")
log.setLevel(logging.INFO)

# L.GRU = L.FastGRU
import nmt_chainer.utilities.faster_gru as faster_gru


class GRUCell(Chain):
    def __init__(self, in_size, out_size, init=None, inner_init=None, bias_init=None):
        log.info("Creating GRUCell(%i, %i)" % (in_size, out_size))
        super(GRUCell, self).__init__(
            gru=L.GRU(out_size, in_size, init=init, inner_init=inner_init, bias_init=bias_init),
        )
        self.add_param("initial_state", (1, out_size))
        self.initial_state.data[...] = self.xp.random.randn(out_size)
        self.out_size = out_size
        self.in_size = in_size

    def get_initial_states(self, mb_size):
        mb_initial_state = F.broadcast_to(F.reshape(
            self.initial_state, (1, self.out_size)), (mb_size, self.out_size))
        return (mb_initial_state,)

    def __call__(self, prev_states, x_in, mode="test"):
        assert mode in "test train".split()
        assert len(prev_states) == 1
        prev_state = prev_states[0]
        new_state = self.gru(prev_state, x_in)
        return (new_state,)

    def get_nb_states(self):
        return 1


class FastGRUCell(Chain):
    def __init__(self, in_size, out_size, init=None, bias_init=None):
        log.info("Creating GRUCell(%i, %i)" % (in_size, out_size))
        super(FastGRUCell, self).__init__(
            gru=faster_gru.GRU(out_size, in_size, init=init, bias_init=bias_init),
        )
        self.add_param("initial_state", (1, out_size))
        self.initial_state.data[...] = self.xp.random.randn(out_size)
        self.out_size = out_size
        self.in_size = in_size

    def get_initial_states(self, mb_size):
        mb_initial_state = F.broadcast_to(F.reshape(
            self.initial_state, (1, self.out_size)), (mb_size, self.out_size))
        return (mb_initial_state,)

    def __call__(self, prev_states, x_in, mode="test"):
        assert mode in "test train".split()
        assert len(prev_states) == 1
        prev_state = prev_states[0]
        new_state = self.gru(prev_state, x_in)
        return (new_state,)

    def get_nb_states(self):
        return 1


class LSTMCell(Chain):
    def __init__(self, in_size, out_size, lateral_init=None, upward_init=None, bias_init=None, forget_bias_init=None):
        log.info("Creating LSTMCell(%i, %i)" % (in_size, out_size))
        super(LSTMCell, self).__init__(
            lstm=L.StatelessLSTM(in_size, out_size,
                                 lateral_init=lateral_init, upward_init=upward_init,
                                 bias_init=bias_init, forget_bias_init=forget_bias_init)
        )
        self.add_param("initial_state", (1, out_size))
        self.initial_state.data[...] = self.xp.random.randn(out_size)
        self.add_persistent("initial_cell", self.xp.zeros((1, out_size), dtype=self.xp.float32))
        self.out_size = out_size
        self.in_size = in_size

    def get_initial_states(self, mb_size):
        mb_initial_state = F.broadcast_to(F.reshape(self.initial_state, (1, self.out_size)), (mb_size, self.out_size))
        mb_initial_cell = Variable(self.xp.broadcast_to(self.initial_cell, (mb_size, self.out_size)), volatile="auto")
        return (mb_initial_cell, mb_initial_state)

    def __call__(self, prev_states, x_in, mode="test"):
        assert mode in "test train".split()
        prev_cell, prev_state = prev_states
        new_cell, new_state = self.lstm(prev_cell, prev_state, x_in)
        return new_cell, new_state

    def get_nb_states(self):
        return 2  # state + cell


class GatedLSTMCell(Chain):
    def __init__(self, in_size, out_size, lateral_init=None, upward_init=None, bias_init=None, forget_bias_init=None):
        log.info("Creating GatedLSTMCell(%i, %i)" % (in_size, out_size))
        assert in_size == out_size

        super(GatedLSTMCell, self).__init__(
            lstm=L.StatelessLSTM(in_size, out_size, lateral_init=lateral_init, upward_init=upward_init, bias_init=bias_init, forget_bias_init=forget_bias_init),
            gate_w=L.Linear(in_size, in_size)
        )
        self.add_param("initial_state", (1, out_size))
        self.initial_state.data[...] = self.xp.random.randn(out_size)
        self.add_persistent("initial_cell", self.xp.zeros((1, out_size), dtype=self.xp.float32))
        self.add_persistent("initial_output", self.xp.zeros((1, out_size), dtype=self.xp.float32))
        self.out_size = out_size
        self.in_size = in_size

    def get_initial_states(self, mb_size):
        mb_initial_state = F.broadcast_to(F.reshape(self.initial_state, (1, self.out_size)), (mb_size, self.out_size))
        mb_initial_cell = Variable(self.xp.broadcast_to(self.initial_cell, (mb_size, self.out_size)), volatile="auto")
        mb_initial_output = Variable(self.xp.broadcast_to(self.initial_output, (mb_size, self.out_size)), volatile="auto")
        return (mb_initial_cell, mb_initial_state, mb_initial_output)

    def __call__(self, prev_states, x_in, mode="test"):
        assert mode in "test train".split()
        prev_cell, prev_state, old_output = prev_states
        new_cell, new_state = self.lstm(prev_cell, prev_state, x_in)

        passthrough_gate_state = F.sigmoid(self.gate_w(x_in))
        output = passthrough_gate_state * x_in + (1 - passthrough_gate_state) * new_state

        return new_cell, new_state, output

    def get_nb_states(self):
        return 3  # state + cell + gated_output


class StackedCell(ChainList):
    def __init__(self, in_size, out_size, cell_type=LSTMCell, nb_stacks=2,
                 dropout=0.5, residual_connection=False, no_dropout_on_input=False,
                 no_residual_connection_on_output=False, no_residual_connection_on_input=False,
                 init=None, inner_init=None, lateral_init=None, upward_init=None, bias_init=None, forget_bias_init=None):
        nb_stacks = nb_stacks or 2
        cell_type = cell_type or LSTMCell

        log.info("Creating StackedCell(%i, %i) X %i" % (in_size, out_size, nb_stacks))
        super(StackedCell, self).__init__()
        self.nb_of_states = []

        if cell_type in (LSTMCell, GatedLSTMCell):
            cell0 = cell_type(in_size, out_size, lateral_init=lateral_init, upward_init=upward_init, bias_init=bias_init, forget_bias_init=forget_bias_init)
        elif cell_type == GRUCell:
            cell0 = cell_type(in_size, out_size, init=init, inner_init=inner_init, bias_init=bias_init)
        elif cell_type == FastGRUCell:
            cell0 = cell_type(in_size, out_size, init=init, bias_init=bias_init)
        else:
            raise ValueError("Unsupported cell_type={0}".format(cell_type))
        self.add_link(cell0)
        self.nb_of_states.append(cell0.get_nb_states())

        for i in xrange(1, nb_stacks):
            if cell_type in (LSTMCell, GatedLSTMCell):
                cell = cell_type(out_size, out_size, lateral_init=lateral_init, upward_init=upward_init, bias_init=bias_init, forget_bias_init=forget_bias_init)
            elif cell_type == GRUCell:
                cell = cell_type(out_size, out_size, init=init, inner_init=inner_init, bias_init=bias_init)
            elif cell_type == FastGRUCell:
                cell = cell_type(out_size, out_size, init=init, bias_init=bias_init)
            else:
                raise ValueError("Unsupported cell_type={0}".format(cell_type))
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

    def __call__(self, prev_states, x_in, mode="test"):
        assert mode in "test train".split()
        input_below = x_in
        states_cursor = 0
        res = []
        for i in xrange(len(self)):
            if self.dropout is not None and not (self.no_dropout_on_input and i == 0):
                input_below = F.dropout(input_below, ratio=self.dropout, train=(mode == "train"))
            new_states = self[i](prev_states[states_cursor:states_cursor + self.nb_of_states[i]], input_below,
                                 mode=mode)
            states_cursor += self.nb_of_states[i]

            if (self.residual_connection and
                not (i == len(self) - 1 and self.no_residual_connection_on_output) and
                    not (i == 0 and self.no_residual_connection_on_input)):
                input_below = new_states[-1] + input_below
            else:
                input_below = new_states[-1]

            res += list(new_states)
        return res


class NStepsCell(Chain):
    def __init__(self, in_size, out_size, nb_stacks, dropout, **kwds):
        super(NStepsCell, self).__init__(
            nstep_lstm=L.NStepLSTM(nb_stacks, in_size, out_size, dropout)
        )
        self.add_param("initial_state", (nb_stacks, 1, out_size))
        self.initial_state.data[...] = self.xp.random.randn(nb_stacks, 1, out_size)
        self.add_persistent("initial_cell", self.xp.zeros((nb_stacks, 1, out_size), dtype=self.xp.float32))

        self.nb_stacks = nb_stacks
        self.out_size = out_size
        self.in_size = in_size

    def get_initial_states(self, mb_size):
        mb_initial_state = F.broadcast_to(self.initial_state, (self.nb_stacks, mb_size, self.out_size))
        mb_initial_cell = Variable(self.xp.broadcast_to(self.initial_cell, (self.nb_stacks, mb_size, self.out_size)), volatile="auto")
        return (mb_initial_cell, mb_initial_state)

    def apply_to_seq(self, seq_list, mode="test"):
        assert mode in "test train".split()
        mb_size = len(seq_list)
        mb_initial_cell, mb_initial_state = self.get_initial_states(mb_size)
        return self.nstep_lstm(mb_initial_cell, mb_initial_state, seq_list, train=mode == "train")


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
    "dlstm": StackedCell,  # for backward compatibility
    "stack": StackedCell,
    "slow_gru": GRUCell,
    "gru": FastGRUCell,
    "glstm": GatedLSTMCell,
    "nsteps": NStepsCell
}

# has_dropout = set(["dlno_dropout_on_input = Falsestm"])


cell_description_keywords = {
    "dropout": float,
    "nb_stacks": int,
    "sub_cell_type": lambda k: cell_dict[k],
    "residual_connection": int,
    "no_dropout_on_input": int,
    "no_residual_connection_on_output": int,
    "no_residual_connection_on_input": int,
    "init_type": str,
    "inner_init_type": str,
    "lateral_init_type": str,
    "upward_init_type": str,
    "bias_init_type": str,
    "forget_bias_init_type": str,
    "init_scale": float,
    "inner_init_scale": float,
    "lateral_init_scale": float,
    "upward_init_scale": float,
    "bias_init_scale": float,
    "forget_bias_init_scale": float,
    "init_fillvalue": float,
    "inner_init_fillvalue": float,
    "lateral_init_fillvalue": float,
    "upward_init_fillvalue": float,
    "bias_init_fillvalue": float,
    "forget_bias_init_fillvalue": float
}


def create_cell_model_from_string(model_str):
    components = model_str.split(",")
    type_str = components[0]

    keywords = {}
    for comp in components[1:]:
        assert ":" in comp
        key, val = comp.split(":")
        if key in cell_description_keywords:
            keywords[key] = cell_description_keywords[key](val)
        else:
            raise ValueError("bad cell parameter: %s (possible parameters: %s)" %
                             (comp, " ".join(cell_description_keywords.keys())))

    return create_cell_model(type_str, **keywords)


def create_cell_model_from_config(config):
    type_str = config["cell_type"]
    keywords = dict((k, config[k]) for k in config if k != "cell_type")
    return create_cell_model(type_str, **keywords)


def create_initializer(init_type, scale=None, fillvalue=None):
    if init_type == 'identity':
        return initializers.Identity() if scale is None else initializers.Identity(scale=scale)
    if init_type == 'constant':
        return initializers.Constant(fillvalue)
    if init_type == 'zero':
        return initializers.Zero()
    if init_type == 'one':
        return initializers.One()
    if init_type == 'normal':
        return initializers.Normal() if scale is None else initializers.Normal(scale)
    if init_type == 'glorotNormal':
        return initializers.GlorotNormal() if scale is None else initializers.GlorotNormal(scale)
    if init_type == 'heNormal':
        return initializers.HeNormal() if scale is None else initializers.HeNormal(scale)
    if init_type == 'orthogonal':
        return initializers.Orthogonal(
            scale) if scale is None else initializers.Orthogonal(scale)
    if init_type == 'uniform':
        return initializers.Uniform(
            scale) if scale is None else initializers.Uniform(scale)
    if init_type == 'leCunUniform':
        return initializers.LeCunUniform(
            scale) if scale is None else initializers.LeCunUniform(scale)
    if init_type == 'glorotUniform':
        return initializers.GlorotUniform(
            scale) if scale is None else initializers.GlorotUniform(scale)
    if init_type == 'heUniform':
        return initializers.HeUniform(
            scale) if scale is None else initializers.HeUniform(scale)
    raise ValueError("Unknown initializer type: {0}".format(init_type))


def create_initializer_table(keywords):
    initializers = {}
    init_params = [k for k in keywords if 'init_' in k]
    init_kind = Counter(map((lambda str: str[:str.index('init') + 4]), init_params))
    for prefix in init_kind:
        init_type = None
        key = "{0}_type".format(prefix)
        if key in keywords:
            init_type = keywords[key]

        init_scale = None
        key = "{0}_scale".format(prefix)
        if key in keywords:
            init_scale = keywords[key]

        init_fillvalue = None
        key = "{0}_fillvalue".format(prefix)
        if key in keywords:
            init_fillvalue = keywords[key]

        initializer = create_initializer(init_type, init_scale, init_fillvalue)
        initializers[prefix] = initializer
    return initializers


def create_cell_model(type_str, **cell_parameters):
    initializers = create_initializer_table(cell_parameters)

    init = initializers['init'] if 'init' in initializers else None
    inner_init = initializers['inner_init'] if 'inner_init' in initializers else None
    lateral_init = initializers['lateral_init'] if 'lateral_init' in initializers else None
    upward_init = initializers['upward_init'] if 'upward_init' in initializers else None
    bias_init = initializers['bias_init'] if 'bias_init' in initializers else None
    forget_bias_init = initializers['forget_bias_init'] if 'forget_bias_init' in initializers else None

    if type_str not in cell_dict:
        raise ValueError("bad cell type: %s (possible types: %s)" %
                         (type_str, " ".join(cell_dict.keys())))
    cell_type = cell_dict[type_str]
    if type_str == "dlstm" or type_str == "stack" or type_str == "nsteps":
        def instantiate(in_size, out_size):
            # return cell_type(in_size, out_size, **cell_parameters)
            params = {}
            if 'sub_cell_type' in cell_parameters:
                params['cell_type'] = cell_parameters['sub_cell_type']
            if 'nb_stacks' in cell_parameters:
                params['nb_stacks'] = cell_parameters['nb_stacks']
            if 'dropout' in cell_parameters:
                params['dropout'] = cell_parameters['dropout']
            if 'residual_connection' in cell_parameters:
                params['residual_connection'] = cell_parameters['residual_connection']
            if 'no_dropout_on_input' in cell_parameters:
                params['no_dropout_on_input'] = cell_parameters['no_dropout_on_input']
            if 'no_residual_connection_on_output' in cell_parameters:
                params['no_residual_connection_on_output'] = cell_parameters['no_residual_connection_on_output']
            if 'no_residual_connection_on_input' in cell_parameters:
                params['no_residual_connection_on_input'] = cell_parameters['no_residual_connection_on_input']
            params['init'] = init
            params['inner_init'] = inner_init
            params['lateral_init'] = lateral_init
            params['upward_init'] = upward_init
            params['bias_init'] = bias_init
            params['forget_bias_init'] = forget_bias_init
            return cell_type(in_size, out_size, **params)

    else:
        def instantiate(in_size, out_size):
            if type_str in ("lstm", "glstm"):
                return cell_type(in_size, out_size, lateral_init=lateral_init, upward_init=upward_init, bias_init=bias_init, forget_bias_init=forget_bias_init)
            elif type_str == "slow_gru":
                return cell_type(in_size, out_size, init=init, inner_init=inner_init, bias_init=bias_init)
            elif type_str == "gru":
                return cell_type(in_size, out_size, init=init, bias_init=bias_init)
            else:
                raise ValueError("Unsupported cell_type={0}".format(cell_type))
    instantiate.meta_data_cell_type = cell_type
    return instantiate
