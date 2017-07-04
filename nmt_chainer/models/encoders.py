#!/usr/bin/env python
"""encoders.py: Implementation of RNNSearch in Chainer"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import chainer
from chainer import cuda, Variable
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import rnn_cells

from nmt_chainer.utilities.utils import ortho_init

import logging
logging.basicConfig()
log = logging.getLogger("rnns:encoders")
log.setLevel(logging.INFO)


def make_encoder(Vi, Ei, Hi, init_orth, use_bn_length, cell_type):
    if hasattr(cell_type, "meta_data_cell_type") and cell_type.meta_data_cell_type == rnn_cells.NStepsCell:
        return EncoderNSteps(Vi, Ei, Hi,
                             cell_type=cell_type)
    else:
        return Encoder(Vi, Ei, Hi, init_orth=init_orth, use_bn_length=use_bn_length,
                       cell_type=cell_type)


class EncoderNSteps(Chain):
    """ Chain that encode a sequence.
        The __call_ takes 2 parameters: sequence and mask.
        mask and length should be 2 python lists of same length #length.

        sequence should be a python list of Chainer Variables wrapping a numpy/cupy array of shape (mb_size,) and type int32 each.
            -- where mb_size is the minibatch size
        sequence[i].data[j] should be the jth element of source sequence number i, or a padding value if the sequence number i is
            shorter than j.

        mask should be a python list of Chainer Variables wrapping a numpy/cupy array of shape (mb_size,) and type bool each.
        mask[i].data[j] should be True if and only if sequence[i].data[j] is not a padding value.

        Return a chainer variable of shape (mb_size, #length, 2*Hi) and type float32
    """

    def __init__(self, Vi, Ei, Hi, cell_type):
        gru_f = cell_type(Ei, Hi)
        gru_b = cell_type(Ei, Hi)

        log.info("constructing encoder [%s]" % (cell_type,))
        super(EncoderNSteps, self).__init__(
            emb=L.EmbedID(Vi, Ei),
            #             gru_f = L.GRU(Hi, Ei),
            #             gru_b = L.GRU(Hi, Ei)

            gru_f=gru_f,
            gru_b=gru_b
        )
        self.Hi = Hi

    def __call__(self, sequence, mask, mode="test"):
        assert mode in "test train".split()

        mb_size = sequence[0].data.shape[0]
        max_length_size = len(sequence)

        seq_length = [None] * mb_size
        for num_seq in range(mb_size):
            seq_length[num_seq] = len(sequence) - sum(not mask[i][num_seq] for i in range(len(mask)))

        de_batched_seq = []
        for num_seq in range(mb_size):
            de_batched_seq.append(self.xp.empty((seq_length[num_seq],), dtype=self.xp.int32))
            for i in xrange(seq_length[num_seq]):
                de_batched_seq[-1][i] = sequence[i].data[num_seq]
            de_batched_seq[-1] = Variable(de_batched_seq[-1], volatile="auto")

        embedded_seq = []
        for elem in de_batched_seq:
            embedded_seq.append(self.emb(elem))

        hx, cx, forward_seq = self.gru_f.apply_to_seq(embedded_seq, mode=mode)

        reversed_embedded_seq = []
        for elem in de_batched_seq:
            reversed_embedded_seq.append(self.emb(elem[::-1]))

        hxb, cxb, backward_seq = self.gru_b.apply_to_seq(reversed_embedded_seq, mode=mode)

        assert len(backward_seq) == len(forward_seq) == mb_size

        res = []
        for num_seq in xrange(mb_size):
            assert backward_seq[num_seq].data.shape[0] == forward_seq[num_seq].data.shape[0]
            fb_concatenated = F.concat(
                (forward_seq[num_seq], backward_seq[num_seq][::-1]), 1)
            if forward_seq[num_seq].data.shape[0] < max_length_size:
                pad_length = max_length_size - forward_seq[num_seq].data.shape[0]
                fb_concatenated = F.concat((fb_concatenated,
                                            self.xp.zeros((pad_length, self.Hi * 2), dtype=self.xp.float32)),
                                           axis=0)

            res.append(F.reshape(fb_concatenated, (1, max_length_size, self.Hi * 2)))
        return F.concat(res, axis=0)


class Encoder(Chain):
    """ Chain that encode a sequence.
        The __call_ takes 2 parameters: sequence and mask.
        mask and length should be 2 python lists of same length #length.

        sequence should be a python list of Chainer Variables wrapping a numpy/cupy array of shape (mb_size,) and type int32 each.
            -- where mb_size is the minibatch size
        sequence[i].data[j] should be the jth element of source sequence number i, or a padding value if the sequence number i is
            shorter than j.

        mask should be a python list of Chainer Variables wrapping a numpy/cupy array of shape (mb_size,) and type bool each.
        mask[i].data[j] should be True if and only if sequence[i].data[j] is not a padding value.

        Return a chainer variable of shape (mb_size, #length, 2*Hi) and type float32
    """

    def __init__(self, Vi, Ei, Hi, init_orth=False, use_bn_length=0, cell_type=rnn_cells.LSTMCell):
        gru_f = cell_type(Ei, Hi)
        gru_b = cell_type(Ei, Hi)

        log.info("constructing encoder [%s]" % (cell_type,))
        super(Encoder, self).__init__(
            emb=L.EmbedID(Vi, Ei),
            #             gru_f = L.GRU(Hi, Ei),
            #             gru_b = L.GRU(Hi, Ei)

            gru_f=gru_f,
            gru_b=gru_b
        )
        self.Hi = Hi

        if use_bn_length > 0:
            self.add_link("bn_f", BNList(Hi, use_bn_length))
#             self.add_link("bn_b", BNList(Hi, use_bn_length)) #TODO
        self.use_bn_length = use_bn_length

        if init_orth:
            ortho_init(self.gru_f)
            ortho_init(self.gru_b)

    def __call__(self, sequence, mask, mode="test"):
        assert mode in "test train".split()

        mb_size = sequence[0].data.shape[0]

        mb_initial_states_f = self.gru_f.get_initial_states(mb_size)
        mb_initial_states_b = self.gru_b.get_initial_states(mb_size)

        embedded_seq = []
        for elem in sequence:
            embedded_seq.append(self.emb(elem))

        prev_states = mb_initial_states_f
        forward_seq = []
        for i, x in enumerate(embedded_seq):
            prev_states = self.gru_f(prev_states, x, mode=mode)
            output = prev_states[-1]
            forward_seq.append(output)

        mask_length = len(mask)
        seq_length = len(sequence)
        assert mask_length <= seq_length
        mask_offset = seq_length - mask_length

        prev_states = mb_initial_states_b

        backward_seq = []
        for pos, x in reversed(list(enumerate(embedded_seq))):
            if pos < mask_offset:
                prev_states = self.gru_b(prev_states, x, mode=mode)
                output = prev_states[-1]
            else:
                reshaped_mask = F.broadcast_to(
                    Variable(self.xp.reshape(mask[pos - mask_offset],
                                             (mb_size, 1)), volatile="auto"), (mb_size, self.Hi))

                prev_states = self.gru_b(prev_states, x, mode=mode)
                output = prev_states[-1]

                masked_prev_states = [None] * len(prev_states)
                for num_state in xrange(len(prev_states)):
                    masked_prev_states[num_state] = F.where(reshaped_mask,
                                                            prev_states[num_state], mb_initial_states_b[num_state])  # TODO: optimize?
                prev_states = tuple(masked_prev_states)
                output = prev_states[-1]

            backward_seq.append(output)

        assert len(backward_seq) == len(forward_seq)
        res = []
        for xf, xb in zip(forward_seq, reversed(backward_seq)):
            res.append(F.reshape(F.concat((xf, xb), 1), (-1, 1, 2 * self.Hi)))

        return F.concat(res, 1)
