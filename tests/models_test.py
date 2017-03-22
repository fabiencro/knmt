#!/usr/bin/env python
"""models_tests.py: Some correctness tests"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import nmt_chainer.models.encoders
import nmt_chainer.models.encoder_decoder
import nmt_chainer.models.attention
import nmt_chainer.utilities.utils as utils
import nmt_chainer.models.decoder_cells as decoder_cells

import logging
logging.basicConfig()
log = logging.getLogger("rnns:models_test")
log.setLevel(logging.INFO)


class EncoderNaive(nmt_chainer.models.encoders.Encoder):
    def __init__(self, Vi, Ei, Hi):
        super(EncoderNaive, self).__init__(Vi, Ei, Hi)

    def naive_call(self, sequence, mask):

        mb_size = sequence[0].data.shape[0]
        assert mb_size == 1

        mb_initial_states_f = self.gru_f.get_initial_states(mb_size)
        mb_initial_states_b = self.gru_b.get_initial_states(mb_size)

        embedded_seq = []
        for elem in sequence:
            embedded_seq.append(self.emb(elem))

#         self.gru_f.reset_state()
        prev_states = mb_initial_states_f
        forward_seq = []
        for i, x in enumerate(embedded_seq):
            prev_states = self.gru_f(prev_states, x)
            forward_seq.append(prev_states)

#         self.gru_b.reset_state()
        prev_states = mb_initial_states_b
        backward_seq = []
        for pos, x in reversed(list(enumerate(embedded_seq))):
            prev_states = self.gru_b(prev_states, x)
            backward_seq.append(prev_states)

        assert len(backward_seq) == len(forward_seq)
        res = []
        for xf, xb in zip(forward_seq, reversed(backward_seq)):
            res.append(F.concat((xf[-1], xb[-1]), 1))

        return res


class TestEncoder:
    def test_naive(self):
        Vi, Ei, Hi = 12, 17, 7
        enc = EncoderNaive(Vi, Ei, Hi)
        raw_seq = [2, 5, 4, 3, 0, 0, 1, 11, 3]
        input_seq = [Variable(np.array([v], dtype=np.int32)) for v in raw_seq]
        mask = [np.array([True], dtype=np.bool) for v in raw_seq]

        fb_naive = enc.naive_call(input_seq, None)

#         mask = np.array([[True] * len(raw_seq)], dtype = np.bool)
        fb = enc(input_seq, mask)

        for i in range(len(raw_seq)):
            #             print i, fb.data[0][i], fb_naive[i].data[0], np.max(np.abs(fb.data[0][i] - fb_naive[i].data[0]))
            assert np.all(fb.data[0][i] == fb_naive[i].data[0])
#             assert False

    def test_multibatch(self):
        Vi, Ei, Hi = 12, 17, 7
        enc = EncoderNaive(Vi, Ei, Hi)

        raw_seq1 = [2, 5, 0, 3]
        raw_seq2 = [2, 5, 4, 3, 0, 0, 1, 11, 3]
        raw_seq3 = [2, 5, 4, 3, 0, 11, 3]
        raw_seq4 = [5, 3, 0, 0, 1, 11, 3]

        src_data = [raw_seq1, raw_seq2, raw_seq3, raw_seq4]
        src_batch, src_mask = utils.make_batch_src(src_data)
        fb = enc(src_batch, src_mask)

        for i in xrange(len(src_data)):
            raw_s = src_data[i]
            input_seq = [Variable(np.array([v], dtype=np.int32)) for v in raw_s]
            fb_naive = enc.naive_call(input_seq, None)
            for j in range(len(raw_s)):
                print "maxdiff:", np.max(np.abs(fb.data[i][j] - fb_naive[j].data[0]))
                assert np.allclose(fb.data[i][j], fb_naive[j].data[0], atol=1e-6)

    def test_multibatch_error(self):
        Vi, Ei, Hi = 12, 17, 7
        enc = EncoderNaive(Vi, Ei, Hi)

        raw_seq1 = [2, 5, 0, 3]
        raw_seq2 = [2, 5, 4, 3, 0, 0, 1, 11, 3]
        raw_seq3 = [2, 5, 4, 3, 0, 11, 3]
        raw_seq4 = [5, 3, 0, 0, 1, 11, 3]

        src_data = [raw_seq1, raw_seq2, raw_seq3, raw_seq4]
        src_batch, src_mask = utils.make_batch_src(src_data)

        assert len(src_mask) == 5
        print [e.data for e in src_mask]

        src_mask[0][0] = True
        print [e.data for e in src_mask]

        fb = enc(src_batch, src_mask)

        i = 0
        raw_s = src_data[i]
        input_seq = [Variable(np.array([v], dtype=np.int32)) for v in raw_s]
        fb_naive = enc.naive_call(input_seq, None)
        for j in range(len(raw_s)):
            print "maxdiff:", np.max(np.abs(fb.data[i][j] - fb_naive[j].data[0]))
            assert not np.allclose(fb.data[i][j], fb_naive[j].data[0], atol=1e-6)


class AttentionModuleNaive(nmt_chainer.models.attention.AttentionModule):
    def __init__(self, Hi, Ha, Ho):
        super(AttentionModuleNaive, self).__init__(Hi, Ha, Ho)

    def naive_call(self, fb_concat, mask):
        #         mb_size, nb_elems, Hi = fb_concat.data.shape
        #
        nb_elems = len(fb_concat)
        mb_size, Hi = fb_concat[0].data.shape
        assert mb_size == 1
        assert Hi == self.Hi

        precomputed_al_factor = []
        for x in fb_concat:
            precomputed_al_factor.append(self.al_lin_h(x))

        def compute_ctxt(previous_state):
            current_mb_size = previous_state.data.shape[0]
            if current_mb_size < mb_size:
                assert False
                al_factor, _ = F.split_axis(
                    precomputed_al_factor, (current_mb_size,), 0)
                used_fb_concat, _ = F.split_axis(
                    fb_concat, (current_mb_size,), 0)
#                 used_concatenated_mask, _ = F.split_axis(concatenated_mask, (current_mb_size,), 0)
            else:
                al_factor = precomputed_al_factor
                used_fb_concat = fb_concat
#                 used_concatenated_mask = concatenated_mask

            state_al_factor = self.al_lin_s(previous_state)

            a_coeffs = []
            for x in al_factor:
                a_coeffs.append(self.al_lin_o(F.tanh(x + state_al_factor)))
#             state_al_factor_bc = F.broadcast_to(F.reshape(state_al_factor, (current_mb_size, 1, self.Ha)), (current_mb_size, nb_elems, self.Ha) )
#             a_coeffs = F.reshape(self.al_lin_o(F.reshape(F.tanh(state_al_factor_bc + al_factor),
#                             (current_mb_size* nb_elems, self.Ha))), (current_mb_size, nb_elems))

#             a_coeffs = a_coeffs - 10000 * (1-used_concatenated_mask.data)
            a_coeffs_concat = F.concat(a_coeffs, 1)
            assert a_coeffs_concat.data.shape == (mb_size, nb_elems)
            attn = F.softmax(a_coeffs_concat)

            splitted_attn = F.split_axis(attn, len(fb_concat), 1)
            ci = None
            for i in xrange(nb_elems):
                contrib = F.broadcast_to(
                    splitted_attn[i], (mb_size, Hi)) * used_fb_concat[i]
                ci = ci + contrib if ci is not None else contrib
#             ci = F.reshape(F.batch_matmul(attn, used_fb_concat, transa = True), (current_mb_size, self.Hi))

            return ci, attn

        return compute_ctxt


class TestAttention:
    def test_naive(self):
        Vi, Ei, Hi = 12, 17, 7
        enc = EncoderNaive(Vi, Ei, Hi)

        Hi_a, Ha, Ho = 2 * Hi, 19, 23
        attn_model = AttentionModuleNaive(Hi_a, Ha, Ho)

        raw_seq = [2, 5, 4, 3, 0, 0, 1, 11, 3]
        input_seq = [Variable(np.array([v], dtype=np.int32)) for v in raw_seq]
        mask = [np.array([True], dtype=np.bool) for v in raw_seq]

        fb_naive = enc.naive_call(input_seq, None)
        compute_ctxt_naive = attn_model.naive_call(fb_naive, None)
#         mask = np.array([[True] * len(raw_seq)], dtype = np.bool)
        fb = enc(input_seq, mask)
        compute_ctxt = attn_model(fb, mask)

        state = Variable(np.random.randn(1, Ho).astype(np.float32))

        ci, attn = compute_ctxt(state)
        ci_naive, attn_naive = compute_ctxt_naive(state)
        assert np.allclose(ci.data, ci_naive.data)
        assert np.allclose(attn.data, attn_naive.data)

    def test_multibatch(self):
        Vi, Ei, Hi = 12, 17, 7
        enc = EncoderNaive(Vi, Ei, Hi)

        Hi_a, Ha, Ho = 2 * Hi, 19, 23
        attn_model = AttentionModuleNaive(Hi_a, Ha, Ho)

        raw_seq1 = [2, 5, 0, 3]
        raw_seq2 = [2, 5, 4, 3, 0, 0, 1, 11, 3]
        raw_seq3 = [2, 5, 4, 3, 0, 11, 3]
        raw_seq4 = [5, 3, 0, 0, 1, 11, 3]

        src_data = [raw_seq1, raw_seq2, raw_seq3, raw_seq4]
        src_batch, src_mask = utils.make_batch_src(src_data)
        fb = enc(src_batch, src_mask)
        compute_ctxt = attn_model(fb, src_mask)

        state_raw = np.random.randn(4, Ho).astype(np.float32)
        state = Variable(state_raw)

        ci, attn = compute_ctxt(state)

        for i in xrange(len(src_data)):
            raw_s = src_data[i]
            input_seq = [Variable(np.array([v], dtype=np.int32)) for v in raw_s]
            fb_naive = enc.naive_call(input_seq, None)
            compute_ctxt_naive = attn_model.naive_call(fb_naive, None)
            ci_naive, attn_naive = compute_ctxt_naive(Variable(state_raw[i].reshape(1, -1)))
            print "maxdiff ci:", np.max(np.abs(ci.data[i] - ci_naive.data[0]))
            assert np.allclose(ci.data[i], ci_naive.data[0], atol=1e-6)
#             print attn.data.shape, attn_naive.data.shape
            print "maxdiff attn:", np.max(np.abs(attn.data[i][:len(raw_s)] - attn_naive.data[0]))
            assert np.allclose(attn.data[i][:len(raw_s)], attn_naive.data[0], atol=1e-6)
            assert np.all(attn.data[i][len(raw_s):] == 0)


class DecoderNaive(decoder_cells.Decoder):
    def __init__(self, Vo, Eo, Ho, Ha, Hi, Hl):
        super(DecoderNaive, self).__init__(Vo, Eo, Ho, Ha, Hi, Hl, cell_type='gru')
        self.attn_module = AttentionModuleNaive(Hi, Ha, Ho)

    def naive_call(self, fb_concat, targets, mask):
        compute_ctxt = self.attn_module.naive_call(fb_concat, mask)
        loss = None
        current_mb_size = targets[0].data.shape[0]
        assert current_mb_size == 1
        previous_states = self.gru.get_initial_states(current_mb_size)
#         previous_word = Variable(np.array([self.bos_idx] * mb_size, dtype = np.int32))
        # xp = cuda.get_array_module(self.gru.initial_state.data)
        with cuda.get_device(self.gru.initial_state.data):
            prev_y = F.broadcast_to(self.bos_embeding, (1, self.Eo))
#             previous_word = Variable(xp.array([self.bos_idx] * current_mb_size, dtype = np.int32))
        previous_word = None
        attn_list = []
        total_nb_predictions = 0
        for i in xrange(len(targets)):
            if previous_word is not None:  # else we are using the initial prev_y
                prev_y = self.emb(previous_word)
            ci, attn = compute_ctxt(previous_states[-1])
            concatenated = F.concat((prev_y, ci))
    #             print concatenated.data.shape
            new_states = self.gru(previous_states, concatenated)

            all_concatenated = F.concat((concatenated, new_states[-1]))
            logits = self.lin_o(self.maxo(all_concatenated))

            local_loss = F.softmax_cross_entropy(logits, targets[i])

            loss = local_loss if loss is None else loss + local_loss
            total_nb_predictions += 1
            previous_word = targets[i]
            previous_states = new_states
            attn_list.append(attn)

        loss = loss / total_nb_predictions
        return loss, attn_list


class EncoderDecoderNaive(nmt_chainer.models.encoder_decoder.EncoderDecoder):
    def __init__(self, Vi, Ei, Hi, Vo, Eo, Ho, Ha, Hl):
        super(
            EncoderDecoderNaive,
            self).__init__(
            Vi,
            Ei,
            Hi,
            Vo,
            Eo,
            Ho,
            Ha,
            Hl)
        self.enc = EncoderNaive(Vi, Ei, Hi)
        self.dec = DecoderNaive(Vo, Eo, Ho, Ha, 2 * Hi, Hl)

    def naive_call(self, src_batch, tgt_batch, src_mask):
        fb_src = self.enc.naive_call(src_batch, src_mask)
        loss = self.dec.naive_call(fb_src, tgt_batch, src_mask)
        return loss


class TestEncoderDecoder:
    def test_naive(self):
        Vi, Ei, Hi, Vo, Eo, Ho, Ha, Hl = 29, 37, 13, 17, 7, 12, 19, 33
        encdec = EncoderDecoderNaive(Vi, Ei, Hi, Vo, Eo, Ho, Ha, Hl)
        raw_seq = [2, 5, 4, 3, 0, 0, 1, 11, 3]
        input_seq = [Variable(np.array([v], dtype=np.int32)) for v in raw_seq]
        mask = [np.array([True], dtype=np.bool) for v in raw_seq]

        raw_seq_tgt = [2, 12, 4, 0, 1, 11, 3]
        tgt_batch = [Variable(np.array([v], dtype=np.int32)) for v in raw_seq_tgt]

        loss_naive, attn_list_naive = encdec.naive_call(input_seq, tgt_batch, None)
        loss, attn = encdec(input_seq, tgt_batch, mask)

        assert np.allclose(loss.data, loss_naive.data)

    def test_multibatch(self):
        Vi, Ei, Hi, Vo, Eo, Ho, Ha, Hl = 29, 37, 13, 17, 7, 12, 19, 33
        encdec = EncoderDecoderNaive(Vi, Ei, Hi, Vo, Eo, Ho, Ha, Hl)
        eos_idx = Vo - 1

        raw_seq1 = [2, 5, 0, 3], [4, 6]
        raw_seq2 = [2, 5, 4, 3, 0, 0, 1, 11, 3], [4, 8, 9, 12, 0]
        raw_seq3 = [2, 5, 4, 3, 0, 11, 3], [5, 7, 1, 4, 4, 1, 0, 0, 5, 5, 3, 4, 6, 7, 8]
        raw_seq4 = [5, 3, 0, 0, 1, 11, 3], [0, 0, 1, 1]

        trg_data = [raw_seq1, raw_seq2, raw_seq3, raw_seq4]
        src_batch, tgt_batch, src_mask = utils.make_batch_src_tgt(trg_data, eos_idx=eos_idx)

        loss, attn = encdec(src_batch, tgt_batch, src_mask)

        total_loss_naive = 0
        total_length = 0
        for i in xrange(len(trg_data)):
            raw_s_raw = trg_data[i]
            raw_s = [raw_s_raw[0], raw_s_raw[1] + [eos_idx]]
            input_seq = [Variable(np.array([v], dtype=np.int32)) for v in raw_s[0]]
            tgt_seq = [Variable(np.array([v], dtype=np.int32)) for v in raw_s[1]]
            loss_naive, attn_naive = encdec.naive_call(input_seq, tgt_seq, None)
            total_loss_naive += float(loss_naive.data) * len(raw_s[1])
            total_length += len(raw_s[1])

        assert abs(total_loss_naive / total_length - float(loss.data)) < 1e-6


class TestBeamSearch:
    def test_1(self):
        import nmt_chainer.translation.evaluation as evaluation
        Vi, Ei, Hi, Vo, Eo, Ho, Ha, Hl = 29, 37, 13, 53, 7, 12, 19, 33
        encdec = nmt_chainer.models.encoder_decoder.EncoderDecoder(
            Vi, Ei, Hi, Vo, Eo, Ho, Ha, Hl)
        eos_idx = Vo - 1
        src_data = [[2, 3, 3, 4, 4, 5], [1, 3, 8, 9, 2]]
#         best1_gen = evaluation.beam_search_translate(encdec, eos_idx, src_data, beam_width = 10, nb_steps = 15, gpu = None, beam_opt = False,
#                           need_attention = False)
#         best2_gen = evaluation.beam_search_translate(encdec, eos_idx, src_data, beam_width = 10, nb_steps = 15, gpu = None, beam_opt = True,
#                           need_attention = False)

        # TODO: not much point to this test now that beam_opt distinction is removed
        best1_gen = evaluation.beam_search_translate(encdec, eos_idx, src_data, beam_width=10, nb_steps=15, gpu=None,
                                                     need_attention=False)
        best2_gen = evaluation.beam_search_translate(encdec, eos_idx, src_data, beam_width=10, nb_steps=15, gpu=None,
                                                     need_attention=False)
        res1a, res1b = next(best1_gen), next(best2_gen)
        res2a, res2b = next(best1_gen), next(best2_gen)
