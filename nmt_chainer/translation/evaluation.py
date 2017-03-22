#!/usr/bin/env python
"""eval.py: Use a RNNSearch Model"""

__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

from nmt_chainer.utilities.utils import make_batch_src, make_batch_src_tgt, minibatch_provider, compute_bleu_with_unk_as_wrong, de_batch
import logging
import numpy as np
import math
import codecs
import operator
import beam_search
# import h5py

logging.basicConfig()
log = logging.getLogger("rnns:evaluation")
log.setLevel(logging.INFO)


def translate_to_file(encdec, eos_idx, test_src_data, mb_size, tgt_indexer,
                      translations_fn, test_references=None, control_src_fn=None, src_indexer=None, gpu=None, nb_steps=50,
                      reverse_src=False, reverse_tgt=False,
                      s_unk_tag="#S_UNK#", t_unk_tag="#T_UNK#"):

    log.info("computing translations")
    translations = greedy_batch_translate(encdec, eos_idx, test_src_data,
                                          batch_size=mb_size, gpu=gpu, nb_steps=nb_steps,
                                          reverse_src=reverse_src, reverse_tgt=reverse_tgt)

    log.info("writing translation of set to %s" % translations_fn)
    out = codecs.open(translations_fn, "w", encoding="utf8")
    for t in translations:
        if t[-1] == eos_idx:
            t = t[:-1]
#         out.write(convert_idx_to_string(t, tgt_voc) + "\n")
        out.write(tgt_indexer.deconvert(t, unk_tag=t_unk_tag) + "\n")

    if control_src_fn is not None:
        assert src_indexer is not None
        control_out = codecs.open(control_src_fn, "w", encoding="utf8")
        log.info("writing src of test set to %s" % control_src_fn)
        for s in test_src_data:
            #             control_out.write(convert_idx_to_string(s, src_voc) + "\n")
            control_out.write(src_indexer.deconvert(s, unk_tag=s_unk_tag) + "\n")

    if test_references is not None:
        #         unk_id = tgt_indexer.get_unk_idx()  #len(tgt_voc) - 1
        new_unk_id_ref = len(tgt_indexer) + 7777
        new_unk_id_cand = len(tgt_indexer) + 9999
        bc = compute_bleu_with_unk_as_wrong(test_references, [t[:-1] for t in translations], tgt_indexer.is_unk_idx, new_unk_id_ref, new_unk_id_cand)
        log.info("bleu: %r" % bc)
        return bc
    else:
        return None


def compute_loss_all(encdec, test_data, eos_idx, mb_size, gpu=None, reverse_src=False, reverse_tgt=False):
    mb_provider_test = minibatch_provider(test_data, eos_idx, mb_size, nb_mb_for_sorting=-1, loop=False,
                                          gpu=gpu, volatile="on",
                                          reverse_src=reverse_src, reverse_tgt=reverse_tgt)
    test_loss = 0
    test_nb_predictions = 0
    for src_batch, tgt_batch, src_mask in mb_provider_test:
        loss, attn = encdec(src_batch, tgt_batch, src_mask, raw_loss_info=True, mode="test")
        test_loss += loss[0].data
        test_nb_predictions += loss[1]
    test_loss /= test_nb_predictions
    return test_loss


def greedy_batch_translate(encdec, eos_idx, src_data, batch_size=80, gpu=None, get_attention=False, nb_steps=50,
                           reverse_src=False, reverse_tgt=False):
    nb_ex = len(src_data)
    nb_batch = nb_ex / batch_size + (1 if nb_ex % batch_size != 0 else 0)
    res = []
    attn_all = []
    for i in range(nb_batch):
        current_batch_raw_data = src_data[i * batch_size: (i + 1) * batch_size]

        if reverse_src:
            current_batch_raw_data_new = []
            for src_side in current_batch_raw_data:
                current_batch_raw_data_new.append(src_side[::-1])
            current_batch_raw_data = current_batch_raw_data_new

        src_batch, src_mask = make_batch_src(current_batch_raw_data, gpu=gpu, volatile="on")
        sample_greedy, score, attn_list = encdec(src_batch, nb_steps, src_mask, use_best_for_sample=True,
                                                 keep_attn_values=get_attention, mode="test")
        deb = de_batch(sample_greedy, mask=None, eos_idx=eos_idx, is_variable=False)
        res += deb
        if get_attention:
            deb_attn = de_batch(attn_list, mask=None, eos_idx=None, is_variable=True, raw=True,
                                reverse=reverse_tgt)
            attn_all += deb_attn

    if reverse_tgt:
        new_res = []
        for t in res:
            if t[-1] == eos_idx:
                new_res.append(t[:-1][::-1] + [t[-1]])
            else:
                new_res.append(t[::-1])

        res = new_res

    if get_attention:
        assert not reverse_tgt, "not implemented"
        return res, attn_all
    else:
        return res


def reverse_rescore(encdec, src_batch, src_mask, eos_idx, translations, gpu=None):
    from nmt_chainer.utilities import utils

    reversed_translations = []
    for t in translations:
        if t[-1] == eos_idx:
            t = t[:-1]
        reversed_translations.append(t[::-1])

    scorer = encdec.nbest_scorer(src_batch, src_mask)
    tgt_batch, arg_sort = utils.make_batch_tgt(reversed_translations,
                                               eos_idx=eos_idx, gpu=gpu, volatile="on", need_arg_sort=True)

    scores, attn = scorer(tgt_batch)
    scores, _ = scores
    scores = scores.data

    assert len(arg_sort) == len(scores)
    de_sorted_scores = [None] * len(scores)
    for xpos in xrange(len(arg_sort)):
        original_pos = arg_sort[xpos]
        de_sorted_scores[original_pos] = scores[xpos]
    return de_sorted_scores


def beam_search_translate(encdec, eos_idx, src_data, beam_width=20, beam_pruning_margin=None, nb_steps=50, gpu=None,
                          beam_score_coverage_penalty=None, beam_score_coverage_penalty_strength=0.2,
                          need_attention=False, nb_steps_ratio=None, beam_score_length_normalization='none', beam_score_length_normalization_strength=0.2, post_score_length_normalization='simple', post_score_length_normalization_strength=0.2,
                          post_score_coverage_penalty='none', post_score_coverage_penalty_strength=0.2,
                          groundhog=False, force_finish=False,
                          prob_space_combination=False,
                          reverse_encdec=None, use_unfinished_translation_if_none_found=False):
    nb_ex = len(src_data)
#     res = []
    for num_ex in range(nb_ex):
        src_batch, src_mask = make_batch_src([src_data[num_ex]], gpu=gpu, volatile="on")
        assert len(src_mask) == 0
        if nb_steps_ratio is not None:
            nb_steps = int(len(src_data[num_ex]) * nb_steps_ratio) + 1

#         if isinstance(encdec, (tuple, list)):
#             assert len(encdec) == 1
#             encdec = encdec[0]
#
#         translations = encdec.beam_search(src_batch, src_mask, nb_steps = nb_steps, eos_idx = eos_idx,
#                                           beam_width = beam_width,
#                                           beam_opt = beam_opt, need_attention = need_attention,
#                                     groundhog = groundhog)

        if not isinstance(encdec, (tuple, list)):
            encdec = [encdec]
        translations = beam_search.ensemble_beam_search(encdec, src_batch, src_mask, nb_steps=nb_steps, eos_idx=eos_idx,
                                                        beam_width=beam_width,
                                                        beam_pruning_margin=beam_pruning_margin,
                                                        beam_score_length_normalization=beam_score_length_normalization,
                                                        beam_score_length_normalization_strength=beam_score_length_normalization_strength,
                                                        beam_score_coverage_penalty=beam_score_coverage_penalty,
                                                        beam_score_coverage_penalty_strength=beam_score_coverage_penalty_strength,
                                                        need_attention=need_attention, force_finish=force_finish,
                                                        prob_space_combination=prob_space_combination,
                                                        use_unfinished_translation_if_none_found=use_unfinished_translation_if_none_found)

        # TODO: This is a quick patch, but actually ensemble_beam_search probably should not return empty translations except when no translation found
        if len(translations) > 1:
            translations = [t for t in translations if len(t[0]) > 0]

#         print "nb_trans", len(translations), [score for _, score in translations]
#         bests = []
#         translations.sort(key = itemgetter(1), reverse = True)
#         bests.append(translations[0])

        if reverse_encdec is not None and len(translations) > 1:
            rescored_translations = []
            reverse_scores = reverse_rescore(
                reverse_encdec, src_batch, src_mask, eos_idx, [
                    t[0] for t in translations], gpu)
            for num_t in xrange(len(translations)):
                tr, sc, attn = translations[num_t]
                rescored_translations.append(
                    (tr, sc + reverse_scores[num_t], attn))
            translations = rescored_translations

        xp = encdec[0].xp

        if post_score_length_normalization == 'none' and post_score_coverage_penalty == 'none':
            ranking_criterion = operator.itemgetter(1)
        else:
            def ranking_criterion(x):
                length_normalization = 1
                if post_score_length_normalization == 'simple':
                    length_normalization = len(x[0]) + 1
                elif post_score_length_normalization == 'google':
                    length_normalization = pow((len(x[0]) + 5), post_score_length_normalization_strength) / pow(6, post_score_length_normalization_strength)

                coverage_penalty = 0
                if post_score_coverage_penalty == 'google':
                    assert len(src_data[num_ex]) == x[2][0].shape[0]

                    # log.info("sum={0}".format(sum(x[2])))
                    # log.info("min={0}".format(xp.minimum(sum(x[2]), xp.array(1.0))))
                    # log.info("log={0}".format(xp.log(xp.minimum(sum(x[2]), xp.array(1.0)))))
                    log_of_min_of_sum_over_j = xp.log(xp.minimum(sum(x[2]), xp.array(1.0)))
                    coverage_penalty = post_score_coverage_penalty_strength * xp.sum(log_of_min_of_sum_over_j)
                    # log.info("cp={0}".format(coverage_penalty))
                    # cp = 0
                    # for i in xrange(len(src_data[num_ex])):
                    #    attn_sum = 0
                    #    for j in xrange(len(x[0])):
                    #        attn_sum += x[2][j][i]
                    #    #log.info("attn_sum={0}".format(attn_sum))
                    #    #log.info("min={0}".format(min(attn_sum, 1.0)))
                    #    #log.info("log={0}".format(math.log(min(attn_sum, 1.0))))
                    #    cp += math.log(min(attn_sum, 1.0))
                    # log.info("cp={0}".format(cp))
                    # cp *= post_score_coverage_penalty_strength

                    # slow = x[1]/length_normalization + cp
                    # opti = x[1]/length_normalization + coverage_penalty
                    # log.info("type={0}....{1}".format(type(slow), type(opti)))
                    # log.info("shape={0} size={1} dim={2} data={3} elem={4}".format(opti.shape, opti.size, opti.ndim, opti.data, opti.item(0)))
                    # test = '!!!'
                    # if "{0}".format(slow) == "{0}".format(opti):
                    #    test = ''
                    # log.info("score slow <=> optimized: {0} <=> {1} {2}".format(slow, opti, test))

                return x[1] / length_normalization + coverage_penalty

        translations.sort(key=ranking_criterion, reverse=True)

#         bests.append(translations[0])
#         yield bests
        yield translations[0]
#         res.append(bests)
#     return res


def batch_align(encdec, eos_idx, src_tgt_data, batch_size=80, gpu=None):
    nb_ex = len(src_tgt_data)
    nb_batch = nb_ex / batch_size + (1 if nb_ex % batch_size != 0 else 0)
    sum_loss = 0
    attn_all = []
    for i in range(nb_batch):
        current_batch_raw_data = src_tgt_data[i * batch_size: (i + 1) * batch_size]
#         print current_batch_raw_data
        src_batch, tgt_batch, src_mask, arg_sort = make_batch_src_tgt(
            current_batch_raw_data, eos_idx=eos_idx, gpu=gpu, volatile="on", need_arg_sort=True)
        loss, attn_list = encdec(src_batch, tgt_batch, src_mask, keep_attn_values=True)
        deb_attn = de_batch(attn_list, mask=None, eos_idx=None, is_variable=True, raw=True)

        assert len(arg_sort) == len(deb_attn)
        de_sorted_attn = [None] * len(deb_attn)
        for xpos in xrange(len(arg_sort)):
            original_pos = arg_sort[xpos]
            de_sorted_attn[original_pos] = deb_attn[xpos]

        attn_all += de_sorted_attn
        sum_loss += float(loss.data)
    return sum_loss, attn_all

# def convert_idx_to_string(seq, voc, eos_idx = None):
#     trans = []
#     for idx_tgt in seq:
#         if eos_idx is not None and idx_tgt == eos_idx:
#             trans.append("#EOS#")
#         else:
#             if idx_tgt >= len(voc):
#                 log.warn("found unknown idx in tgt : %i / %i"% (idx_tgt, len(voc)))
#             else:
#                 trans.append(voc[idx_tgt])
#     return " ".join(trans)
#
# def convert_idx_to_string_with_attn(seq, voc, attn, unk_idx, unk_pattern = "#T_UNK_%i#"):
#     trans = []
#     for num, idx_tgt in enumerate(seq):
#         if idx_tgt == unk_idx:
#             a = attn[num]
#             xp = cuda.get_array_module(a)
#             src_pos = int(xp.argmax(a))
#             trans.append(unk_pattern%src_pos)
#         else:
#             if idx_tgt >= len(voc):
#                 log.warn("found unknown idx in tgt : %i / %i"% (idx_tgt, len(voc)))
#             else:
#                 trans.append(voc[idx_tgt])
#     return " ".join(trans)


def sample_once(encdec, src_batch, tgt_batch, src_mask, src_indexer, tgt_indexer, eos_idx, max_nb=None,
                s_unk_tag="#S_UNK#", t_unk_tag="#T_UNK#"):
    print "sample"
    sample_greedy, score, attn_list = encdec(src_batch, 50, src_mask, use_best_for_sample=True, need_score=True,
                                             mode="test")


#                 sample, score = encdec(src_batch, 50, src_mask, use_best_for_sample = False)
    assert len(src_batch[0].data) == len(tgt_batch[0].data)
    assert len(sample_greedy[0]) == len(src_batch[0].data)

    debatched_src = de_batch(src_batch, mask=src_mask, eos_idx=None, is_variable=True)
    debatched_tgt = de_batch(tgt_batch, eos_idx=eos_idx, is_variable=True)
    debatched_sample = de_batch(sample_greedy, eos_idx=eos_idx)

    sample_random, score_random, attn_list_random = encdec(src_batch, 50, src_mask, use_best_for_sample=False, need_score=True,
                                                           mode="test")
    debatched_sample_random = de_batch(sample_random, eos_idx=eos_idx)

    for sent_num in xrange(len(debatched_src)):
        if max_nb is not None and sent_num > max_nb:
            break
        src_idx_seq = debatched_src[sent_num]
        tgt_idx_seq = debatched_tgt[sent_num]
        sample_idx_seq = debatched_sample[sent_num]
        sample_random_idx_seq = debatched_sample_random[sent_num]

        print "sent num", sent_num

        for name, seq, unk_tag, indexer, this_eos_idx in zip("src tgt sample sample_random".split(" "),
                                                             [src_idx_seq, tgt_idx_seq, sample_idx_seq, sample_random_idx_seq],
                                                             [s_unk_tag, t_unk_tag, t_unk_tag, t_unk_tag],
                                                             [src_indexer, tgt_indexer, tgt_indexer, tgt_indexer],
                                                             [None, eos_idx, eos_idx, eos_idx]):
            print name, "idx:", seq
            print name, "raw:", " ".join(indexer.deconvert_swallow(seq, unk_tag=unk_tag, eos_idx=this_eos_idx)).encode('utf-8')
            print name, "postp:", indexer.deconvert(seq, unk_tag=unk_tag, eos_idx=this_eos_idx).encode('utf-8')
