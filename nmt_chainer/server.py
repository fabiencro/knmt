#!/usr/bin/env python
"""server.py: Process requests to RNNSearch"""
__author__ = "Frederic Bergeron"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "bergeron@pa.jst.jp"
__status__ = "Development"

import datetime
import json
import numpy as np
from chainer import cuda, serializers
import logging
import sys
#import h5py
import models
from make_data import Indexer, build_dataset_one_side_from_string
from evaluation import (greedy_batch_translate, 
#                         convert_idx_to_string, 
                        batch_align, 
                        beam_search_translate, 
#                         convert_idx_to_string_with_attn
                        )

# import visualisation
import bleu_computer
import logging
import codecs

import rnn_cells

import time
import socket
import xml.etree.ElementTree as ET
import re
import subprocess
import replace_tgt_unk

logging.basicConfig()
log = logging.getLogger("rnns:eval")
log.setLevel(logging.INFO)

class Evaluator:

    def __init__(self, training_config, trained_model, additional_training_config, additional_trained_model, reverse_training_config, reverse_trained_model, 
            max_nb_ex, beam_width, nb_steps, beam_opt, nb_steps_ratio, use_raw_score, groundhog, tgt_unk_id, force_finish, prob_space_combination, gpu,
            dic, remove_unk, normalize_unicode_unk, attempt_to_relocate_unk_source):
        self.training_config = training_config
        self.trained_model = trained_model
        self.additional_training_config = additional_training_config
        self.additional_trained_model = additional_trained_model
        self.reverse_training_config = reverse_training_config
        self.reverse_trained_model = reverse_trained_model
        self.max_nb_ex = max_nb_ex
        self.beam_width = beam_width
        self.nb_steps = nb_steps
        self.beam_opt = beam_opt
        self.nb_steps_ratio = nb_steps_ratio
        self.use_raw_score = use_raw_score
        self.groundhog = groundhog
        self.tgt_unk_id = tgt_unk_id
        self.force_finish = force_finish
        self.prob_space_combination = prob_space_combination
        self.gpu = gpu
        self.dic = dic
        self.remove_unk = remove_unk
        self.normalize_unicode_unk = normalize_unicode_unk
        self.attempt_to_relocate_unk_source = attempt_to_relocate_unk_source
    
        self.encdec, self.eos_idx, self.src_indexer, self.tgt_indexer = create_and_load_encdec_from_files(self.training_config, self.trained_model)
        if self.gpu is not None:
            self.encdec = self.encdec.to_gpu(self.gpu)
            
        self.encdec_list = [self.encdec]
        
        if self.additional_training_config is not None:
            assert len(self.additional_training_config) == len(self.additional_trained_model)
            
            for (config_training_fn, trained_model_fn) in zip(self.additional_training_config, 
                                                              self.additional_trained_model):
                this_encdec, this_eos_idx, this_src_indexer, this_tgt_indexer = create_and_load_encdec_from_files(
                                config_training_fn, trained_model_fn)
            
                if eos_idx != this_eos_idx:
                    raise Exception("incompatible models")
                    
                if len(src_indexer) != len(this_src_indexer):
                    raise Exception("incompatible models")
                  
                if len(tgt_indexer) != len(this_tgt_indexer):
                    raise Exception("incompatible models")
                                  
                if self.gpu is not None:
                    this_encdec = this_encdec.to_gpu(self.gpu)
                
                self.encdec_list.append(this_encdec)
                
        if self.reverse_training_config is not None:
            self.reverse_encdec, self.reverse_eos_idx, self.reverse_src_indexer, self.reverse_tgt_indexer = create_and_load_encdec_from_files(
                                self.reverse_training_config, self.reverse_trained_model)
            
            if eos_idx != reverse_eos_idx:
                raise Exception("incompatible models")
                
            if len(src_indexer) != len(reverse_src_indexer):
                raise Exception("incompatible models")
              
            if len(tgt_indexer) != len(reverse_tgt_indexer):
                raise Exception("incompatible models")
                              
            if self.gpu is not None:
                self.reverse_encdec = self.reverse_encdec.to_gpu(self.gpu)
        else:
            self.reverse_encdec = None    
            
    def __translate_with_beam_search(self, gpu, encdec, eos_idx, src_data, beam_width, nb_steps, beam_opt, 
           nb_steps_ratio, use_raw_score, 
           groundhog,
           tgt_unk_id, tgt_indexer, force_finish = False,
           prob_space_combination = False, reverse_encdec = None):
        #log.info("writing translation of to %s"% dest_fn)
        #out = codecs.open(dest_fn, "w", encoding = "utf8")
        out = ''
        with cuda.get_device(gpu):
            translations_gen = beam_search_translate(
                        encdec, eos_idx, src_data, beam_width = beam_width, nb_steps = nb_steps, 
                                        gpu = gpu, beam_opt = beam_opt, nb_steps_ratio = nb_steps_ratio,
                                        need_attention = True, score_is_divided_by_length = not use_raw_score,
                                        groundhog = groundhog, force_finish = force_finish,
                                        prob_space_combination = prob_space_combination,
                                        reverse_encdec = reverse_encdec)
            
            
#         for num_t in range(len(translations)):
#             print num_t
#             for t, score in translations[num_t]:
#                 ct = convert_idx_to_string(t[:-1], tgt_voc + ["#T_UNK#"])
#                 print ct, score
#                 out.write(ct + "\n")
            for num_t, (t, score, attn) in enumerate(translations_gen):
                if num_t %200 == 0:
                    print >>sys.stderr, num_t,
                elif num_t %40 == 0:
                    print >>sys.stderr, "*",
#                 t, score = bests[1]
#                 ct = convert_idx_to_string(t, tgt_voc + ["#T_UNK#"])
#                 ct = convert_idx_to_string_with_attn(t, tgt_voc, attn, unk_idx = len(tgt_voc))
                if tgt_unk_id == "align":
                    def unk_replacer(num_pos, unk_id):
                        unk_pattern = "#T_UNK_%i#"
                        a = attn[num_pos]
                        xp = cuda.get_array_module(a)
                        src_pos = int(xp.argmax(a))
                        return unk_pattern%src_pos
                elif tgt_unk_id == "id":
                    def unk_replacer(num_pos, unk_id):
                        unk_pattern = "#T_UNK_%i#"
                        return unk_pattern%unk_id         
                else:
                    assert False
                
                ct = " ".join(tgt_indexer.deconvert(t, unk_tag = unk_replacer))
                
#                 print convert_idx_to_string(bests[0][0], tgt_voc + ["#T_UNK#"]) , bests[0][1]
#                 print convert_idx_to_string(bests[1][0], tgt_voc + ["#T_UNK#"]), bests[1][1], bests[1][1] / len(bests[1][0])
                #out.write(ct + "\n")
                out += ct + "\n"
            print >>sys.stderr
        return out

    def eval(self, request, mode):
        log.info("processing source string %s" % request)
        src_data, dic_src, make_data_infos = build_dataset_one_side_from_string(request, 
                    src_voc_limit = None, max_nb_ex = self.max_nb_ex, dic_src = self.src_indexer)
        log.info("%i sentences loaded" % make_data_infos.nb_ex)
        log.info("#tokens src: %i   of which %i (%f%%) are unknown"%(make_data_infos.total_token, 
                                                                     make_data_infos.total_count_unk, 
                                                                     float(make_data_infos.total_count_unk * 100) / 
                                                                        make_data_infos.total_token))
        assert dic_src == self.src_indexer

        tgt_data = None

        if mode == "beam_search":
            response = self.__translate_with_beam_search(self.gpu, self.encdec, self.eos_idx, src_data, self.beam_width, 
                                               self.nb_steps, self.beam_opt, 
                                               self.nb_steps_ratio, self.use_raw_score, 
                                               self.groundhog,
                                               self.tgt_unk_id, self.tgt_indexer, self.force_finish, #force_finish = self.force_finish,
                                               self.prob_space_combination, #prob_space_combination = self.prob_space_combination,
                                               self.reverse_encdec) #reverse_encdec = self.reverse_encdec)
                
        print(timestamped_msg('Response: {0}'.format(response)))
        response = replace_tgt_unk.replace_unk_from_string(response, request, self.dic, self.remove_unk, self.normalize_unicode_unk, self.attempt_to_relocate_unk_source)
        return response

class Server:

    def __init__(self, evaluator, parse_server_command, port=44666):
        self.evaluator = evaluator
        self.port = port
        self.parse_server_command = parse_server_command
        self.start()

    def __build_response(self, out):
        response = {}
        response['out'] = out
        return json.dumps(response)

    def __handle_request(self, request):
        print(timestamped_msg("Handling request..."))
        root = ET.fromstring(request)
        article_id = root.attrib['id']
        mode = root.attrib['mode']
        print("Article id: %s" % article_id)
        out = ""
        sentences = root.findall('sentence')
        for idx, sentence in enumerate(sentences):
            text = sentence.findtext('i_sentence')
            # print "text=%s" % text
            
            cmd = self.parse_server_command % text
            # print "cmd=%s" % cmd
            parser_output = subprocess.check_output(cmd, shell=True)
            # print "parser_output=%s" % parser_output

            words = []
            for line in parser_output.split("\n"):
                if (line.startswith('#')):
                    continue
                elif (not line.strip()):
                    break
                else:
                    parts = line.split("\t")
                    word = parts[3]
                    words.append(word)
            splitted_sentence = ' '.join(words)
            # print "splitted_sentence=" + splitted_sentence

            print(timestamped_msg("Translating sentence %d" % idx))
            translation = self.evaluator.eval(splitted_sentence.decode('utf-8'), mode)
            out += translation
        response = self.__build_response(out)
        return response

    def start(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print(timestamped_msg("Start listening for requests on {0} port {1}...".format(socket.gethostname(), self.port)))
        server_socket.bind(('', self.port))
        server_socket.listen(5)

        while True:
            (client_socket, address) = server_socket.accept()
            print(timestamped_msg('Got connection from {0}'.format(address)))
            request = client_socket.recv(1024)
            # print "request=" + request
            response = self.__handle_request(request)
            client_socket.send(response)
            client_socket.close()
        
def timestamped_msg(msg):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    return "{0}: {1}".format(timestamp, msg) 

def create_encdec_from_config(config_training):

    voc_fn = config_training["voc"]
    log.info("loading voc from %s"% voc_fn)
    src_voc, tgt_voc = json.load(open(voc_fn))
    
    src_indexer = Indexer.make_from_serializable(src_voc)
    tgt_indexer = Indexer.make_from_serializable(tgt_voc)
    tgt_voc = None
    src_voc = None
    
    
#     Vi = len(src_voc) + 1 # + UNK
#     Vo = len(tgt_voc) + 1 # + UNK
    
    Vi = len(src_indexer) # + UNK
    Vo = len(tgt_indexer) # + UNK
    
    print config_training
    
    Ei = config_training["command_line"]["Ei"]
    Hi = config_training["command_line"]["Hi"]
    Eo = config_training["command_line"]["Eo"]
    Ho = config_training["command_line"]["Ho"]
    Ha = config_training["command_line"]["Ha"]
    Hl = config_training["command_line"]["Hl"]
    
    encoder_cell_type = config_training["command_line"].get("encoder_cell_type", "gru")
    decoder_cell_type = config_training["command_line"].get("decoder_cell_type", "gru")
    
    use_bn_length = config_training["command_line"].get("use_bn_length", None)
    
    import gzip
    
    if "lexical_probability_dictionary" in config_training["command_line"] and config_training["command_line"]["lexical_probability_dictionary"] is not None:
        log.info("opening lexical_probability_dictionary %s" % config_training["command_line"]["lexical_probability_dictionary"])
        lexical_probability_dictionary_all = json.load(gzip.open(config_training["command_line"]["lexical_probability_dictionary"], "rb"))
        log.info("computing lexical_probability_dictionary_indexed")
        lexical_probability_dictionary_indexed = {}
        for ws in lexical_probability_dictionary_all:
            ws_idx = src_indexer.convert([ws])[0]
            if ws_idx in lexical_probability_dictionary_indexed:
                assert src_indexer.is_unk_idx(ws_idx)
            else:
                lexical_probability_dictionary_indexed[ws_idx] = {}
            for wt in lexical_probability_dictionary_all[ws]:
                wt_idx = tgt_indexer.convert([wt])[0]
                if wt_idx in lexical_probability_dictionary_indexed[ws_idx]:
                    assert src_indexer.is_unk_idx(ws_idx) or tgt_indexer.is_unk_idx(wt_idx)
                    lexical_probability_dictionary_indexed[ws_idx][wt_idx] += lexical_probability_dictionary_all[ws][wt]
                else:
                    lexical_probability_dictionary_indexed[ws_idx][wt_idx] = lexical_probability_dictionary_all[ws][wt]
        lexical_probability_dictionary = lexical_probability_dictionary_indexed
    else:
        lexical_probability_dictionary = None
    
    eos_idx = Vo
    encdec = models.EncoderDecoder(Vi, Ei, Hi, Vo + 1, Eo, Ho, Ha, Hl, use_bn_length = use_bn_length,
                                   encoder_cell_type = rnn_cells.create_cell_model_from_string(encoder_cell_type),
                                       decoder_cell_type = rnn_cells.create_cell_model_from_string(decoder_cell_type),
                                       lexical_probability_dictionary = lexical_probability_dictionary,
                                       lex_epsilon = config_training["command_line"].get("lexicon_prob_epsilon", 0.001))
    
    return encdec, eos_idx, src_indexer, tgt_indexer
    
def create_and_load_encdec_from_files(config_training_fn, trained_model):
    log.info("loading model config from %s" % config_training_fn)
    config_training = json.load(open(config_training_fn))

    encdec, eos_idx, src_indexer, tgt_indexer = create_encdec_from_config(config_training)
    
    log.info("loading model from %s" % trained_model)
    serializers.load_npz(trained_model, encdec)
    
    return encdec, eos_idx, src_indexer, tgt_indexer
    
def commandline():
    import argparse
    parser = argparse.ArgumentParser(description= "Use a RNNSearch model", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("training_config", help = "prefix of the trained model")
    parser.add_argument("trained_model", help = "prefix of the trained model")
    
    parser.add_argument("--additional_training_config", nargs = "*", help = "prefix of the trained model")
    parser.add_argument("--additional_trained_model", nargs = "*", help = "prefix of the trained model")
    
    parser.add_argument("--tgt_fn", help = "target text")
    
    parser.add_argument("--nbest_to_rescore", help = "nbest list in moses format")
    
    #parser.add_argument("--mode", default = "translate", 
    #                    choices = ["translate", "align", "translate_attn", "beam_search", "eval_bleu",
    #                               "score_nbest"], help = "target text")
    
    parser.add_argument("--ref", help = "target text")
    
    parser.add_argument("--gpu", type = int, help = "specify gpu number to use, if any")
    
    parser.add_argument("--max_nb_ex", type = int, help = "only use the first MAX_NB_EX examples")
    parser.add_argument("--mb_size", type = int, default= 80, help = "Minibatch size")
    parser.add_argument("--beam_width", type = int, default= 20, help = "beam width")
    parser.add_argument("--nb_steps", type = int, default= 50, help = "nb_steps used in generation")
    parser.add_argument("--nb_steps_ratio", type = float, help = "nb_steps used in generation as a ratio of input length")
    parser.add_argument("--nb_batch_to_sort", type = int, default= 20, help = "Sort this many batches by size.")
    parser.add_argument("--beam_opt", default = False, action = "store_true")
    parser.add_argument("--tgt_unk_id", choices = ["attn", "id"], default = "align")
    parser.add_argument("--groundhog", default = False, action = "store_true")
    
    parser.add_argument("--force_finish", default = False, action = "store_true")
    
    # arguments for unk replace
    parser.add_argument("--dic")
    parser.add_argument("--remove_unk", default = False, action = "store_true")
    parser.add_argument("--normalize_unicode_unk", default = False, action = "store_true")
    parser.add_argument("--attempt_to_relocate_unk_source", default = False, action = "store_true")
    
    parser.add_argument("--use_raw_score", default = False, action = "store_true")
    
    parser.add_argument("--prob_space_combination", default = False, action = "store_true")
    
    parser.add_argument("--reverse_training_config", help = "prefix of the trained model")
    parser.add_argument("--reverse_trained_model", help = "prefix of the trained model")
    
    parser.add_argument("--port", help = "port for listening request", default = 44666)
    parser.add_argument("--parse_server_command", help = "command to communicate with the parse-server")
    args = parser.parse_args()

    evaluator = Evaluator(args.training_config, args.trained_model, args.additional_training_config, args.additional_trained_model, 
                   args.reverse_training_config, args.reverse_trained_model, args.max_nb_ex, args.beam_width, args.nb_steps, args.beam_opt, args.nb_steps_ratio, args.use_raw_score, 
                   args.groundhog, args.tgt_unk_id, args.force_finish, args.prob_space_combination, args.gpu, args.dic, args.remove_unk, args.normalize_unicode_unk, args.attempt_to_relocate_unk_source)

    server = Server(evaluator, args.parse_server_command, int(args.port))
    server.start()
    
if __name__ == '__main__':
    commandline() 
