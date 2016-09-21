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

import visualisation
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
import bokeh.embed
import bokeh.resources

logging.basicConfig()
log = logging.getLogger("rnns:eval")
log.setLevel(logging.INFO)

class Evaluator:

    def __init__(self, training_config, trained_model, additional_training_config, additional_trained_model, reverse_training_config, reverse_trained_model, 
            max_nb_ex, mb_size, beam_opt, tgt_unk_id, gpu, dic):
        self.training_config = training_config
        self.trained_model = trained_model
        self.additional_training_config = additional_training_config
        self.additional_trained_model = additional_trained_model
        self.reverse_training_config = reverse_training_config
        self.reverse_trained_model = reverse_trained_model
        self.max_nb_ex = max_nb_ex
        self.mb_size = mb_size
        self.beam_opt = beam_opt
        self.tgt_unk_id = tgt_unk_id
        self.gpu = gpu
        self.dic = dic
    
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
            
                if self.eos_idx != this_eos_idx:
                    raise Exception("incompatible models")
                    
                if len(self.src_indexer) != len(this_src_indexer):
                    raise Exception("incompatible models")
                  
                if len(self.tgt_indexer) != len(this_tgt_indexer):
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
            
    def eval(self, request, request_number, beam_width, nb_steps, nb_steps_ratio, 
            remove_unk, normalize_unicode_unk, attempt_to_relocate_unk_source, use_raw_score, groundhog, force_finish, prob_space_combination):
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

        out = ''
        with cuda.get_device(self.gpu):
            translations_gen = beam_search_translate(
                        self.encdec, self.eos_idx, src_data, beam_width = beam_width, nb_steps = nb_steps, 
                                        gpu = self.gpu, beam_opt = self.beam_opt, nb_steps_ratio = nb_steps_ratio,
                                        need_attention = True, score_is_divided_by_length = not use_raw_score,
                                        groundhog = groundhog, force_finish = force_finish,
                                        prob_space_combination = prob_space_combination,
                                        reverse_encdec = self.reverse_encdec)

            for num_t, (t, score, attn) in enumerate(translations_gen):
                if num_t %200 == 0:
                    print >>sys.stderr, num_t,
                elif num_t %40 == 0:
                    print >>sys.stderr, "*",
                if self.tgt_unk_id == "align":
                    def unk_replacer(num_pos, unk_id):
                        unk_pattern = "#T_UNK_%i#"
                        a = attn[num_pos]
                        xp = cuda.get_array_module(a)
                        src_pos = int(xp.argmax(a))
                        return unk_pattern%src_pos
                elif self.tgt_unk_id == "id":
                    def unk_replacer(num_pos, unk_id):
                        unk_pattern = "#T_UNK_%i#"
                        return unk_pattern%unk_id         
                else:
                    assert False
                
                ct = " ".join(self.tgt_indexer.deconvert(t, unk_tag = unk_replacer))
                ct = replace_tgt_unk.replace_unk_from_string(ct, request, self.dic, remove_unk, normalize_unicode_unk, attempt_to_relocate_unk_source)

                out += ct + "\n"

                plots_list = []
                src_idx_list = src_data[num_t]
                tgt_idx_list = t[:-1]
                alignment = np.zeros((len(src_idx_list), len(tgt_idx_list)))
                sum_al =[0] * len(tgt_idx_list)

                for i in xrange(len(src_idx_list)):
                    for j in xrange(len(tgt_idx_list)):
                        alignment[i,j] = attn[j][0][i]
                    
                src_w = self.src_indexer.deconvert(src_idx_list, unk_tag = "#S_UNK#")
                tgt_w = self.tgt_indexer.deconvert(tgt_idx_list, unk_tag = "#T_UNK#")
                p1 = visualisation.make_alignment_figure(src_w, tgt_w, alignment, 'Sentence #%s' % str(request_number), 'below')
                plots_list.append(p1)
                p_all = visualisation.vplot(*plots_list)

                js_resources = bokeh.resources.INLINE.render_js()
                css_resources = bokeh.resources.INLINE.render_css()

                script, div = bokeh.embed.components(p_all, bokeh.resources.INLINE)
            print >>sys.stderr

        return out, script, div

class Server:

    def __init__(self, evaluator, parse_server_command, port=44666):
        self.evaluator = evaluator
        self.port = port
        self.parse_server_command = parse_server_command

    def __build_response(self, out, graph_data):
        response = {}
        response['out'] = out
        graphes = [];
        for gd in graph_data:
            script, div = gd
            graphes.append({'script': script, 'div': div})
        response['attn_graphes'] = graphes
        return json.dumps(response)

    def __handle_request(self, request):
        print(timestamped_msg("Handling request..."))
        root = ET.fromstring(request)
        article_id = root.attrib['id']
        beam_width = int(root.attrib['beam_width'])
        nb_steps = int(root.attrib['nb_steps'])
        nb_steps_ratio = None
        try:
            nb_steps_ratio = float(root.attrib['nb_steps_ratio'])
        except:
            pass
        groundhog = ('true' == root.attrib['groundhog'])
        force_finish = ('true' == root.attrib['force_finish'])
        use_raw_score = ('true' == root.attrib['use_raw_score'])
        prob_space_combination = ('true' == root.attrib['prob_space_combination'])
        remove_unk = ('true' == root.attrib['remove_unk'])
        normalize_unicode_unk = ('true' == root.attrib['normalize_unicode_unk'])
        attempt_to_relocate_unk_source = ('true' == root.attrib['attempt_to_relocate_unk_source'])
        print("Article id: %s" % article_id)
        out = ""
        graph_data = []
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
                    word = parts[2]
                    words.append(word)
            splitted_sentence = ' '.join(words)
            # print "splitted_sentence=" + splitted_sentence

            print(timestamped_msg("Translating sentence %d" % idx))
            translation, script, div = self.evaluator.eval(splitted_sentence.decode('utf-8'), idx, 
                beam_width, nb_steps, nb_steps_ratio, remove_unk, normalize_unicode_unk, attempt_to_relocate_unk_source,
                use_raw_score, groundhog, force_finish, prob_space_combination)
            out += translation
            graph_data.append((script.encode('utf-8'), div.encode('utf-8')))

        response = self.__build_response(out, graph_data)
        return response

    def start(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print(timestamped_msg("Start listening for requests on {0} port {1}...".format(socket.gethostname(), self.port)))
        server_socket.bind(('', self.port))
        server_socket.listen(5)

        while True:
            (client_socket, address) = server_socket.accept()
            client_socket.settimeout(2)
            print(timestamped_msg('Got connection from {0}'.format(address)))
            request = ''
            while True:
                try:
                    data = client_socket.recv(1024)
                    if data:
                        request += data
                    else:
                        break
                except:
                    break 
            response = self.__handle_request(request)
            client_socket.sendall(response)
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
    
    parser.add_argument("--ref", help = "target text")
    
    parser.add_argument("--gpu", type = int, help = "specify gpu number to use, if any")
    
    parser.add_argument("--max_nb_ex", type = int, help = "only use the first MAX_NB_EX examples")
    parser.add_argument("--mb_size", type = int, default= 80, help = "Minibatch size")
    parser.add_argument("--nb_batch_to_sort", type = int, default= 20, help = "Sort this many batches by size.")
    parser.add_argument("--beam_opt", default = False, action = "store_true")
    parser.add_argument("--tgt_unk_id", choices = ["attn", "id"], default = "align")
    
    # arguments for unk replace
    parser.add_argument("--dic")
    
    parser.add_argument("--reverse_training_config", help = "prefix of the trained model")
    parser.add_argument("--reverse_trained_model", help = "prefix of the trained model")
    
    parser.add_argument("--port", help = "port for listening request", default = 44666)
    parser.add_argument("--parse_server_command", help = "command to communicate with the parse-server")
    args = parser.parse_args()

    evaluator = Evaluator(args.training_config, args.trained_model, args.additional_training_config, args.additional_trained_model, 
                   args.reverse_training_config, args.reverse_trained_model, args.max_nb_ex, args.mb_size, args.beam_opt, 
                   args.tgt_unk_id, args.gpu, args.dic)

    server = Server(evaluator, args.parse_server_command, int(args.port))
    server.start()
    
if __name__ == '__main__':
    commandline() 
