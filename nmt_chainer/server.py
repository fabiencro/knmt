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

from eval import (create_encdec_from_config, create_and_load_encdec_from_files) 

import visualisation
import bleu_computer
import logging
import codecs
import traceback

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
            remove_unk, normalize_unicode_unk, attempt_to_relocate_unk_source, use_raw_score, groundhog, force_finish, prob_space_combination, attn_graph_width, attn_graph_height):
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
                
                script = ''
                div = '<div/>'
                ct = " ".join(self.tgt_indexer.deconvert(t, unk_tag = unk_replacer))
                if (ct != ''):
                    ct = replace_tgt_unk.replace_unk_from_string(ct, request, self.dic, remove_unk, normalize_unicode_unk, attempt_to_relocate_unk_source)

                    out += ct + "\n"

                    plots_list = []
                    src_idx_list = src_data[num_t]
                    tgt_idx_list = t[:-1]
                    alignment = np.zeros((len(src_idx_list), len(tgt_idx_list)))
                    sum_al =[0] * len(tgt_idx_list)

                    for i in xrange(len(src_idx_list)):
                        for j in xrange(len(tgt_idx_list)):
                            alignment[i,j] = attn[j][i]
                        
                    src_w = self.src_indexer.deconvert(src_idx_list, unk_tag = "#S_UNK#")
                    tgt_w = self.tgt_indexer.deconvert(tgt_idx_list, unk_tag = "#T_UNK#")
                    if (attn_graph_width > 0 and attn_graph_height > 0):
                        p1 = visualisation.make_alignment_figure(src_w, tgt_w, alignment, 'Sentence #%s' % str(request_number), 'below', plot_width = attn_graph_width, plot_height = attn_graph_height)
                        plots_list.append(p1)
                        p_all = visualisation.vplot(*plots_list)

                        js_resources = bokeh.resources.INLINE.render_js()
                        css_resources = bokeh.resources.INLINE.render_css()

                        script, div = bokeh.embed.components(p_all, bokeh.resources.INLINE)
            print >>sys.stderr

        return out, script, div

class Server:

    def __init__(self, evaluator, segmenter_command, segmenter_format = 'parse_server', port = 44666):
        self.evaluator = evaluator
        self.port = port
        self.segmenter_command = segmenter_command
        self.segmenter_format = segmenter_format

    def __build_error_response(self, error_lines):
        response = {}
        response['error'] = error_lines[-1]
        response['stacktrace'] = error_lines
        return json.dumps(response)
        
    def __build_successful_response(self, out, graph_data):
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
        article_id = root.get('id')
        try:
            attn_graph_width = int(root.get('attn_graph_width', 0))
        except:
            attn_graph_width = 0
        try:
            attn_graph_height = int(root.get('attn_graph_height', 0))
        except:
            attn_graph_height = 0
        beam_width = int(root.get('beam_width', 30))
        nb_steps = int(root.get('nb_steps', 50))
        nb_steps_ratio = None
        try:
            nb_steps_ratio = float(root.get('nb_steps_ratio', 1.2))
        except:
            pass
        groundhog = ('true' == root.get('groundhog', 'false'))
        force_finish = ('true' == root.get('force_finish', 'false'))
        use_raw_score = ('true' == root.get('use_raw_score', 'false'))
        prob_space_combination = ('true' == root.get('prob_space_combination', 'false'))
        remove_unk = ('true' == root.get('remove_unk', 'false'))
        normalize_unicode_unk = ('true' == root.get('normalize_unicode_unk', 'true'))
        attempt_to_relocate_unk_source = ('true' == root.get('attempt_to_relocate_unk_source', 'false'))
        print("Article id: %s" % article_id)
        out = ""
        graph_data = []
        sentences = root.findall('sentence')
        for idx, sentence in enumerate(sentences):
            text = sentence.findtext('i_sentence')
            # print "text=%s" % text
            
            cmd = self.segmenter_command % text
            # print "cmd=%s" % cmd
            parser_output = subprocess.check_output(cmd, shell=True)
            # print "parser_output=%s" % parser_output

            words = []
            if 'parse_server' == self.segmenter_format:
                for line in parser_output.split("\n"):
                    if (line.startswith('#')):
                        continue
                    elif (not line.strip()):
                        break
                    else:
                        parts = line.split("\t")
                        word = parts[2]
                        words.append(word)
            elif 'morph' == self.segmenter_format:
                for pair in parser_output.split(' '):
                    if pair != '':
                        word, pos = pair.split('_')
                        words.append(word)
            else:
                pass
            splitted_sentence = ' '.join(words)
            # print "splitted_sentence=" + splitted_sentence

            print(timestamped_msg("Translating sentence %d" % idx))
            translation, script, div = self.evaluator.eval(splitted_sentence.decode('utf-8'), idx, 
                beam_width, nb_steps, nb_steps_ratio, remove_unk, normalize_unicode_unk, attempt_to_relocate_unk_source,
                use_raw_score, groundhog, force_finish, prob_space_combination, attn_graph_width, attn_graph_height)
            out += translation
            graph_data.append((script.encode('utf-8'), div.encode('utf-8')))

        response = self.__build_successful_response(out, graph_data)
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
            try:
                response = self.__handle_request(request)
            except:
                traceback.print_exc()
                response = self.__build_error_response(traceback.format_exc().splitlines())
            client_socket.sendall(response)
            client_socket.close()
        
def timestamped_msg(msg):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    return "{0}: {1}".format(timestamp, msg) 

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
    parser.add_argument("--segmenter_command", help = "command to communicate with the segmenter server")
    parser.add_argument("--segmenter_format", help = "format to expect from the segmenter (parse_server, morph)", default = 'parse_server')
    args = parser.parse_args()

    evaluator = Evaluator(args.training_config, args.trained_model, args.additional_training_config, args.additional_trained_model, 
                   args.reverse_training_config, args.reverse_trained_model, args.max_nb_ex, args.mb_size, args.beam_opt, 
                   args.tgt_unk_id, args.gpu, args.dic)

    server = Server(evaluator, args.segmenter_command, args.segmenter_format, int(args.port))
    server.start()
    
if __name__ == '__main__':
    commandline() 
