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


import bleu_computer
import codecs
import traceback

import rnn_cells

import time
import timeit
import socket
import threading
import SocketServer
import xml.etree.ElementTree as ET
import re
import subprocess
import replace_tgt_unk
import bokeh.embed

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
            
    def eval(self, request, request_number, beam_width, beam_pruning_margin, nb_steps, nb_steps_ratio, 
            remove_unk, normalize_unicode_unk, attempt_to_relocate_unk_source, post_score_length_normalization, length_normalization_strength, groundhog, force_finish, prob_space_combination, attn_graph_width, attn_graph_height):
        import visualisation
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
        unk_mapping = []
        with cuda.get_device(self.gpu):
            translations_gen = beam_search_translate(
                        self.encdec, self.eos_idx, src_data, beam_width = beam_width, beam_pruning_margin = beam_pruning_margin, 
                                        nb_steps = nb_steps, 
                                        gpu = self.gpu, beam_opt = self.beam_opt, nb_steps_ratio = nb_steps_ratio,
                                        need_attention = True, post_score_length_normalization = post_score_length_normalization, 
                                        length_normalization_strength = length_normalization_strength,
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
                    unk_pattern = re.compile("#T_UNK_(\d+)#")
                    for idx, word in enumerate(ct.split(' ')):
                        match = unk_pattern.match(word)
                        if (match):
                            unk_mapping.append(match.group(1) + '-' + str(idx))    

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
                        p1 = visualisation.make_alignment_figure(src_w, tgt_w, alignment, title = '', toolbar_location = 'below', plot_width = attn_graph_width, plot_height = attn_graph_height)
                        plots_list.append(p1)
                        p_all = visualisation.Column(*plots_list)

                        script, div = bokeh.embed.components(p_all)

            print >>sys.stderr

        return out, script, div, unk_mapping

class RequestHandler(SocketServer.BaseRequestHandler):

    def handle(self):
        start_request = timeit.default_timer()
        log.info(timestamped_msg("Handling request..."))
        data = self.request.recv(4096)

        response = {}
        if (data):
            try:
                cur_thread = threading.current_thread()

                log.info("data={0}".format(data))
                root = ET.fromstring(data)
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
                beam_pruning_margin = None
                try:
                    beam_pruning_margin = float(root.get('beam_pruning_margin'))
                except:
                    pass
                nb_steps_ratio = None
                try:
                    nb_steps_ratio = float(root.get('nb_steps_ratio', 1.2))
                except:
                    pass
                groundhog = ('true' == root.get('groundhog', 'false'))
                force_finish = ('true' == root.get('force_finish', 'false'))
                post_score_length_normalization = root.get('post_score_length_normalization', 'simple')
                length_normalization_strength = None
                try:
                    length_normalization_strength = float(root.get('length_normalization_strength', 0.2))
                except:
                    pass
                prob_space_combination = ('true' == root.get('prob_space_combination', 'false'))
                remove_unk = ('true' == root.get('remove_unk', 'false'))
                normalize_unicode_unk = ('true' == root.get('normalize_unicode_unk', 'true'))
                log.info('normalize_unicode_unk=' + str(normalize_unicode_unk))
                attempt_to_relocate_unk_source = ('true' == root.get('attempt_to_relocate_unk_source', 'false'))
                log.info("Article id: %s" % article_id)
                out = ""
                graph_data = []
                segmented_input = []
                segmented_output = []
                mapping = []
                sentences = root.findall('sentence')
                for idx, sentence in enumerate(sentences):
                    sentence_number = sentence.get('id');
                    text = sentence.findtext('i_sentence').strip()
                    log.info("text=@@@%s@@@" % text)
                    
                    cmd = self.server.segmenter_command % text
                    log.info("cmd=%s" % cmd)
                    start_cmd = timeit.default_timer()

                    parser_output = subprocess.check_output(cmd, shell=True)

                    log.info("Segmenter request processed in {} s.".format(timeit.default_timer() - start_cmd))
                    log.info("parser_output=%s" % parser_output)

                    words = []
                    if 'parse_server' == self.server.segmenter_format:
                        for line in parser_output.split("\n"):
                            if (line.startswith('#')):
                                continue
                            elif (not line.strip()):
                                break
                            else:
                                parts = line.split("\t")
                                word = parts[2]
                                words.append(word)
                    elif 'morph' == self.server.segmenter_format:
                        for pair in parser_output.split(' '):
                            if pair != '':
                                word, pos = pair.split('_')
                                words.append(word)
                    elif 'plain' == self.server.segmenter_format:
                        words = parser_output.split(' ')
                    else:
                        pass
                    splitted_sentence = ' '.join(words)
                    # log.info("splitted_sentence=" + splitted_sentence)

                    log.info(timestamped_msg("Translating sentence %d" % idx))
                    decoded_sentence = splitted_sentence.decode('utf-8')
                    translation, script, div, unk_mapping = self.server.evaluator.eval(decoded_sentence, sentence_number, 
                        beam_width, beam_pruning_margin, nb_steps, nb_steps_ratio, remove_unk, normalize_unicode_unk, attempt_to_relocate_unk_source,
                        post_score_length_normalization, length_normalization_strength, groundhog, force_finish, prob_space_combination, attn_graph_width, attn_graph_height)
                    out += translation
                    segmented_input.append(splitted_sentence)
                    segmented_output.append(translation)
                    mapping.append(unk_mapping)
                    graph_data.append((script.encode('utf-8'), div.encode('utf-8')))

                    # There should always be only one sentence for now. - FB
                    break

                response['article_id'] = article_id
                response['sentence_number'] = sentence_number
                response['out'] = out
                response['segmented_input'] = segmented_input
                response['segmented_output'] = segmented_output
                response['mapping'] = map(lambda x: ' '.join(x), mapping)
                graphes = [];
                for gd in graph_data:
                    script, div = gd
                    graphes.append({'script': script, 'div': div})
                response['attn_graphes'] = graphes
            except:
                traceback.print_exc()
                error_lines = traceback.format_exc().splitlines()
                response['error'] = error_lines[-1]
                response['stacktrace'] = error_lines

        log.info("Request processed in {0} s. by {1}".format(timeit.default_timer() - start_request, cur_thread.name))

        response = json.dumps(response)
        self.request.sendall(response)

class Server(SocketServer.ThreadingMixIn, SocketServer.TCPServer):

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, handler_class, segmenter_command, segmenter_format, evaluator):
        SocketServer.TCPServer.__init__(self, server_address, handler_class)
        self.segmenter_command = segmenter_command
        self.segmenter_format = segmenter_format
        self.evaluator = evaluator

def timestamped_msg(msg):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    return "{0}: {1}".format(timestamp, msg) 

def define_parser(parser):
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
    
    parser.add_argument("--netiface", help = "network interface for listening request", default = 'eth0')
    parser.add_argument("--port", help = "port for listening request", default = 44666)
    parser.add_argument("--segmenter_command", help = "command to communicate with the segmenter server")
    parser.add_argument("--segmenter_format", help = "format to expect from the segmenter (parse_server, morph)", default = 'parse_server')

def command_line(arguments = None):
    import argparse
    parser = argparse.ArgumentParser(description= "Launch a RNNSearch server", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    define_parser(parser)
    args = parser.parse_args(args = arguments)
    do_start_server(args)
   
def do_start_server(args):
    evaluator = Evaluator(args.training_config, args.trained_model, args.additional_training_config, args.additional_trained_model, 
                   args.reverse_training_config, args.reverse_trained_model, args.max_nb_ex, args.mb_size, args.beam_opt, 
                   args.tgt_unk_id, args.gpu, args.dic)

    retrieve_ip_cmd = "/sbin/ifconfig | grep -A1 '{0}' | grep 'inet addr' | cut -f 2 -d ':' | cut -f 1 -d ' '".format(args.netiface)
    external_ip = subprocess.check_output(retrieve_ip_cmd, shell=True) 
    server = Server((external_ip, int(args.port)), RequestHandler, args.segmenter_command, args.segmenter_format, evaluator)
    ip, port = server.server_address
    log.info(timestamped_msg("Start listening for requests on {0}({1}) port {2}...".format(socket.gethostname(), external_ip, port)))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        server.server_close()
    
    sys.exit(0)

if __name__ == '__main__':
    command_line() 
