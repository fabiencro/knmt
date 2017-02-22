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
from chainer import cuda
import logging
import sys
import tempfile

from nmt_chainer.dataprocessing.processors import build_dataset_one_side_pp
from nmt_chainer.translation.evaluation import beam_search_translate
from nmt_chainer.translation.eval import create_encdec
from nmt_chainer.translation.server_arg_parsing import make_config_server

import traceback

import time
import timeit
import socket
import threading
import SocketServer
import xml.etree.ElementTree as ET
import re
import subprocess
import bokeh.embed

logging.basicConfig()
log = logging.getLogger("rnns:server")
log.setLevel(logging.INFO)

class Translator:

    def __init__(self, config_server):
        self.config_server = config_server 
        self.encdec, self.eos_idx, self.src_indexer, self.tgt_indexer, self.reverse_encdec = create_encdec(config_server)
        if config_server.process.gpu is not None:
            self.encdec = self.encdec.to_gpu(config_server.process.gpu)
            
        self.encdec_list = [self.encdec]
        
    def translate(self, request, request_number, beam_width, beam_pruning_margin, nb_steps, nb_steps_ratio, 
            remove_unk, normalize_unicode_unk, attempt_to_relocate_unk_source, post_score_length_normalization, length_normalization_strength, groundhog, force_finish, prob_space_combination, attn_graph_width, attn_graph_height):
        from nmt_chainer.utilities import visualisation
        log.info("processing source string %s" % request)

        request_file = tempfile.NamedTemporaryFile()
        request_file.write(request.encode('utf-8'))
        request_file.seek(0)
        try:
            src_data, stats_src_pp = build_dataset_one_side_pp(request_file.name, self.src_indexer, max_nb_ex = self.config_server.process.max_nb_ex)
            log.info(stats_src_pp.make_report())

            tgt_data = None

            out = ''
            script = ''
            div = '<div/>'
            unk_mapping = []
            with cuda.get_device(self.config_server.process.gpu):
                translations_gen = beam_search_translate(
                            self.encdec, self.eos_idx, src_data, beam_width = beam_width, nb_steps = nb_steps,
                                            gpu = self.config_server.process.gpu, beam_pruning_margin = beam_pruning_margin, nb_steps_ratio = nb_steps_ratio,
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

                    if self.config_server.output.tgt_unk_id == "align":
                        def unk_replacer(num_pos, unk_id):
                            unk_pattern = "#T_UNK_%i#"
                            a = attn[num_pos]
                            xp = cuda.get_array_module(a)
                            src_pos = int(xp.argmax(a))
                            return unk_pattern%src_pos
                    elif self.config_server.output.tgt_unk_id == "id":
                        def unk_replacer(num_pos, unk_id):
                            unk_pattern = "#T_UNK_%i#"
                            return unk_pattern%unk_id         
                    else:
                        assert False
                    
                    ct = " ".join(self.tgt_indexer.deconvert_swallow(t, unk_tag = unk_replacer))
                    if (ct != ''):
                        unk_pattern = re.compile("#T_UNK_(\d+)#")
                        for idx, word in enumerate(ct.split(' ')):
                            match = unk_pattern.match(word)
                            if (match):
                                unk_mapping.append(match.group(1) + '-' + str(idx))    

                        from nmt_chainer.utilities import replace_tgt_unk
                        ct = replace_tgt_unk.replace_unk_from_string(ct, request, self.config_server.output.dic, remove_unk, normalize_unicode_unk, attempt_to_relocate_unk_source)

                        out += ct + "\n"
                        
                        plots_list = []
                        src_idx_list = src_data[num_t]
                        tgt_idx_list = t[:-1]
                        alignment = np.zeros((len(src_idx_list), len(tgt_idx_list)))
                        sum_al =[0] * len(tgt_idx_list)

                        for i in xrange(len(src_idx_list)):
                            for j in xrange(len(tgt_idx_list)):
                                alignment[i,j] = attn[j][i]
                            
                        src_w = self.src_indexer.deconvert_swallow(src_idx_list, unk_tag = "#S_UNK#")
                        tgt_w = self.tgt_indexer.deconvert_swallow(tgt_idx_list, unk_tag = "#T_UNK#")
                        if (attn_graph_width > 0 and attn_graph_height > 0):
                            p1 = visualisation.make_alignment_figure(src_w, tgt_w, alignment, title = '', toolbar_location = 'below', plot_width = attn_graph_width, plot_height = attn_graph_height)
                            plots_list.append(p1)
                            p_all = visualisation.Column(*plots_list)

                            script, div = bokeh.embed.components(p_all)

            print >>sys.stderr

        finally:
            request_file.close()

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
                    translation, script, div, unk_mapping = self.server.translator.translate(decoded_sentence, sentence_number, 
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

    def __init__(self, server_address, handler_class, segmenter_command, segmenter_format, translator):
        SocketServer.TCPServer.__init__(self, server_address, handler_class)
        self.segmenter_command = segmenter_command
        self.segmenter_format = segmenter_format
        self.translator = translator

def timestamped_msg(msg):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    return "{0}: {1}".format(timestamp, msg) 

def do_start_server(args):
    config_server = make_config_server(args)
    translator = Translator(config_server)
    server = Server((config_server.host, int(config_server.port)), RequestHandler, config_server.segmenter_command, config_server.segmenter_format, translator)
    ip, port = server.server_address
    log.info(timestamped_msg("Start listening for requests on {0}:{1}...".format(socket.gethostname(), port)))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        server.server_close()
    
    sys.exit(0)

if __name__ == '__main__':
    command_line() 
