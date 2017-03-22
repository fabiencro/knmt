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
import nmt_chainer.translation.eval
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
        from nmt_chainer.translation.eval import create_encdec
        self.encdec, self.eos_idx, self.src_indexer, self.tgt_indexer, self.reverse_encdec = create_encdec(
            config_server)
        if 'gpu' in config_server.process and config_server.process.gpu is not None:
            self.encdec = self.encdec.to_gpu(config_server.process.gpu)

        self.encdec_list = [self.encdec]

    def translate(self, sentence, beam_width, beam_pruning_margin, beam_score_coverage_penalty, beam_score_coverage_penalty_strength, nb_steps, nb_steps_ratio,
                  remove_unk, normalize_unicode_unk, attempt_to_relocate_unk_source, beam_score_length_normalization, beam_score_length_normalization_strength, post_score_length_normalization, post_score_length_normalization_strength,
                  post_score_coverage_penalty, post_score_coverage_penalty_strength,
                  groundhog, force_finish, prob_space_combination, attn_graph_width, attn_graph_height):
        from nmt_chainer.utilities import visualisation
        log.info("processing source string %s" % sentence)

        src_file = tempfile.NamedTemporaryFile()
        src_file.write(sentence.encode('utf-8'))
        src_file.seek(0)

        dest_file = tempfile.NamedTemporaryFile()
        rich_output_file = tempfile.NamedTemporaryFile()
        attn_graph_script_file = tempfile.NamedTemporaryFile()
        attn_graph_div_file = tempfile.NamedTemporaryFile()

        try:
            out = ''
            script = ''
            div = '<div/>'
            unk_mapping = []

            src_data, stats_src_pp = build_dataset_one_side_pp(src_file.name, self.src_indexer, max_nb_ex=self.config_server.process.max_nb_ex)

            from nmt_chainer.translation.eval import translate_to_file_with_beam_search
            translate_to_file_with_beam_search(dest_file.name, self.config_server.process.gpu, self.encdec, self.eos_idx, src_data, beam_width, beam_pruning_margin,
                                               beam_score_coverage_penalty=beam_score_coverage_penalty,
                                               beam_score_coverage_penalty_strength=beam_score_coverage_penalty_strength,
                                               nb_steps=nb_steps,
                                               nb_steps_ratio=nb_steps_ratio,
                                               beam_score_length_normalization=beam_score_length_normalization,
                                               beam_score_length_normalization_strength=beam_score_length_normalization_strength,
                                               post_score_length_normalization=post_score_length_normalization,
                                               post_score_length_normalization_strength=post_score_length_normalization_strength,
                                               post_score_coverage_penalty=post_score_coverage_penalty,
                                               post_score_coverage_penalty_strength=post_score_coverage_penalty_strength,
                                               groundhog=groundhog,
                                               tgt_unk_id=self.config_server.output.tgt_unk_id,
                                               tgt_indexer=self.tgt_indexer,
                                               force_finish=force_finish,
                                               prob_space_combination=prob_space_combination, reverse_encdec=self.reverse_encdec,
                                               generate_attention_html=(attn_graph_script_file.name, attn_graph_div_file.name),
                                               attn_graph_with_sum=False,
                                               attn_graph_attribs={'title': '', 'toolbar_location': 'below', 'plot_width': attn_graph_width, 'plot_height': attn_graph_height}, src_indexer=self.src_indexer,
                                               rich_output_filename=rich_output_file.name,
                                               use_unfinished_translation_if_none_found=False,
                                               replace_unk=True, src=sentence, dic=self.config_server.output.dic,
                                               remove_unk=remove_unk, normalize_unicode_unk=normalize_unicode_unk, attempt_to_relocate_unk_source=attempt_to_relocate_unk_source)

            dest_file.seek(0)
            out = dest_file.read()

            rich_output_file.seek(0)
            rich_output_data = json.loads(rich_output_file.read())
            unk_mapping = rich_output_data[0]['unk_mapping']

            attn_graph_script_file.seek(0)
            script = attn_graph_script_file.read()

            attn_graph_div_file.seek(0)
            div = attn_graph_div_file.read()

        finally:
            src_file.close()
            dest_file.close()
            rich_output_file.close()
            attn_graph_script_file.close()
            attn_graph_div_file.close()

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
                except BaseException:
                    attn_graph_width = 0
                try:
                    attn_graph_height = int(root.get('attn_graph_height', 0))
                except BaseException:
                    attn_graph_height = 0
                beam_width = int(root.get('beam_width', 30))
                nb_steps = int(root.get('nb_steps', 50))
                beam_pruning_margin = None
                try:
                    beam_pruning_margin = float(root.get('beam_pruning_margin'))
                except BaseException:
                    pass
                beam_score_coverage_penalty = root.get(
                    'beam_score_coverage_penalty', 'none')
                beam_score_coverage_penalty_strength = None
                try:
                    beam_score_coverage_penalty_strength = float(root.get('beam_score_coverage_penalty_strength', 0.2))
                except BaseException:
                    pass
                nb_steps_ratio = None
                try:
                    nb_steps_ratio = float(root.get('nb_steps_ratio', 1.2))
                except BaseException:
                    pass
                groundhog = ('true' == root.get('groundhog', 'false'))
                force_finish = ('true' == root.get('force_finish', 'false'))
                beam_score_length_normalization = root.get(
                    'beam_score_length_normalization', 'none')
                beam_score_length_normalization_strength = None
                try:
                    beam_score_length_normalization_strength = float(root.get('beam_score_length_normalization_strength', 0.2))
                except BaseException:
                    pass
                post_score_length_normalization = root.get(
                    'post_score_length_normalization', 'simple')
                post_score_length_normalization_strength = None
                try:
                    post_score_length_normalization_strength = float(root.get('post_score_length_normalization_strength', 0.2))
                except BaseException:
                    pass
                post_score_coverage_penalty = root.get(
                    'post_score_coverage_penalty', 'none')
                post_score_coverage_penalty_strength = None
                try:
                    post_score_coverage_penalty_strength = float(root.get('post_score_coverage_penalty_strength', 0.2))
                except BaseException:
                    pass
                prob_space_combination = (
                    'true' == root.get(
                        'prob_space_combination', 'false'))
                remove_unk = ('true' == root.get('remove_unk', 'false'))
                normalize_unicode_unk = (
                    'true' == root.get(
                        'normalize_unicode_unk', 'true'))
                log.info('normalize_unicode_unk=' + str(normalize_unicode_unk))
                attempt_to_relocate_unk_source = ('true' == root.get(
                    'attempt_to_relocate_unk_source', 'false'))
                log.info("Article id: %s" % article_id)
                out = ""
                graph_data = []
                segmented_input = []
                segmented_output = []
                mapping = []
                sentences = root.findall('sentence')
                for idx, sentence in enumerate(sentences):
                    sentence_number = sentence.get('id')
                    text = sentence.findtext('i_sentence').strip()
                    log.info("text=@@@%s@@@" % text)

                    cmd = self.server.segmenter_command % text
                    log.info("cmd=%s" % cmd)
                    start_cmd = timeit.default_timer()

                    parser_output = subprocess.check_output(cmd, shell=True)

                    log.info(
                        "Segmenter request processed in {} s.".format(
                            timeit.default_timer() - start_cmd))
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
                    translation, script, div, unk_mapping = self.server.translator.translate(decoded_sentence,
                                                                                             beam_width, beam_pruning_margin, beam_score_coverage_penalty, beam_score_coverage_penalty_strength, nb_steps, nb_steps_ratio, remove_unk, normalize_unicode_unk, attempt_to_relocate_unk_source,
                                                                                             beam_score_length_normalization, beam_score_length_normalization_strength, post_score_length_normalization, post_score_length_normalization_strength, post_score_coverage_penalty, post_score_coverage_penalty_strength,
                                                                                             groundhog, force_finish, prob_space_combination, attn_graph_width, attn_graph_height)
                    out += translation
                    segmented_input.append(splitted_sentence)
                    segmented_output.append(translation)
                    mapping.append(unk_mapping)
                    graph_data.append(
                        (script.encode('utf-8'), div.encode('utf-8')))

                    # There should always be only one sentence for now. - FB
                    break

                response['article_id'] = article_id
                response['sentence_number'] = sentence_number
                response['out'] = out
                response['segmented_input'] = segmented_input
                response['segmented_output'] = segmented_output
                response['mapping'] = map(lambda x: ' '.join(x), mapping)
                graphes = []
                for gd in graph_data:
                    script, div = gd
                    graphes.append({'script': script, 'div': div})
                response['attn_graphes'] = graphes
            except BaseException:
                traceback.print_exc()
                error_lines = traceback.format_exc().splitlines()
                response['error'] = error_lines[-1]
                response['stacktrace'] = error_lines

        log.info(
            "Request processed in {0} s. by {1}".format(
                timeit.default_timer() -
                start_request,
                cur_thread.name))

        response = json.dumps(response)
        self.request.sendall(response)


class Server(SocketServer.ThreadingMixIn, SocketServer.TCPServer):

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
            self,
            server_address,
            handler_class,
            segmenter_command,
            segmenter_format,
            translator):
        SocketServer.TCPServer.__init__(self, server_address, handler_class)
        self.segmenter_command = segmenter_command
        self.segmenter_format = segmenter_format
        self.translator = translator


def timestamped_msg(msg):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    return "{0}: {1}".format(timestamp, msg)


def do_start_server(config_server):
    translator = Translator(config_server)
    server_host, server_port = config_server.process.server.split(":")
    server = Server(
        (server_host,
         int(server_port)),
        RequestHandler,
        config_server.process.segmenter_command,
        config_server.process.segmenter_format,
        translator)
    ip, port = server.server_address
    log.info(
        timestamped_msg(
            "Start listening for requests on {0}:{1}...".format(
                socket.gethostname(),
                port)))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        server.server_close()

    sys.exit(0)


if __name__ == '__main__':
    command_line()
