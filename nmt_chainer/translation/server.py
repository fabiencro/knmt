#!/usr/bin/env python
"""server.py: Process requests to RNNSearch"""
from __future__ import absolute_import, division, print_function, unicode_literals
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
import logging.config
import sys
import tempfile
import os
from os import listdir
from os.path import isfile, join, dirname, basename
import six

from nmt_chainer.dataprocessing.processors import build_dataset_one_side_pp
import nmt_chainer.translation.eval

import traceback

import time
import timeit
import socket
import threading
import xml.etree.ElementTree as ET
import re
import subprocess
import bokeh.embed

PAGE_SIZE = 5000

log = None

class TranslatorThread(threading.Thread):
    """Thread class with a stop() method useful to interrupt the translation before it ends."""

    def __init__(self, 
                 dest_filename, gpu, encdec, eos_idx, src_data, beam_width, beam_pruning_margin,
                 beam_score_coverage_penalty=None,
                 beam_score_coverage_penalty_strength=None,
                 nb_steps=None,
                 nb_steps_ratio=None,
                 beam_score_length_normalization=None,
                 beam_score_length_normalization_strength=None,
                 post_score_length_normalization=None,
                 post_score_length_normalization_strength=None,
                 post_score_coverage_penalty=None,
                 post_score_coverage_penalty_strength=None,
                 groundhog=None,
                 tgt_unk_id=None,
                 tgt_indexer=None,
                 force_finish=None,
                 prob_space_combination=None, reverse_encdec=None,
                 generate_attention_html=None,
                 attn_graph_with_sum=None,
                 attn_graph_attribs=None, src_indexer=None,
                 rich_output_filename=None,
                 use_unfinished_translation_if_none_found=None,
                 replace_unk=None, src=None, dic=None,
                 remove_unk=None, normalize_unicode_unk=None, attempt_to_relocate_unk_source=None):
        threading.Thread.__init__(self)
        self._stop_event = threading.Event()
        self.dest_filename = dest_filename
        self.gpu = gpu
        self.encdec = encdec
        self.eos_idx = eos_idx
        self.src_data = src_data
        self.beam_width = beam_width
        self.beam_pruning_margin = beam_pruning_margin
        self.beam_score_coverage_penalty=beam_score_coverage_penalty
        self.beam_score_coverage_penalty_strength=beam_score_coverage_penalty_strength
        self.nb_steps=nb_steps
        self.nb_steps_ratio=nb_steps_ratio
        self.beam_score_length_normalization=beam_score_length_normalization
        self.beam_score_length_normalization_strength=beam_score_length_normalization_strength
        self.post_score_length_normalization=post_score_length_normalization
        self.post_score_length_normalization_strength=post_score_length_normalization_strength
        self.post_score_coverage_penalty=post_score_coverage_penalty
        self.post_score_coverage_penalty_strength=post_score_coverage_penalty_strength
        self.groundhog=groundhog
        self.tgt_unk_id=tgt_unk_id
        self.tgt_indexer=tgt_indexer
        self.force_finish=force_finish
        self.prob_space_combination=prob_space_combination
        self.reverse_encdec=reverse_encdec
        self.generate_attention_html=generate_attention_html
        self.attn_graph_with_sum=attn_graph_with_sum
        self.attn_graph_attribs=attn_graph_attribs
        self.src_indexer=src_indexer
        self.rich_output_filename=rich_output_filename
        self.use_unfinished_translation_if_none_found=use_unfinished_translation_if_none_found
        self.replace_unk=replace_unk
        self.src=src
        self.dic=dic
        self.remove_unk=remove_unk
        self.normalize_unicode_unk=normalize_unicode_unk
        self.attempt_to_relocate_unk_source=attempt_to_relocate_unk_source

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        from nmt_chainer.translation.eval import translate_to_file_with_beam_search
        translate_to_file_with_beam_search(self.dest_filename, self.gpu, self.encdec, self.eos_idx, self.src_data, self.beam_width, self.beam_pruning_margin,
                                           beam_score_coverage_penalty=self.beam_score_coverage_penalty,
                                           beam_score_coverage_penalty_strength=self.beam_score_coverage_penalty_strength,
                                           nb_steps=self.nb_steps,
                                           nb_steps_ratio=self.nb_steps_ratio,
                                           beam_score_length_normalization=self.beam_score_length_normalization,
                                           beam_score_length_normalization_strength=self.beam_score_length_normalization_strength,
                                           post_score_length_normalization=self.post_score_length_normalization,
                                           post_score_length_normalization_strength=self.post_score_length_normalization_strength,
                                           post_score_coverage_penalty=self.post_score_coverage_penalty,
                                           post_score_coverage_penalty_strength=self.post_score_coverage_penalty_strength,
                                           groundhog=self.groundhog,
                                           tgt_unk_id=self.tgt_unk_id,
                                           tgt_indexer=self.tgt_indexer,
                                           force_finish=self.force_finish,
                                           prob_space_combination=self.prob_space_combination, reverse_encdec=self.reverse_encdec,
                                           generate_attention_html=self.generate_attention_html,
                                           attn_graph_with_sum=self.attn_graph_with_sum,
                                           attn_graph_attribs=self.attn_graph_attribs, src_indexer=self.src_indexer,
                                           rich_output_filename=self.rich_output_filename,
                                           use_unfinished_translation_if_none_found=self.use_unfinished_translation_if_none_found,
                                           replace_unk=self.replace_unk, src=self.src, dic=self.dic,
                                           remove_unk=self.remove_unk, normalize_unicode_unk=self.normalize_unicode_unk, attempt_to_relocate_unk_source=self.attempt_to_relocate_unk_source, 
                                           thread=self)

class Translator(object):

    def __init__(self, config_server):
        self.config_server = config_server
        from nmt_chainer.translation.eval import create_encdec
        self.encdec, self.eos_idx, self.src_indexer, self.tgt_indexer, self.reverse_encdec, model_infos_list = create_encdec(
            config_server)

        self.encdec_list = [self.encdec]
        self.translator_thread = None

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

        try:
            out = ''
            unk_mapping = []

            src_data, stats_src_pp = build_dataset_one_side_pp(src_file.name, self.src_indexer, max_nb_ex=self.config_server.process.max_nb_ex)

            self.translator_thread = TranslatorThread(dest_file.name, self.config_server.process.gpu, self.encdec, self.eos_idx, src_data, beam_width, beam_pruning_margin,
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
                                                 generate_attention_html=None,
                                                 attn_graph_with_sum=False,
                                                 attn_graph_attribs={'title': '', 'toolbar_location': 'below', 'plot_width': attn_graph_width, 'plot_height': attn_graph_height}, src_indexer=self.src_indexer,
                                                 rich_output_filename=rich_output_file.name,
                                                 use_unfinished_translation_if_none_found=True,
                                                 replace_unk=True, src=sentence, dic=self.config_server.output.dic,
                                                 remove_unk=remove_unk, normalize_unicode_unk=normalize_unicode_unk, attempt_to_relocate_unk_source=attempt_to_relocate_unk_source)
            self.translator_thread.start()
            self.translator_thread.join()

            dest_file.seek(0)
            out = dest_file.read()

            rich_output_file.seek(0)
            rich_output_data = json.loads(rich_output_file.read().decode('utf-8'))
            if len(rich_output_data) > 0 and 'unk_mapping' in rich_output_data[0]:
                unk_mapping = rich_output_data[0]['unk_mapping']

        finally:
            src_file.close()
            dest_file.close()
            rich_output_file.close()

        return out, unk_mapping

    def stop(self):
        if self.translator_thread:
            self.translator_thread.stop()


class RequestHandler(six.moves.socketserver.BaseRequestHandler):

    def handle(self):
        start_request = timeit.default_timer()
        log.info("Handling request...")
        data = self.request.recv(4096)

        response = {}
        if (data):
            try:
                cur_thread = threading.current_thread()

                log.info("request={0}".format(data))

                if "get_log_files" in data:
                    all_log_files = []
                    for handler in log.root.handlers:
                        if hasattr(handler, 'baseFilename'):
                            log_dir = os.path.dirname(handler.baseFilename)
                            log_base_fn = os.path.basename(handler.baseFilename)
                            log_files = [f for f in listdir(log_dir) if isfile(join(log_dir, f)) and f.startswith(log_base_fn)]
                            all_log_files += log_files
                    response['log_files'] = all_log_files
                elif "get_log_file" in data:
                    root = ET.fromstring(data)
                    filename = root.get('filename') 
                    try:
                        page = int(root.get('page'))
                    except BaseException:
                        page = 1
                    for handler in log.root.handlers:
                        if hasattr(handler, 'baseFilename'):
                            log_dir = os.path.dirname(handler.baseFilename)
                            log_base_fn = os.path.basename(handler.baseFilename)
                            log_file = "{0}/{1}".format(log_dir, filename)
                            if log_base_fn in filename and os.path.isfile(log_file):
                                page_count = 1
                                log_file_content = ''
                                line_in_page = 0
                                start = (page - 1) * PAGE_SIZE
                                stop = start + PAGE_SIZE
                                with open(log_file, 'r') as f:
                                    for line, str_line in enumerate(f):
                                        if line >= start and line < stop:
                                            log_file_content += str_line
                                        line_in_page += 1
                                        if line_in_page == PAGE_SIZE:
                                            page_count += 1
                                            line_in_page = 0
                                response['content'] = log_file_content
                                response['page'] = page
                                response['pageCount'] = page_count
                                response['status'] = 'OK'
                                break
                    else:
                        response['status'] = 'NOT FOUND'
                elif "cancel_translation" in data:
                    self.server.translator.stop()
                else:
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
                    log.debug('normalize_unicode_unk=' + str(normalize_unicode_unk))
                    attempt_to_relocate_unk_source = ('true' == root.get(
                        'attempt_to_relocate_unk_source', 'false'))
                    log.debug("Article id: %s" % article_id)
                    in_ = ""
                    out = ""
                    segmented_input = []
                    segmented_output = []
                    mapping = []
                    sentences = root.findall('sentence')
                    for idx, sentence in enumerate(sentences):
                        sentence_number = sentence.get('id')
                        text = sentence.findtext('i_sentence').strip()
                        log.info("text=%s" % text)

                        # cmd = self.server.segmenter_command % text.replace("'", "'\\''").encode('utf-8')
                        cmd = self.server.segmenter_command % text.replace("'", "'\\''") # p3
                        log.info("cmd=%s" % cmd)
                        start_cmd = timeit.default_timer()

                        #parser_output = subprocess.check_output(cmd, shell=True)
                        parser_output = subprocess.check_output(cmd, shell=True, universal_newlines=True) # p3

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

                        #decoded_sentence = splitted_sentence.decode('utf-8')
                        # log.info("decoded_sentence={0}".format(decoded_sentence))
                        # translation, unk_mapping = self.server.translator.translate(decoded_sentence,
                        translation, unk_mapping = self.server.translator.translate(splitted_sentence,
                                                                                                 beam_width, beam_pruning_margin, beam_score_coverage_penalty, beam_score_coverage_penalty_strength, nb_steps, nb_steps_ratio, remove_unk, normalize_unicode_unk, attempt_to_relocate_unk_source,
                                                                                                 beam_score_length_normalization, beam_score_length_normalization_strength, post_score_length_normalization, post_score_length_normalization_strength, post_score_coverage_penalty, post_score_coverage_penalty_strength,
                                                                                                 groundhog, force_finish, prob_space_combination, attn_graph_width, attn_graph_height)
                        # in_ += decoded_sentence
                        in_ += text
                        out += translation

                        if self.server.pp_command is not None:
                            def apply_pp(str):
                                pp_cmd = self.server.pp_command % out.replace("'", "''")
                                log.info("pp_cmd=%s" % pp_cmd)

                                start_pp_cmd = timeit.default_timer()

                                pp_output = subprocess.check_output(pp_cmd, shell=True, universal_newlines=True)

                                log.info("Postprocessor request processed in {0} s.".format(timeit.default_timer() - start_pp_cmd))
                                log.info("pp_output={0}".format(pp_output))
                                return pp_output
                            out = apply_pp(out)

                        segmented_input.append(splitted_sentence)
                        segmented_output.append(translation)
                        mapping.append(unk_mapping)

                        # There should always be only one sentence for now. - FB
                        break

                    response['article_id'] = article_id
                    response['sentence_number'] = sentence_number
                    response['in_'] = in_
                    response['out'] = out
                    # log.info("in_={0}".format(in_))
                    log.info("out={0}".format(out))
                    response['segmented_input'] = segmented_input
                    response['segmented_output'] = segmented_output
                    response['mapping'] = map(lambda x: ' '.join(x), mapping)
            except BaseException:
                traceback.print_exc()
                error_lines = traceback.format_exc().splitlines()
                response['error'] = error_lines[-1]
                response['stacktrace'] = error_lines

        log.info("Request processed in {0} s. by {1}".format(timeit.default_timer() - start_request, cur_thread.name))

        response = json.dumps(response)
        # self.request.sendall(response)
        self.request.sendall(response.encode('utf-8')) # p3


class Server(six.moves.socketserver.ThreadingMixIn, six.moves.socketserver.TCPServer):

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
            self,
            server_address,
            handler_class,
            segmenter_command,
            segmenter_format,
            translator,
            pp_command):
        six.moves.socketserver.TCPServer.__init__(self, server_address, handler_class)
        self.segmenter_command = segmenter_command
        self.segmenter_format = segmenter_format
        self.translator = translator
        self.pp_command = pp_command


def do_start_server(config_server):
    if config_server.output.log_config:
        logging.config.fileConfig(config_server.output.log_config)
    global log
    log = logging.getLogger("default")
    log.setLevel(logging.INFO)

    translator = Translator(config_server)
    server_host, server_port = config_server.process.server.split(":")
    server = Server(
        (server_host,
         int(server_port)),
        RequestHandler,
        config_server.process.segmenter_command,
        config_server.process.segmenter_format,
        translator,
        config_server.process.pp_command)
    ip, port = server.server_address
    log.info("Start listening for requests on {0}:{1}...".format(socket.gethostname(), port))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        server.server_close()

    sys.exit(0)
