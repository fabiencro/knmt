#!/usr/bin/env python
"""multiserver.py: Translation server that works asynchronously and dispatches requests to multiple translation servers."""
__author__ = "Frederic Bergeron"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "bergeron@pa.jst.jp"
__status__ = "Development"

import datetime
import json
# import numpy as np
# from chainer import cuda
import logging
import sys
# import tempfile
# 
# from nmt_chainer.dataprocessing.processors import build_dataset_one_side_pp
# import nmt_chainer.translation.eval
# 
# import traceback
# 
import time
import timeit
import socket
import threading
import SocketServer
# import xml.etree.ElementTree as ET
# import re
# import subprocess

import Queue

logging.basicConfig()
log = logging.getLogger("rnns:server")
log.setLevel(logging.INFO)

# class TranslationResponse:
# 
#     def __init__(self, session_id, article_id, sentence_id, out, segmented_input, segmented_output, mapping, attn_graphes):
#         self.session_id = session_id
#         self.article_id = article_id
#         self.sentence_id = sentence_id
#         self.out = out
#         self.segmented_input = segmented_input
#         self.segmented_output = segmented_output
#         self.mapping = mapping
#         self.attn_graphes = attn_graphes
# 
#     def __hash__(self):
#         return hash((self.session_id, self.article_id, self.sentence_id, self.out))
# 
#     def __repr__(self):
#         return "{0}/{1} from {2}: {3}".format(self.article_id, self.sentence_id, self.session_id, self.out)
# 
# 
# class TranslationRequest:
# 
#     def __init__(self, session_id, article_id, sentence_id, sentence, attn_graph_width=0, attn_graph_height=0, 
#                  beam_width=30, nb_steps=50, beam_pruning_margin=0, 
#                  beam_score_coverage_penalty="none", beam_score_coverage_penalty_strength=0.2, 
#                  nb_steps_ratio=1.2, groundhog=False, force_finish=False, 
#                  beam_score_length_normalization="none", beam_score_length_normalization_strength=0.2, 
#                  post_score_length_normalization="simple", post_score_length_normalization_strength=0.2, 
#                  post_score_coverage_penalty="none", post_score_coverage_penalty_strength=0.2, 
#                  prob_space_combination=False, remove_unk=False, normalize_unicode_unk=True,
#                  attempt_to_relocate_unk_source=False):
#         self.session_id = session_id
#         self.article_id = article_id
#         self.sentence_id = sentence_id
#         self.sentence = sentence
#         self.attn_graph_width = attn_graph_width
#         self.attn_graph_height = attn_graph_height
#         self.beam_width = beam_width
#         self.nb_steps = nb_steps
#         self.beam_pruning_margin = beam_pruning_margin
#         self.beam_score_coverage_penalty = beam_score_coverage_penalty
#         self.beam_score_coverage_penalty_strength = beam_score_coverage_penalty_strength
#         self.nb_steps_ratio = nb_steps_ratio
#         self.groundhog = groundhog
#         self.force_finish = force_finish
#         self.beam_score_length_normalization = beam_score_length_normalization
#         self.beam_score_length_normalization_strength = beam_score_length_normalization_strength
#         self.post_score_length_normalization = post_score_length_normalization
#         self.post_score_length_normalization_strength = post_score_length_normalization_strength
#         self.post_score_coverage_penalty = post_score_coverage_penalty
#         self.post_score_coverage_penalty_strength = post_score_coverage_penalty_strength
#         self.prob_space_combination = prob_space_combination
#         self.remove_unk = remove_unk
#         self.normalize_unicode_unk = normalize_unicode_unk
#         self.attempt_to_relocate_unk_source = attempt_to_relocate_unk_source
# 
#     def __hash__(self):
#         return hash((self.session_id, self.article_id, self.sentence_id, self.sentence))
# 
#     def __repr__(self):
#         return "{0}/{1} from {2}: {3}".format(self.article_id, self.sentence_id, self.session_id, self.sentence)
# 
# 
# # The RequestHandler should parse the XML and process the 2 kinds of requests that we have.
# # Either a translation request or just a checkup request that will return responses if available.
# 
class RequestHandler(SocketServer.BaseRequestHandler, object):

    def __init__(self, *args, **kwargs):
        self.translation_request_queue = Queue.Queue()
        self.clients = {}
        super(RequestHandler, self).__init__(*args, **kwargs)

    def handle(self):
        log.info("handle request")
        pass
#         start_request = timeit.default_timer()
#         log.info(timestamped_msg("Handling request..."))
#         data = self.request.recv(4096)
# 
#         response = {}
#         if (data):
#             try:
#                 cur_thread = threading.current_thread()
# 
#                 log.info("data={0}".format(data))
#                 root = ET.fromstring(data)
#                 session_id = root.get('session_id')
#                 if root.tag == 'article':
#                     article_id = root.get('id')
#                     try:
#                         attn_graph_width = int(root.get('attn_graph_width', 0))
#                     except BaseException:
#                         attn_graph_width = 0
#                     try:
#                         attn_graph_height = int(root.get('attn_graph_height', 0))
#                     except BaseException:
#                         attn_graph_height = 0
#                     beam_width = int(root.get('beam_width', 30))
#                     nb_steps = int(root.get('nb_steps', 50))
#                     beam_pruning_margin = None
#                     try:
#                         beam_pruning_margin = float(root.get('beam_pruning_margin'))
#                     except BaseException:
#                         pass
#                     beam_score_coverage_penalty = root.get('beam_score_coverage_penalty', 'none')
#                     beam_score_coverage_penalty_strength = None
#                     try:
#                         beam_score_coverage_penalty_strength = float(root.get('beam_score_coverage_penalty_strength', 0.2))
#                     except BaseException:
#                         pass
#                     nb_steps_ratio = None
#                     try:
#                         nb_steps_ratio = float(root.get('nb_steps_ratio', 1.2))
#                     except BaseException:
#                         pass
#                     groundhog = ('true' == root.get('groundhog', 'false'))
#                     force_finish = ('true' == root.get('force_finish', 'false'))
#                     beam_score_length_normalization = root.get('beam_score_length_normalization', 'none')
#                     beam_score_length_normalization_strength = None
#                     try:
#                         beam_score_length_normalization_strength = float(root.get('beam_score_length_normalization_strength', 0.2))
#                     except BaseException:
#                         pass
#                     post_score_length_normalization = root.get('post_score_length_normalization', 'simple')
#                     post_score_length_normalization_strength = None
#                     try:
#                         post_score_length_normalization_strength = float(root.get('post_score_length_normalization_strength', 0.2))
#                     except BaseException:
#                         pass
#                     post_score_coverage_penalty = root.get('post_score_coverage_penalty', 'none')
#                     post_score_coverage_penalty_strength = None
#                     try:
#                         post_score_coverage_penalty_strength = float(root.get('post_score_coverage_penalty_strength', 0.2))
#                     except BaseException:
#                         pass
#                     prob_space_combination = ('true' == root.get('prob_space_combination', 'false'))
#                     remove_unk = ('true' == root.get('remove_unk', 'false'))
#                     normalize_unicode_unk = ('true' == root.get('normalize_unicode_unk', 'true'))
#                     attempt_to_relocate_unk_source = ('true' == root.get('attempt_to_relocate_unk_source', 'false'))
# 
#                     sentences = root.findall('sentence')
#                     for idx, sentence in enumerate(sentences):
#                         sentence_id = sentence.get('id')
#                         sentence_text = sentence.findtext('i_sentence').strip()
#                         # I suspect that a different instance of TranslationRequest is used each time.
#                         # Because of that, the request queue never grows.  It always contains 1 element.
#                         # So the instance should rather point to the server and the server should have the
#                         # request queue.
#                         translation_request = TranslationRequest(session_id, 
#                                                                  article_id, sentence_id, sentence_text,
#                                                                  attn_graph_width, attn_graph_height, 
#                                                                  beam_width, nb_steps, beam_pruning_margin, 
#                                                                  beam_score_coverage_penalty, beam_score_coverage_penalty_strength, 
#                                                                  nb_steps_ratio, groundhog, force_finish, 
#                                                                  beam_score_length_normalization, beam_score_length_normalization_strength, 
#                                                                  post_score_length_normalization, post_score_length_normalization_strength, 
#                                                                  post_score_coverage_penalty, post_score_coverage_penalty_strength, 
#                                                                  prob_space_combination, remove_unk, normalize_unicode_unk,
#                                                                  )
#                         log.info("req={0} h={1}".format(translation_request, translation_request.__hash__()))
#                         self.translation_request_queue.put(translation_request)
#                         log.info("q={0} sz={1}".format(self.translation_request_queue, self.translation_request_queue.qsize()))
#                         # There should always be only one sentence for now. - FB
#                         break
# 
#                 # if clients.has_key(session_id):
#                 #     for translation_response in clients[session_id]:
#                 #         response[translation_response.article_id] =  
#                 #     # Check if we have some translation response, package them and return them.
# 
#                 # Return an empty 
# 
#                     #     # There should always be only one sentence for now. - FB
#                     #     break
# 
#                     # response['article_id'] = article_id
#                     # response['sentence_number'] = sentence_number
#                     # response['out'] = out
#                     # response['segmented_input'] = segmented_input
#                     # response['segmented_output'] = segmented_output
#                     # response['mapping'] = map(lambda x: ' '.join(x), mapping)
#                     # graphes = []
#                     # for gd in graph_data:
#                     #     script, div = gd
#                     #     graphes.append({'script': script, 'div': div})
#                     # response['attn_graphes'] = graphes
# 
#                 # response['article_id'] = ''
#                 # response['sentence_number'] = ''
#                 # response['out'] = ''
#                 # response['segmented_input'] = ''
#                 # response['segmented_output'] = ''
#                 # response['mapping'] = map(lambda x: ' '.join(x), mapping)
#                 # response['attn_graphes'] = ''
#             except BaseException:
#                 traceback.print_exc()
#                 error_lines = traceback.format_exc().splitlines()
#                 response['error'] = error_lines[-1]
#                 response['stacktrace'] = error_lines
# 
#         log.info(
#             "Request processed in {0} s. by {1}".format(
#                 timeit.default_timer() -
#                 start_request,
#                 cur_thread.name))
# 
#         response = json.dumps(response)
#         self.request.sendall(response)


class Manager(object):

    def __init__(self):
        self.clients = {}
        self.translation_request_queue = Queue.Queue()
    

class Server(SocketServer.ThreadingMixIn, SocketServer.TCPServer):

    daemon_threads = True
    allow_reuse_address = True

    def make_request_handler(self, manager):
        class ServerRequestHandler(SocketServer.BaseRequestHandler, object):

            def __init__(self, *args, **kwargs):
                self.manager = manager
                super(ServerRequestHandler, self).__init__(*args, **kwargs)

            def handle(self):
                start_request = timeit.default_timer()
                log.info(timestamped_msg("Handling request..."))
                str_data = self.request.recv(4096)
                log.info("data={0}".format(str_data))
                json_data = json.loads(str_data)
                
                if json_data['type'] == 'translate':
                    self.manager.translation_request_queue.put(json_data)
                    response = {'msg': 'Translation request has been added to queue.'}
                elif json_data['type'] == 'debug':
                    log.info(list(self.manager.translation_request_queue.queue))
                    response = {
                        'translation_request_queue': list(self.manager.translation_request_queue.queue),
                        'clients': dict(self.manager.clients)
                    }
                else:
                    response = {'msg': 'Unknown request type. Request has been ignored.'}
                    pass
                log.info("q={0} sz={1}".format(self.manager.translation_request_queue, self.manager.translation_request_queue.qsize()))
                log.info("Request processed in {0} s. by {1}".format(
                    timeit.default_timer() - start_request, threading.current_thread().name))
                response = json.dumps(response)
                self.request.sendall(response)

        return ServerRequestHandler

    # def __init__(self, server_address, handler_class, translation_server_config):
    def __init__(self, server_address, translation_server_config):
        self.manager = Manager()
        handler_class = self.make_request_handler(self.manager)
        SocketServer.TCPServer.__init__(self, server_address, handler_class)

def timestamped_msg(msg):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    return "{0}: {1}".format(timestamp, msg)

 
def do_start_server(config_file):
    config = json.load(open(config_file))
    log.info("config={0}".format(config))
    log.info("host={0} port={1}".format(config['host'], config['port']))
    log.info("servers={0}".format(config['servers']))
    #server = Server( (config['host'], int(config['port'])), RequestHandler, config['servers'])
    server = Server( (config['host'], int(config['port'])), config['servers'])
    ip, port = server.server_address
    log.info(timestamped_msg("Start listening for requests on {0}:{1}...".format( socket.gethostname(), port)))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        server.server_close()

    sys.exit(0)
