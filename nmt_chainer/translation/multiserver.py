#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""multiserver.py: Translation server that works asynchronously and dispatches requests to multiple translation servers."""
__author__ = "Frederic Bergeron"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "bergeron@pa.jst.jp"
__status__ = "Development"

import datetime
import json
import logging
import sys
 
import time
from time import sleep
import timeit
import socket
import threading
import SocketServer

from urllib import urlencode
import requests
import Queue

from nmt_chainer.translation.client import Client

logging.basicConfig()
log = logging.getLogger("rnns:server")
log.setLevel(logging.INFO)

class Worker(threading.Thread):

    def __init__(self, name, src_lang, tgt_lang, host, port, manager):
        threading.Thread.__init__(self)
        self.name = name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.host = host
        self.port = port
        self.manager = manager

    def run(self):
        while True:
            key = "{0}-{1}".format(self.src_lang, self.tgt_lang)
            translation_request = self.manager.translation_request_queues[key].get(True)
            start_request = timeit.default_timer()
            log.info(timestamped_msg("Thead {0}: Handling request: {1}:{2}".format(self.name, translation_request['session_id'], translation_request['article_id'])))

            if 'text' in translation_request:
                log.info("TEXT: {0} {1}".format(type(translation_request['text']), translation_request['text'].encode('utf-8')))
                data = {
                    'text': translation_request['text'],
                    'lang_source': translation_request['src_lang'],
                    'lang_target': translation_request['tgt_lang']
                }
                splitter_url = 'http://lotus.kuee.kyoto-u.ac.jp/~frederic/webmt-rnnsearch-multiple-requests/cgi-bin/split_sentences.cgi'
                r = requests.post(splitter_url, data)
                if r.status_code == 200:
                    json_resp = r.json()
                    for index, sentence in enumerate(json_resp['sentences']):
                        translate_sentence_request = dict(translation_request)
                        del translate_sentence_request['text']
                        translate_sentence_request['sentence'] = sentence
                        translate_sentence_request['sentence_number'] = index
                        translate_sentence_request['lang_source'] = self.src_lang
                        translate_sentence_request['lang_target'] = self.tgt_lang
                        log.info("new_req={0}".format(translate_sentence_request))
                        self.manager.translation_request_queues[key].put(translate_sentence_request)
            elif 'sentence' in translation_request:
                log.info("SENTENCE for worker {0}:{1}: {2} {3}".format(self.host, self.port, type(translation_request['sentence']), translation_request['sentence'].encode('utf-8')))
                client = Client(self.host, self.port)
                resp = client.query(translation_request['sentence'].encode('utf-8'), 
                    article_id=translation_request['article_id'],
                    beam_width=translation_request['beam_width'],
                    nb_steps=translation_request['nb_steps'],
                    nb_steps_ratio=translation_request['nb_steps_ratio'],
                    prob_space_combination=translation_request['prob_space_combination'],
                    normalize_unicode_unk=translation_request['normalize_unicode_unk'],
                    remove_unk=translation_request['remove_unk'],
                    attempt_to_relocate_unk_source=translation_request['attempt_to_relocate_unk_source'],
                    sentence_id=translation_request['sentence_number'],
                    attn_graph_width=translation_request['attn_graph_width'],
                    attn_graph_height=translation_request['attn_graph_height'])
                json_resp = json.loads(resp)
                if 'out' in json_resp:
                    log.info("TRANSLATION: {0}".format(json_resp['out'].encode('utf-8')))
                    if  translation_request['session_id'] in self.manager.client_responses:
                        response_queue = self.manager.client_responses[translation_request['session_id']]
                    else:
                        response_queue = Queue.Queue()
                        self.manager.client_responses[translation_request['session_id']] = response_queue
                    response_queue.put(json_resp)
                else:
                    log.info("RESPONSE: {0}".format(json_resp))

            log.info("Request processed in {0} s. by {1} [{2}:{3}]".format(timeit.default_timer() - start_request, self.name, translation_request['session_id'], translation_request['article_id']))

class Manager(object):

    def __init__(self, translation_server_config):
        self.client_responses = {}
        self.translation_request_queues = {}
        self.workers = {}
        for key in translation_server_config:
            src_lang, tgt_lang = key.split("-")
            self.translation_request_queues[key] = Queue.Queue()
            workerz = []
            self.workers[key] = workerz
            for idx, server in enumerate(translation_server_config[key]):
                worker = Worker("Translater-{0}({1}-{2})".format(idx, src_lang, tgt_lang), src_lang, tgt_lang, server['host'], server['port'], self)
                workerz.append(worker)
                worker.start()
        log.info("qs={0}".format(self.translation_request_queues))
        for k, q in list(self.translation_request_queues.items()):
            q.join()

    def poll(self, session_id):
        resp = []
        if session_id in self.client_responses:
            while not self.client_responses[session_id].empty():
                resp.append(self.client_responses[session_id].get(False))
        return resp

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
                
                key = "{0}-{1}".format(json_data['src_lang'], json_data['tgt_lang'])
                if json_data['type'] == 'translate':
                    self.manager.translation_request_queues[key].put(json_data)
                    response = {'msg': 'Translation request has been added to queue.'}
                elif json_data['type'] == 'poll':
                    response = self.manager.poll(json_data['session_id'])
                elif json_data['type'] == 'debug':
                    log.info(self.manager.translation_request_queues)
                    trans_queues = {}
                    for k, q in list(self.manager.translation_request_queues.items()):
                        log.info("k={0} q={1}".format(k, q))
                        trans_queues[k] = list(q.queue)
                    response = {
                        'translation_request_queue': trans_queues,
                        'clients': dict(self.manager.client_responses)
                    }
                else:
                    response = {'msg': 'Unknown request type. Request has been ignored.'}
                    pass
                log.info("q={0} sz={1}".format(self.manager.translation_request_queues[key], self.manager.translation_request_queues[key].qsize()))
                log.info("Request processed in {0} s. by {1}".format(
                    timeit.default_timer() - start_request, threading.current_thread().name))
                response = json.dumps(response)
                self.request.sendall(response)

        return ServerRequestHandler

    def __init__(self, server_address, translation_server_config):
        self.manager = Manager(translation_server_config)
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
    server = Server((config['host'], int(config['port'])), config['servers'])
    ip, port = server.server_address
    log.info(timestamped_msg("Start listening for requests on {0}:{1}...".format( socket.gethostname(), port)))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        server.server_close()

    sys.exit(0)
