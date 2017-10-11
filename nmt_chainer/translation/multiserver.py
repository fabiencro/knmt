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
from collections import deque

from urllib import urlencode
import requests
import Queue

from nmt_chainer.translation.client import Client

logging.basicConfig()
log = logging.getLogger("rnns:server")
log.setLevel(logging.INFO)

class RequestQueue(Queue.Queue):

    def __init__(self):
        Queue.Queue.__init__(self)

    def redistribute_requests(self):
        log.debug("redistribute_requests begin")
        self.mutex.acquire()
        temp = {}
        log.debug('begin')
        while self._qsize() > 0:
            item = self._get()
            #log.debug("item={0}".format(item))
            #log.debug("{0} {1}".format(item['session_id'], item['msg_id']))
            #log.debug("{0} {1}".format(item['session_id'], item['article_id']))
            key = "{0}-{1}".format(item['session_id'], item['client_tab_id'])
            log.debug("{0} {1}".format(key, item['sentence_number']))

            if not key in temp:
                temp[key] = deque()
            temp[key].append(item)
        log.debug('end')
        keys = temp.keys()[:]
        log.debug('keys={0}'.format(keys))
        
        log.debug('begin2')
        while keys:
            for k in keys:
                item = temp[k].popleft()
                key = "{0}-{1}".format(item['session_id'], item['client_tab_id'])
                log.debug("{0} {1}".format(key, item['sentence_number']))
                #log.debug("item={0}".format(item))
                self._put(item)
            keys = [k for k in keys if temp[k]]
        log.debug('end2')
        self.mutex.release()
        log.debug("redistribute_requests end")

    def cancel_requests_from(self, client_id):
        log.info("cancel_requests_from {0} begin".format(client_id)) 
        self.mutex.acquire()
        temp = deque()
        while self._qsize() > 0:
            item = self._get()
            item_key = "{0}-{1}".format(item['session_id'], item['client_tab_id'])
            if 'sentence_number' in item:
                log.info("{0} {1}".format(item_key, item['sentence_number']))
            elif 'type' in item:
                log.info("{0} {1}".format(item_key, item['type']))

            if item_key != client_id:
                temp.append(item)
        while len(temp) > 0:
            item = temp.pop()
            self._put(item)
        self.mutex.release()
        log.info("cancel_requests_from {0} end".format(client_id)) 

    def content(self):
        # Beware, if the base class changes, this will break.
        self.mutex.acquire()
        queue_copy = list(self.queue)
        self.mutex.release()
        return queue_copy

    def refresh_keepalive_timer(self, client_id, timestamp):
        self.mutex.acquire()
        for trans_req in self.queue:
            key = "{0}-{1}".format(trans_req['session_id'], trans_req['client_tab_id']) 
            if key == client_id:
                trans_req['keepalive'] = timestamp
        self.mutex.release()

    def remove_expired_requests(self):
        now = time.time()
        self.mutex.acquire()
        self.queue = deque([x for x in self.queue if now - x['keepalive'] < 10])
        self.mutex.release()

class QueueCleaner(threading.Thread):

    def __init__(self, queues, delay=10):
        threading.Thread.__init__(self)
        self.queues = queues
        self.delay = delay

    def run(self):
        while True:
            time.sleep(self.delay)
            for k in self.queues:
                self.queues[k].remove_expired_requests()

class Worker(threading.Thread):

    def __init__(self, name, src_lang, tgt_lang, host, port, manager):
        threading.Thread.__init__(self)
        self.name = name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.host = host
        self.port = port
        self.manager = manager
        self.current_client_key = None

    def stop(self):
        client = Client(self.host, self.port)
        resp = client.cancel()

    def run(self):
        lang_pair = "{0}-{1}".format(self.src_lang, self.tgt_lang)
        while True:
            translation_request = self.manager.translation_request_queues[lang_pair].get(True)
            log.info("Request for worker={0}".format(translation_request))
            self.current_client_key = "{0}-{1}".format(translation_request['session_id'], translation_request['client_tab_id']) 
            start_request = timeit.default_timer()
            log.info(timestamped_msg("Thread {0}: Handling request: {1}:{2}".format(self.name, self.current_client_key, translation_request['article_id'])))

            log.info("SENTENCE for worker {0}:{1}: {2}".format(self.host, self.port, translation_request['sentence'].encode('utf-8')))
            self.manager.add_active_translation(lang_pair, translation_request)
            client = Client(self.host, self.port)
            resp = client.query(translation_request['sentence'].encode('utf-8'), 
                article_id=translation_request['article_id'],
                beam_width=translation_request['beam_width'] if 'beam_width' in translation_request else 30,
                nb_steps=translation_request['nb_steps'] if 'nb_steps' in translation_request else 50,
                nb_steps_ratio=translation_request['nb_steps_ratio'] if 'nb_steps_ratio' in translation_request else 1.2,
                beam_pruning_margin=translation_request['beam_pruning_margin'] if 'beam_pruning_margin' in translation_request else 'none',
                beam_score_length_normalization=translation_request['beam_score_length_normalization'] if 'beam_score_length_normalization' in translation_request else 'none',
                beam_score_length_normalization_strength=translation_request['beam_score_length_normalization_strength'] if 'beam_score_length_normalization_strength' in translation_request else 0.2,
                post_score_length_normalization=translation_request['post_score_length_normalization'] if 'post_score_length_normalization' in translation_request else 'simple',
                post_score_length_normalization_strength=translation_request['post_score_length_normalization_strength'] if 'post_score_length_normalization_strength' in translation_request else 0.2,
                beam_score_coverage_penalty=translation_request['beam_score_coverage_penalty'] if 'beam_score_coverage_penalty' in translation_request else 'none',
                beam_score_coverage_penalty_strength=translation_request['beam_score_coverage_penalty_strength'] if 'beam_score_coverage_penalty_strength' in translation_request else 0.2,
                post_score_coverage_penalty=translation_request['post_score_coverage_penalty'] if 'post_score_coverage_penalty' in translation_request else 'none',
                post_score_coverage_penalty_strength=translation_request['post_score_coverage_penalty_strength'] if 'post_score_coverage_penalty_strength' in translation_request else 0.2,
                prob_space_combination=translation_request['prob_space_combination'] if 'prob_space_combination' in translation_request else 'false',
                normalize_unicode_unk=translation_request['normalize_unicode_unk'] if 'normalize_unicode_unk' in translation_request else 'true',
                remove_unk=translation_request['remove_unk'] if 'remove_unk' in translation_request else 'false',
                attempt_to_relocate_unk_source=translation_request['attempt_to_relocate_unk_source'] if 'attempt_to_relocate_unk_source' in translation_request else 'false',
                sentence_id=translation_request['sentence_number'],
                attn_graph_width=translation_request['attn_graph_width'] if 'attn_graph_width' in translation_request else 0,
                attn_graph_height=translation_request['attn_graph_height'] if 'attn_graph_height' in translation_request else 0)
            self.manager.remove_active_translation(lang_pair, translation_request)
            json_resp = json.loads(resp)
            log.info("TRANSLATION: {0}".format(json_resp['out'].encode('utf-8')))
            if not self.current_client_key in self.manager.client_cancellations or not self.manager.client_cancellations[self.current_client_key]:
                response_queue = self.manager.client_responses[self.current_client_key]
                response_queue.put(json_resp)

            log.info("Request processed in {0} s. by {1} [{2}:{3}]".format(timeit.default_timer() - start_request, self.name, self.current_client_key, translation_request['article_id']))

class Manager(object):

    def __init__(self, translation_server_config, text_to_sentences_splitter):
        self.text_to_sentences_splitter = text_to_sentences_splitter
        self.client_responses = {}
        self.client_cancellations = {}
        self.translation_request_queues = {}
        self.active_translations = {}
        self.workers = {}
        self.mutex = threading.Lock()
        for lang_pair in translation_server_config:
            src_lang, tgt_lang = lang_pair.split("-")
            self.translation_request_queues[lang_pair] = RequestQueue()
            self.active_translations[lang_pair] = []
            workerz = []
            self.workers[lang_pair] = workerz
            for idx, server in enumerate(translation_server_config[lang_pair]):
                worker = Worker("Translater-{0}({1}-{2})".format(idx, src_lang, tgt_lang), src_lang, tgt_lang, server['host'], server['port'], self)
                workerz.append(worker)
                worker.start()
        self.translation_request_queue_cleaner = QueueCleaner(self.translation_request_queues)
        self.translation_request_queue_cleaner.start()
        for k, q in list(self.translation_request_queues.items()):
            q.join()

    def add_active_translation(self, lang_pair, req):
        self.mutex.acquire()
        self.active_translations[lang_pair].append(req)
        self.mutex.release()

    def remove_active_translation(self, lang_pair, req):
        self.mutex.acquire()
        self.active_translations[lang_pair].remove(req)
        self.mutex.release()

    def get_stats(self, lang_pair):
        clients = set()
        all_requests = self.translation_request_queues[lang_pair].content() + self.active_translations[lang_pair]
        for item in all_requests:
            if 'session_id' in item and 'client_tab_id' in item:
                client_key = "{0}-{1}".format(item['session_id'], item['client_tab_id'])
                clients.add(client_key)
        return {'clients': len(clients), 'requests': len(all_requests)};

    def poll(self, lang_pair, client_id):
        self.translation_request_queues[lang_pair].refresh_keepalive_timer(client_id, time.time())

        resp = {};

        resp['workload'] = {
            'workers': len(self.workers[lang_pair])
        };
        resp['workload'].update(self.get_stats(lang_pair))
        req_per_worker = resp['workload']['requests'] / resp['workload']['workers']
        resp['workload']['factor'] = req_per_worker
        if resp['workload']['clients'] > resp['workload']['workers']:
            resp['workload']['factor'] += (resp['workload']['clients'] - resp['workload']['workers']) * req_per_worker
        log.info("req/work={0} r={1} c={2} w={3} f={4}".format(req_per_worker, resp['workload']['requests'], resp['workload']['clients'], resp['workload']['workers'], resp['workload']['factor']))

        responses = []
        if client_id in self.client_responses:
            while not self.client_responses[client_id].empty():
                responses.append(self.client_responses[client_id].get(False))
        resp['responses'] = responses;

        return resp

    def cancel(self, lang_pair, client_id):
        self.client_cancellations[client_id] = True
        self.translation_request_queues[lang_pair].cancel_requests_from(client_id)
        for worker in self.workers[lang_pair]:
            if worker.current_client_key == client_id:
                worker.stop()


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
                str_data = self.request.recv(16384)
                log.info("Request to server={0}".format(str_data))
                response = {}
                json_data = json.loads(str_data)
                lang_pair = "{0}-{1}".format(json_data['src_lang'], json_data['tgt_lang'])
                key = "{0}-{1}".format(json_data['session_id'], json_data['client_tab_id']) 
                if json_data['type'] == 'translate':
                    log.info("TRANSLATE from {0}: {1}".format(key, datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
                    data = {
                        'text': json_data['text'],
                        'lang_source': json_data['src_lang'],
                        'lang_target': json_data['tgt_lang']
                    }
                    r = requests.post(self.manager.text_to_sentences_splitter, data)
                    log.info('Splitter Status: {0}'.format(r.status_code))
                    if r.status_code == 200:
                        self.manager.client_cancellations[key] = False 

                        json_resp = r.json()

                        sentence_count = len(json_resp['sentences'])
                        if key in self.manager.client_responses:
                            response_queue = self.manager.client_responses[key]
                        else:
                            response_queue = Queue.Queue()
                            self.manager.client_responses[key] = response_queue
                        response_queue.put(json_resp)

                        log.info("sentenceCount={0}".format(len(json_resp['sentences'])))
                        for index, sentence in enumerate(json_resp['sentences']):
                            translate_sentence_request = dict(json_data)
                            del translate_sentence_request['text']
                            translate_sentence_request['sentence'] = sentence
                            translate_sentence_request['sentence_number'] = index
                            translate_sentence_request['lang_source'] = translate_sentence_request['src_lang']
                            translate_sentence_request['lang_target'] = translate_sentence_request['tgt_lang']
                            translate_sentence_request['keepalive'] = time.time()
                            self.manager.translation_request_queues[lang_pair].put(translate_sentence_request)
                        self.manager.translation_request_queues[lang_pair].redistribute_requests()
                        response = {'msg': 'All sentences have been queued.'}
                elif json_data['type'] == 'poll':
                    log.info("POLL from {0}".format(key))   
                    response = self.manager.poll(lang_pair, key)
                elif json_data['type'] == 'cancelTranslation':
                    log.info("CANCEL from {0}".format(key))
                    self.manager.cancel(lang_pair, key)
                    response = {'msg': 'Translation request has been cancelled.'}
                else:
                    response = {'msg': 'Unknown request type. Request has been ignored.'}

                log.info("Request processed in {0} s. by {1}".format(timeit.default_timer() - start_request, threading.current_thread().name))
                response = json.dumps(response)
                log.info("Response from server={0}".format(response))
                self.request.sendall(response)

        return ServerRequestHandler

    def __init__(self, server_address, translation_server_config, text_to_sentences_splitter):
        self.manager = Manager(translation_server_config, text_to_sentences_splitter)
        handler_class = self.make_request_handler(self.manager)
        SocketServer.TCPServer.__init__(self, server_address, handler_class)

def timestamped_msg(msg):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    return "{0}: {1}".format(timestamp, msg)

 
def do_start_server(config_file):
    config = json.load(open(config_file))
    server = Server((config['host'], int(config['port'])), config['servers'], config['text_to_sentences_splitter'])
    ip, port = server.server_address
    log.info(timestamped_msg("Start listening for requests on {0}:{1}...".format( socket.gethostname(), port)))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        server.server_close()

    sys.exit(0)
