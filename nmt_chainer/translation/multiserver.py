#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""multiserver.py: Translation server that works asynchronously and dispatches requests to multiple translation servers."""
__author__ = "Frederic Bergeron"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "bergeron@pa.jst.jp"
__status__ = "Development"

import json
import logging
import logging.config
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

log = None

logging.basicConfig()

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
        log.debug("cancel_requests_from {0} begin".format(client_id)) 
        self.mutex.acquire()
        temp = deque()
        while self._qsize() > 0:
            item = self._get()
            item_key = "{0}-{1}".format(item['session_id'], item['client_tab_id'])
            if 'sentence_number' in item:
                log.debug("{0} {1}".format(item_key, item['sentence_number']))
            elif 'type' in item:
                log.debug("{0} {1}".format(item_key, item['type']))

            if item_key != client_id:
                temp.append(item)
        while len(temp) > 0:
            item = temp.pop()
            self._put(item)
        self.mutex.release()
        log.debug("cancel_requests_from {0} end".format(client_id)) 

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

    def __init__(self, name, category, host, port, manager):
        threading.Thread.__init__(self)
        self.name = name
        self.category = category
        self.lang_pair, self.model = category.split("_")
        self.src_lang, self.tgt_lang = self.lang_pair.split("-")
        self.host = host
        self.port = port
        self.manager = manager
        self.current_client_key = None

    def stop(self):
        client = Client(self.host, self.port)
        try:
            resp = client.cancel()
        except BaseException as err:
            log.info("An error has occurred when the client named '{0}' performed a CANCEL query for the '{1}' category: '{2}'".format(self.name, self.category, err))

    def run(self):
        while True:
            translation_request = self.manager.translation_request_queues[self.category].get(True)
            log.debug("Request for worker={0}".format(translation_request))
            self.current_client_key = "{0}-{1}".format(translation_request['session_id'], translation_request['client_tab_id']) 
            start_request = timeit.default_timer()
            log.debug("Thread {0}: Handling request: {1}:{2}".format(self.name, self.current_client_key, translation_request['article_id']))

            log.debug("SENTENCE for worker {0}:{1}: {2}".format(self.host, self.port, translation_request['sentence'].encode('utf-8')))
            self.manager.add_active_translation(self.category, translation_request)
            client = Client(self.host, self.port)
            try:
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
                    force_finish=translation_request['force_finish'] if 'force_finish' in translation_request else 'false',
                    sentence_id=translation_request['sentence_number'],
                    attn_graph_width=translation_request['attn_graph_width'] if 'attn_graph_width' in translation_request else 0,
                    attn_graph_height=translation_request['attn_graph_height'] if 'attn_graph_height' in translation_request else 0)
                json_resp = json.loads(resp)
                if 'error' in json_resp:
                    log.info("An error has occurred: {0}\n".format(json_resp['error']))
                    if 'stacktrace' in json_resp:
                        for item in json_resp['stacktrace']:
                            log.info(item)
                else:
                    log.debug("TRANSLATION: {0}".format(json_resp['out'].encode('utf-8')))
                    if not self.current_client_key in self.manager.client_cancellations or not self.manager.client_cancellations[self.current_client_key]:
                        response_queue = self.manager.client_responses[self.current_client_key]
                        response_queue.put(json_resp)
            except BaseException as err:
                log.info("An error has occurred when the client named '{0}' performed a TRANSLATE query for the '{1}' category: '{2}'".format(self.name, self.category, err))
            finally:
                self.manager.remove_active_translation(self.category, translation_request)

            log.debug("Request processed in {0} s. by {1} [{2}:{3}]".format(timeit.default_timer() - start_request, self.name, self.current_client_key, translation_request['article_id']))

class Manager(object):

    def __init__(self, translation_server_config, text_to_sentences_splitter):
        self.translation_server_config = translation_server_config
        self.text_to_sentences_splitter = text_to_sentences_splitter
        self.client_responses = {}
        self.client_cancellations = {}
        self.translation_request_queues = {}
        self.active_translations = {}
        self.workers = {}
        self.mutex = threading.Lock()
        for server_category in translation_server_config:
            self.translation_request_queues[server_category] = RequestQueue()
            self.active_translations[server_category] = []
            workerz = []
            self.workers[server_category] = workerz
            for idx, server in enumerate(translation_server_config[server_category]):
                worker = Worker("Translater-{0}({1})".format(idx, server), server_category, server['host'], server['port'], self)
                workerz.append(worker)
                worker.start()
        self.translation_request_queue_cleaner = QueueCleaner(self.translation_request_queues)
        self.translation_request_queue_cleaner.start()
        for k, q in list(self.translation_request_queues.items()):
            q.join()

    def add_active_translation(self, category, req):
        self.mutex.acquire()
        self.active_translations[category].append(req)
        self.mutex.release()

    def remove_active_translation(self, category, req):
        self.mutex.acquire()
        self.active_translations[category].remove(req)
        self.mutex.release()

    def get_stats(self, category):
        clients = set()
        all_requests = self.translation_request_queues[category].content() + self.active_translations[category]
        for item in all_requests:
            if 'session_id' in item and 'client_tab_id' in item:
                client_key = "{0}-{1}".format(item['session_id'], item['client_tab_id'])
                clients.add(client_key)
        return {'clients': len(clients), 'requests': len(all_requests)};

    def poll(self, category, client_id):
        if category and category in self.translation_request_queues:
            self.translation_request_queues[category].refresh_keepalive_timer(client_id, time.time())

        resp = {};
        if category is None:
            resp['servers'] = self.translation_server_config.keys()
        elif category in self.workers:
            resp['workload'] = {
                'workers': len(self.workers[category])
            };
            resp['workload'].update(self.get_stats(category))
            req_per_worker = resp['workload']['requests'] / resp['workload']['workers']
            resp['workload']['factor'] = req_per_worker
            if resp['workload']['clients'] > resp['workload']['workers']:
                resp['workload']['factor'] += (resp['workload']['clients'] - resp['workload']['workers']) * req_per_worker
            log.debug("req/work={0} r={1} c={2} w={3} f={4}".format(req_per_worker, resp['workload']['requests'], resp['workload']['clients'], resp['workload']['workers'], resp['workload']['factor']))

        responses = []
        if client_id in self.client_responses:
            while not self.client_responses[client_id].empty():
                responses.append(self.client_responses[client_id].get(False))
        resp['responses'] = responses;

        return resp

    def cancel(self, category, client_id):
        self.client_cancellations[client_id] = True
        self.translation_request_queues[category].cancel_requests_from(client_id)
        for worker in self.workers[category]:
            if worker.current_client_key == client_id:
                worker.stop()

EOM = "==== EOM ===="

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
                log.debug("Handling request...")

                # Read until EOM delimiter is met. 
                total_data = []
                data = ''
                while True:
                    data = self.request.recv(4096)
                    if EOM in data:
                        total_data.append(data[:data.find(EOM)])
                        break
                    total_data.append(data)
                    if len(total_data)>1:
                        #check if EOM was split
                        last_pair = total_data[-2] + total_data[-1]
                        if EOM in last_pair:
                            total_data[-2] = last_pair[:last_pair.find(EOM)]
                            total_data.pop()
                            break
                str_data = ''.join(total_data)

                log.debug("Request to server={0}".format(str_data))
                response = {}
                json_data = None
                try:
                    json_data = json.loads(str_data)
                except Exception as e:
                    log.info("Invalid JSON data. Request ignored.")

                if json_data:    
                    lang_pair = None
                    if json_data['src_lang'] and json_data['tgt_lang']:
                        lang_pair = "{0}-{1}".format(json_data['src_lang'], json_data['tgt_lang'])
                    model = None
                    if json_data['model']:
                        model = json_data['model']
                    category = None
                    if lang_pair and model:
                        category = "{0}_{1}".format(lang_pair, model)
                    key = "{0}-{1}".format(json_data['session_id'], json_data['client_tab_id']) 
                    log.debug("lang_pair={0} model={1} category={2}".format(lang_pair, model, category))
                    if json_data['type'] == 'translate':
                        log.info("TRANSLATE from {0}".format(key))
                        data = {
                            'text': json_data['text'],
                            'lang_source': json_data['src_lang'],
                            'lang_target': json_data['tgt_lang']
                        }
                        r = requests.post(self.manager.text_to_sentences_splitter, data)
                        log.debug('Splitter Status: {0}'.format(r.status_code))
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

                            log.debug("sentenceCount={0}".format(len(json_resp['sentences'])))
                            for index, sentence in enumerate(json_resp['sentences']):
                                translate_sentence_request = dict(json_data)
                                del translate_sentence_request['text']
                                translate_sentence_request['sentence'] = sentence
                                translate_sentence_request['sentence_number'] = index
                                translate_sentence_request['lang_source'] = translate_sentence_request['src_lang']
                                translate_sentence_request['lang_target'] = translate_sentence_request['tgt_lang']
                                translate_sentence_request['keepalive'] = time.time()
                                self.manager.translation_request_queues[category].put(translate_sentence_request)
                            self.manager.translation_request_queues[category].redistribute_requests()
                            response = {'msg': 'All sentences have been queued.'}
                    elif json_data['type'] == 'poll':
                        log.debug("POLL from {0}".format(key))   
                        response = self.manager.poll(category, key)
                    elif json_data['type'] == 'cancelTranslation':
                        log.info("CANCEL from {0}".format(key))
                        self.manager.cancel(category, key)
                        response = {'msg': 'Translation request has been cancelled.'}
                    else:
                        response = {'msg': 'Unknown request type. Request has been ignored.'}

                    log.debug("Request processed in {0} s. by {1}".format(timeit.default_timer() - start_request, threading.current_thread().name))
                    response = json.dumps(response)
                    log.debug("Response from server={0}".format(response))
                    self.request.sendall(response)

        return ServerRequestHandler

    def __init__(self, server_address, translation_server_config, text_to_sentences_splitter):
        self.manager = Manager(translation_server_config, text_to_sentences_splitter)
        handler_class = self.make_request_handler(self.manager)
        SocketServer.TCPServer.__init__(self, server_address, handler_class)

def do_start_server(config_file, log_config):
    if log_config:
        logging.config.fileConfig(log_config)
    global log
    log = logging.getLogger("default")
    log.setLevel(logging.INFO)

    config = json.load(open(config_file))
    server = Server((config['host'], int(config['port'])), config['servers'], config['text_to_sentences_splitter'])
    ip, port = server.server_address
    log.info("Start listening for requests on {0}:{1}...".format( socket.gethostname(), port))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        server.server_close()

    sys.exit(0)
