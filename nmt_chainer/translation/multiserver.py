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

TEXT_REQ_PRIORITY = 100
SENTENCE_REQ_PRIORITY = 2000

class RequestQueue(Queue.PriorityQueue):

    def __init__(self):
        Queue.Queue.__init__(self)

    def get_stats(self):
        self.mutex.acquire()
        clients = set()
        for item in self.queue:
            priority, actual_item = item
            if 'session_id' in actual_item and 'client_tab_id' in actual_item:
                client_key = "{0}-{1}".format(actual_item['session_id'], actual_item['client_tab_id'])
                clients.add(client_key)
        self.mutex.release()
        return {'clients': len(clients), 'requests': self._qsize()};

    def redistribute_requests(self):
        log.info("redistribute_requests begin")
        self.mutex.acquire()
        temp = {}
        log.info('begin')
        while self._qsize() > 0:
            priority, item = self._get()
            #log.info("priority={0} item={1}".format(priority, item))
            #log.info("{0} {1}".format(item['session_id'], item['msg_id']))
            #log.info("{0} {1}".format(item['session_id'], item['article_id']))
            key = "{0}-{1}".format(item['session_id'], item['client_tab_id'])
            if 'sentence_number' in item:
                log.info("{0} {1} {2}".format(priority, key, item['sentence_number']))
            elif 'type' in item:
                log.info("{0} {1} {2}".format(priority, key, item['type']))

            if not key in temp:
                temp[key] = deque()
            temp[key].append((priority, item))
        log.info('end')
        keys = temp.keys()[:]
        log.info('keys={0}'.format(keys))
        
        log.info('begin2')
        r = 0
        while keys:
            for k in keys:
                priority, item = temp[k].popleft()
                key = "{0}-{1}".format(item['session_id'], item['client_tab_id'])
                if 'sentence_number' in item:
                    new_priority = SENTENCE_REQ_PRIORITY + r
                    log.info("{0} {1} {2}".format(new_priority, key, item['sentence_number']))
                elif 'type' in item:
                    new_priority = TEXT_REQ_PRIORITY + r
                    log.info("{0} {1} {2}".format(new_priority, key, item['type']))
                r += 1
                #log.info("priority={0} item={1}".format(new_priority, item))
                self._put((new_priority, item))
            keys = [k for k in keys if temp[k]]
        log.info('end2')
        self.mutex.release()
        log.info("redistribute_requests end")

    # A faster approach would be to tag the requests as stale (given that we have a reference on them)
    # instead of removing them from the queue.
    def cancel_requests_from(self, key_to_remove):
        log.info("cancel_requests_from {0} begin".format(key_to_remove)) 
        self.mutex.acquire()
        temp = deque()
        while self._qsize() > 0:
            priority, item = self._get()
            item_key = "{0}-{1}".format(item['session_id'], item['client_tab_id'])
            if 'sentence_number' in item:
                log.info("{0} {1} {2}".format(priority, item_key, item['sentence_number']))
            elif 'type' in item:
                log.info("{0} {1} {2}".format(priority, item_key, item['type']))

            if item_key != key_to_remove:
                temp.append((priority, item))
        log.info('begin2')
        while len(temp) > 0:
            priority, item = temp.pop()
            self._put((priority, item))
        log.info('end2')
        self.mutex.release()
        log.info("cancel_requests_from {0} end".format(key_to_remove)) 


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
        lang_pair = "{0}-{1}".format(self.src_lang, self.tgt_lang)
        while True:
            request_priority, translation_request = self.manager.translation_request_queues[lang_pair].get(True)
            log.info("request={0}".format(translation_request))
            key = "{0}-{1}".format(translation_request['session_id'], translation_request['client_tab_id']) 
            start_request = timeit.default_timer()
            log.info(timestamped_msg("Thread {0}: Handling request: {1}:{2}".format(self.name, key, translation_request['article_id'])))

            if 'text' in translation_request:
                log.info("TEXT: {0} {1}".format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') ,translation_request['text'].encode('utf-8')))
                data = {
                    'text': translation_request['text'],
                    'lang_source': translation_request['src_lang'],
                    'lang_target': translation_request['tgt_lang']
                }
                splitter_url = 'http://lotus.kuee.kyoto-u.ac.jp/~frederic/webmt-rnnsearch-multiple-requests/cgi-bin/split_sentences.cgi'
                r = requests.post(splitter_url, data)
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
                        translate_sentence_request = dict(translation_request)
                        del translate_sentence_request['text']
                        translate_sentence_request['sentence'] = sentence
                        translate_sentence_request['sentence_number'] = index
                        translate_sentence_request['lang_source'] = self.src_lang
                        translate_sentence_request['lang_target'] = self.tgt_lang
                        log.info("new_req={0}".format(translate_sentence_request))
                        self.manager.translation_request_queues[lang_pair].put((SENTENCE_REQ_PRIORITY, translate_sentence_request))
                    log.info("q before sz={0}".format(self.manager.translation_request_queues[lang_pair].qsize()))
                    self.manager.translation_request_queues[lang_pair].redistribute_requests()
                    log.info("q after sz={0}".format(self.manager.translation_request_queues[lang_pair].qsize()))
            elif 'sentence' in translation_request:
                log.info("SENTENCE for worker {0}:{1}: {2}".format(self.host, self.port, translation_request['sentence'].encode('utf-8')))
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
                    if not key in self.manager.client_cancellations or not self.manager.client_cancellations[key]:
                        response_queue = self.manager.client_responses[key]
                        response_queue.put(json_resp)
                else:
                    log.info("RESPONSE: {0}".format(json_resp))

            log.info("Request processed in {0} s. by {1} [{2}:{3}]".format(timeit.default_timer() - start_request, self.name, key, translation_request['article_id']))

class Manager(object):

    def __init__(self, translation_server_config):
        self.client_responses = {}
        self.client_cancellations = {}
        self.translation_request_queues = {}
        self.workers = {}
        for lang_pair in translation_server_config:
            src_lang, tgt_lang = lang_pair.split("-")
            #self.translation_request_queues[lang_pair] = Queue.Queue()
            self.translation_request_queues[lang_pair] = RequestQueue()
            workerz = []
            self.workers[lang_pair] = workerz
            for idx, server in enumerate(translation_server_config[lang_pair]):
                worker = Worker("Translater-{0}({1}-{2})".format(idx, src_lang, tgt_lang), src_lang, tgt_lang, server['host'], server['port'], self)
                workerz.append(worker)
                worker.start()
        log.info("qs={0}".format(self.translation_request_queues))
        for k, q in list(self.translation_request_queues.items()):
            q.join()

    def poll(self, lang_pair, key):
        resp = {};

        resp['workload'] = {
            'workers': len(self.workers[lang_pair])
        };
        resp['workload'].update(self.translation_request_queues[lang_pair].get_stats())
        req_per_worker = resp['workload']['requests'] / resp['workload']['workers']
        resp['workload']['factor'] = req_per_worker
        if resp['workload']['clients'] > resp['workload']['workers']:
            resp['workload']['factor'] += (resp['workload']['clients'] - resp['workload']['workers']) * req_per_worker
        log.info("req/work={0} r={1} c={2} w={3} f={4}".format(req_per_worker, resp['workload']['requests'], resp['workload']['clients'], resp['workload']['workers'], resp['workload']['factor']))

        responses = []
        if key in self.client_responses:
            while not self.client_responses[key].empty():
                responses.append(self.client_responses[key].get(False))
        resp['responses'] = responses;

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
                
                response = {}
                lang_pair = "{0}-{1}".format(json_data['src_lang'], json_data['tgt_lang'])
                key = "{0}-{1}".format(json_data['session_id'], json_data['client_tab_id']) 
                if json_data['type'] == 'translate':
                    log.info("TRANSLATE!!!! {0} from {1}".format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), key))
                    self.manager.translation_request_queues[lang_pair].put((TEXT_REQ_PRIORITY, json_data))
                    # self.manager.translation_request_queues[lang_pair].redistribute_requests()
                    response = {'msg': 'Translation request has been added to queue.'}
                elif json_data['type'] == 'poll':
                    log.info("POLL from {0}".format(key))   
                    response = self.manager.poll(lang_pair, key)
                elif json_data['type'] == 'cancelTranslation':
                    log.info("CANCEL from {0}".format(key))
                    self.manager.client_cancellations[key] = True
                    self.manager.translation_request_queues[lang_pair].cancel_requests_from(key)
                    response = {'msg': 'Translation request has been cancelled.'}
                elif json_data['type'] == 'debug':
                    log.info('debug!')
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

                log.info("q={0} sz={1}".format(self.manager.translation_request_queues[lang_pair], self.manager.translation_request_queues[lang_pair].qsize()))
                log.info("Request processed in {0} s. by {1}".format(timeit.default_timer() - start_request, threading.current_thread().name))
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
    server = Server((config['host'], int(config['port'])), config['servers'])
    ip, port = server.server_address
    log.info(timestamped_msg("Start listening for requests on {0}:{1}...".format( socket.gethostname(), port)))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        server.server_close()

    sys.exit(0)
