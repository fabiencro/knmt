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
                    response_queue = self.manager.client_responses[key]
                    response_queue.put(json_resp)
                else:
                    log.info("RESPONSE: {0}".format(json_resp))

            log.info("Request processed in {0} s. by {1} [{2}:{3}]".format(timeit.default_timer() - start_request, self.name, key, translation_request['article_id']))

class Manager(object):

    def __init__(self, translation_server_config):
        self.client_responses = {}
        self.translation_request_queues = {}
        self.workers = {}
        for key in translation_server_config:
            src_lang, tgt_lang = key.split("-")
            #self.translation_request_queues[key] = Queue.Queue()
            self.translation_request_queues[key] = RequestQueue()
            workerz = []
            self.workers[key] = workerz
            for idx, server in enumerate(translation_server_config[key]):
                worker = Worker("Translater-{0}({1}-{2})".format(idx, src_lang, tgt_lang), src_lang, tgt_lang, server['host'], server['port'], self)
                workerz.append(worker)
                worker.start()
        log.info("qs={0}".format(self.translation_request_queues))
        for k, q in list(self.translation_request_queues.items()):
            q.join()

    def poll(self, key):
        resp = []
        if key in self.client_responses:
            while not self.client_responses[key].empty():
                resp.append(self.client_responses[key].get(False))
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
                    response = self.manager.poll(key)
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


class ProducerWorker(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global z
        log.info("Producer started.")
        while True:
            for i in range(4):
                q.put(({'session_id': z, 'msg_id': "{0}_{1}".format(z, i)}))
            z += 1
            q.redistribute_requests()
            sleep(5)

class ConsumerWorker(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        log.info("Consumer started.")
        while True:
            priority, item = q.get(True)
            log.info("Processing item={0}".format(item['msg_id']))
            sleep(4)

if __name__ == "__main__":
    z = 1
    q = RequestQueue()

    #p = ProducerWorker()
    c = ConsumerWorker()
    # p.start()
    # c.start()
    # p.join()
    # c.join()

    q.put((SENTENCE_REQ_PRIORITY, {'session_id': 1, 'msg_id': '1_1'}))
    q.put((SENTENCE_REQ_PRIORITY, {'session_id': 1, 'msg_id': '1_2'}))
    q.put((SENTENCE_REQ_PRIORITY, {'session_id': 1, 'msg_id': '1_3'}))
    q.put((SENTENCE_REQ_PRIORITY, {'session_id': 1, 'msg_id': '1_4'}))
    q.put((SENTENCE_REQ_PRIORITY, {'session_id': 2, 'msg_id': '2_1'}))
    q.put((SENTENCE_REQ_PRIORITY, {'session_id': 2, 'msg_id': '2_2'}))
    q.put((SENTENCE_REQ_PRIORITY, {'session_id': 2, 'msg_id': '2_3'}))
    q.put((SENTENCE_REQ_PRIORITY, {'session_id': 2, 'msg_id': '2_4'}))
    q.redistribute_requests()
    #p.start()
    c.start()
    #p.join()
    c.join()

    log.info(q)
    # q.redistribute_requests()

# if __name__ == "__main__":
#     # q = Queue.PriorityQueue()
#     # q.put((TEXT_REQ_PRIORITY, {'type': 'text', 'id': 't1'}))
#     # q.put((SENTENCE_REQ_PRIORITY, {'type': 'sentence', 'id': 's1_1'}))
#     # q.put((SENTENCE_REQ_PRIORITY, {'type': 'sentence', 'id': 's1_2'}))
#     # q.put((SENTENCE_REQ_PRIORITY, {'type': 'sentence', 'id': 's1_3'}))
#     # q.put((SENTENCE_REQ_PRIORITY, {'type': 'sentence', 'id': 's1_4'}))
# 
#     # q.put((TEXT_REQ_PRIORITY, {'type': 'text', 'id': 't2'}))
#     # q.put((SENTENCE_REQ_PRIORITY, {'type': 'sentence', 'id': 's2_1'}))
#     # q.put((SENTENCE_REQ_PRIORITY, {'type': 'sentence', 'id': 's2_2'}))
#     # q.put((SENTENCE_REQ_PRIORITY, {'type': 'sentence', 'id': 's2_3'}))
#     # q.put((SENTENCE_REQ_PRIORITY, {'type': 'sentence', 'id': 's2_4'}))
# 
#     # q.put((TEXT_REQ_PRIORITY, {'type': 'text', 'id': 't3'}))
#     # q.put((SENTENCE_REQ_PRIORITY, {'type': 'sentence', 'id': 's3_1'}))
#     # q.put((SENTENCE_REQ_PRIORITY, {'type': 'sentence', 'id': 's3_2'}))
#     # q.put((SENTENCE_REQ_PRIORITY, {'type': 'sentence', 'id': 's3_3'}))
#     # q.put((SENTENCE_REQ_PRIORITY, {'type': 'sentence', 'id': 's3_4'}))
# 
#     # while not q.empty():
#     #     item = q.get()
#     #     print item
# 
# 
#     # q = Queue.PriorityQueue()
#     # # q.put(10)
#     # # q.put(1)
#     # # q.put(5)
#     # # q.put('dix', 10)
#     # # q.put('un', 1)
#     # # q.put('cinq', 5)
#     # # q.put(('dix', 10))
#     # # q.put(('un', 1))
#     # # q.put(('cinq', 5))
#     # q.put((10, 'dix'))
#     # q.put((1, 'un'))
#     # q.put((5, 'cinq'))
#     # while not q.empty():
#     #     print q.get(),
