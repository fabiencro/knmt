#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""mail_handler.py: Daemon that monitors an IMAP account for messages containing translation requests."""
__author__ = "Frederic Bergeron"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "bergeron@pa.jst.jp"
__status__ = "Development"

import argparse
import base64
import codecs
import daemon
import email
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
import gettext
import gzip
import httplib
import imaplib
import io
import json
import locale
import logging
import logging.config
import os.path
import re
import smtplib
import subprocess
import tempfile
import time
import urllib

from nmt_chainer._version import __version__ as knmt_version
from nmt_chainer.translation.client import Client

list_resp_pattern = re.compile('(.*?) "(.*)" (.*)')
subject_pattern = re.compile('([a-zA-Z][a-zA-Z])_([a-zA-Z][a-zA-Z])')


def split_text_into_paragraphes(text, src_lang, tgt_lang):
    if src_lang in ['ja', 'zh'] and '\u3000' in text:
        paragraphes = text.replace('\n', '').split('\u3000')
        paragraphes = filter(lambda p: len(p) > 0, paragraphes)
        return paragraphes

    paragraphes = text.split("\n")
    paragraphes = filter(lambda p: len(p) > 0, paragraphes)
    return paragraphes


# Taken from https://gist.githubusercontent.com/miohtama/5389146/raw/03362b9aa72f9d3a4e68fb99af40a0ef160e0ec5/gistfile1.py
def get_decoded_email_body(message_body):
    """ Decode email body.
    Detect character set if the header is not set.
    We try to get text/plain, but if there is not one then fallback to text/html.
    :param message_body: Raw 7-bit message body input e.g. from imaplib. Double encoded in quoted-printable and latin-1
    :return: Message body as unicode string
    """

    msg = email.message_from_string(message_body)

    text = ""
    if msg.is_multipart():
        html = None
        for part in msg.get_payload():

            print "%s, %s" % (part.get_content_type(), part.get_content_charset())

            if part.get_content_type() not in ['text/plain', 'text/html']:
                continue

            if part.get_content_charset() is None:
                # We cannot know the character set, so return decoded "something"
                text = part.get_payload(decode=True)
                continue

            charset = part.get_content_charset()
            if part.get_content_type() == 'text/plain':
                text = unicode(part.get_payload(decode=True), str(charset), "ignore").encode('utf8', 'replace')

            if part.get_content_type() == 'text/html':
                html = unicode(part.get_payload(decode=True), str(charset), "ignore").encode('utf8', 'replace')

        if text is not None:
            return text.strip()
        else:
            return html.strip()
    else:
        text = unicode(msg.get_payload(decode=True), msg.get_content_charset(), 'ignore').encode('utf8', 'replace')
        return text.strip()


class MailHandler:

    def __init__(self, config, log_config):
        self.config_file = config
        self.log_config_file = log_config
        self._requests = []

    def _send_mail(self, to, subject, text):
        config = json.load(open(self.config_file))
        resp_msg = MIMEMultipart()
        resp_msg['From'] = config['smtp']['from']
        resp_msg['To'] = to
        cc = None
        bcc = None
        if 'cc' in config['smtp']:
            resp_msg['Cc'] = config['smtp']['cc']
            cc = config['smtp']['cc'].split(",")
        if 'bcc' in config['smtp']:
            resp_msg['Bcc'] = config['smtp']['bcc']
            bcc = config['smtp']['bcc'].split(",")
        dests = to
        if cc or bcc:
            dests = [to]
            if cc:
                dests += cc
            if bcc:
                dests += bcc

        resp_msg['Subject'] = subject
        resp_msg.attach(MIMEText(text, 'plain', 'utf-8'))

        smtp_server = smtplib.SMTP(config['smtp']['host'], config['smtp']['port'])
        smtp_server.starttls()
        smtp_server.login(config['smtp']['user'], config['smtp']['password'])
        text = resp_msg.as_string().encode('ascii')
        smtp_server.sendmail(config['smtp']['from'], dests, text)
        smtp_server.quit()

    def _split_text_into_sentences(self, src_lang, tgt_lang, text):
        config = json.load(open(self.config_file))
        sentences = []
        params = urllib.urlencode({'lang_source': src_lang.encode('utf-8'),
                                   'lang_target': tgt_lang.encode('utf-8'),
                                   'text': text.encode('utf-8')})
        headers = {'Content-type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'}
        conn = httplib.HTTPConnection("{0}".format(config['text_splitter']['host']))
        try:
            conn.request('POST', config['text_splitter']['path'], params, headers)
            resp = conn.getresponse()
            data = resp.read()
            json_data = json.loads(data)
            sentences = json_data['sentences']
        except Exception as ex:
            self.logger.error(ex)
        finally:
            conn.close()
        return sentences

    def _translate_sentence(self, src_lang, tgt_lang, sentence):
        config = json.load(open(self.config_file))
        lang_pair = '{0}-{1}'.format(src_lang, tgt_lang)
        server_data = config['servers'][lang_pair]
        client = Client(server_data['host'], server_data['port'])
        resp = client.query(sentence)
        json_resp = json.loads(resp)
        translation = json_resp['out']

        if tgt_lang in ['ja', 'zh']:
            translation = translation.replace(' ', '')

        translation = translation.replace('&quot;', '"')
        translation = translation.replace('&apos;', "'")

        return translation

    def _parse_list_response(self, line):
        flags, delimiter, mailbox_name = list_resp_pattern.match(line).groups()
        mailbox_name = mailbox_name.strip('"')
        return (flags, delimiter, mailbox_name)

    def _init_localization(self):
        self.logger.info("Initializating localization.")
        config = json.load(open(self.config_file))
        ui_locale = config.get('ui_locale', '')
        self.logger.info("Interface locale: {0}".format(ui_locale))
        locale.setlocale(locale.LC_ALL, ui_locale.encode('ascii'))
        filename = "res/messages_%s.mo" % locale.getlocale()[0][0:2]

        try:
            logging.debug("Opening message file %s for locale %s", filename, locale.getlocale()[0])
            trans = gettext.GNUTranslations(open(filename, "rb"))
        except IOError:
            logging.debug("Locale not found. Using default messages")
            trans = gettext.NullTranslations()

        trans.install()

    def _prepare_mailboxes(self):
        self.logger.info("Preparing mailboxes.")
        config = json.load(open(self.config_file))
        mail = None
        try:
            mail = imaplib.IMAP4_SSL(config['imap']['host'], config['imap']['port'])
            mail.login(config['imap']['user'], config['imap']['password'])

            type, data = mail.list()
            processed_mailbox = config['imap']['processed_request_mailbox']
            ignored_mailbox = config['imap']['ignored_request_mailbox']
            is_processed_request_mailbox_found = False
            is_ignored_request_mailbox_found = False
            for line in data:
                flags, delimiter, mailbox_name = self._parse_list_response(line)
                if not is_processed_request_mailbox_found and mailbox_name == processed_mailbox:
                    is_processed_request_mailbox_found = True
                if not is_ignored_request_mailbox_found and mailbox_name == ignored_mailbox:
                    is_ignored_request_mailbox_found = True
            if not is_processed_request_mailbox_found:
                mail.create(processed_mailbox)
            if not is_ignored_request_mailbox_found:
                mail.create(ignored_mailbox)

        except Exception, ex:
            raise

        finally:
            if mail is not None:
                mail.logout()

    def _load_requests(self):
        config = json.load(open(self.config_file))
        requests_filename = "{0}_requests.json.gz".format(config['id'])
        if os.path.isfile(requests_filename):
            with gzip.open(requests_filename, 'r') as input_file:
                str = input_file.read().decode('utf-8')
                self._requests = json.loads(str)

    def _save_requests(self):
        config = json.load(open(self.config_file))
        requests_filename = "{0}_requests.json.gz".format(config['id'])
        if self._requests:
            with gzip.open(requests_filename, 'w') as output_file:
                output_file.write(json.dumps(self._requests, output_file, ensure_ascii=False).encode('utf-8'))
        else:
            if (os.path.isfile(requests_filename)):
                os.remove(requests_filename)

    def _enqueue_request(self, req):
        self._requests.append(req)
        self._save_requests()

    def _dequeue_request(self):
        self._requests.pop(0)
        self._save_requests()

    def _fetch_requests(self, mail):
        config = json.load(open(self.config_file))

        if self._first_time:
            self._load_requests()
            if len(self._requests) > 0:
                self.logger.info("Previous {0} pending requests found on disk.".format(len(self._requests)))
                return

        self._requests = []

        type, data = mail.uid('SEARCH', None, 'ALL')
        mail_uids = data[0]
        uid_list = mail_uids.split()
        if (len(uid_list) > 0):
            first_email_uid = int(uid_list[0])
            last_email_uid = int(uid_list[-1])

            for mail_uid in range(first_email_uid, last_email_uid + 1):
                email_from = None
                try:
                    type, data = mail.uid('FETCH', mail_uid, '(RFC822)')

                    for response_part in data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_string(response_part[1])
                            email_date = msg['date'].decode('utf-8')
                            email_from = msg['from'].decode('utf-8')
                            email_subject = msg['subject'].decode('utf-8')
                            email_body = get_decoded_email_body(response_part[1]).decode('utf-8')

                            subject_match = subject_pattern.match(email_subject)
                            if subject_match:
                                src_lang = subject_match.group(1).lower()
                                tgt_lang = subject_match.group(2).lower()
                                lang_pair = '{0}-{1}'.format(src_lang, tgt_lang)

                                if lang_pair not in config['servers']:
                                    raise Exception('Unsupported language pair: {0}'.format(lang_pair))

                                self.logger.info('Queuing request...')
                                self.logger.info("Uid: {0}".format(mail_uid))
                                self.logger.info("Date: {0}".format(email_date))
                                self.logger.info("From: {0}".format(email_from))
                                self.logger.info("Subject: {0}".format(email_subject))

                                reply = _("Translation request received.\n\n{0}\n\nPlease wait...\n\n---------- Original text ----------\n\nLanguage pair: {1}-{2}\n\n{3}\n\n-- The content of this message has been generated by KNMT {4}. --").decode('utf-8')
                                if (len(self._requests) == 0):
                                    queue_msg = _("It will be processed shortly.")
                                else:
                                    queue_msg = _("Requests before you: {0}").format(len(self._requests))

                                subject = _('Translation request received: {0}').format(email_subject)
                                self._send_mail(email_from, subject.decode('utf-8'),
                                                reply.format(queue_msg.decode('utf-8'), src_lang, tgt_lang, email_body, knmt_version.decode('utf-8')))

                                req = {'uid': mail_uid, 'date': email_date, 'ffrom': email_from, 'subject': email_subject, 'body': email_body.replace('\r', '')}
                                self._enqueue_request(req)

                            else:
                                raise Exception('Incorrect protocol or unknown language pair: {0}'.format(email_subject))
                except Exception, ex_msg:
                    self.logger.info("Error: {0}\n\nMoving message to Ignored mailbox\n\n".format(ex_msg))
                    mail.uid('COPY', mail_uid, config['imap']['ignored_request_mailbox'])
                    mail.uid('STORE', mail_uid, '+FLAGS', '\\Deleted')
                    mail.expunge()
                    if email_from is not None:
                        reply = _("Error: {0}\n\nMoving message to Ignored mailbox\n\n").format(ex_msg)
                        subject = _('Translation error: {0}').format(email_subject)
                        self._send_mail(email_from, subject.decode('utf-8'), reply)

    def _process_mail(self):
        self._first_time = True
        while True:
            config = json.load(open(self.config_file))

            mail = None
            try:
                mail = imaplib.IMAP4_SSL(config['imap']['host'], config['imap']['port'])
                mail.login(config['imap']['user'], config['imap']['password'])
                mail.select(config['imap']['incoming_request_mailbox'])

                self._fetch_requests(mail)

                while self._requests:
                    req = self._requests[0]
                    try:
                        self.logger.info("Processing mail...")
                        self.logger.info("Uid: {0}".format(req['uid']))
                        self.logger.info("Date: {0}".format(req['date']))
                        self.logger.info("From: {0}".format(req['ffrom']))
                        self.logger.info("Subject: {0}".format(req['subject']))

                        subject_match = subject_pattern.match(req['subject'])
                        src_lang = subject_match.group(1).lower()
                        tgt_lang = subject_match.group(2).lower()
                        lang_pair = '{0}-{1}'.format(src_lang, tgt_lang)
                        self.logger.info('Text: {0}\n'.format(req['body']))

                        paragraphes = split_text_into_paragraphes(req['body'], src_lang, tgt_lang)

                        translation = ''
                        for paragraph in paragraphes:
                            sentences = self._split_text_into_sentences(src_lang, tgt_lang, paragraph)

                            if tgt_lang in ['ja', 'zh']:
                                translation += '\u3000'  # Ideographic space.

                            for sentence in sentences:
                                translated_sentence = self._translate_sentence(src_lang, tgt_lang, sentence.encode('utf-8'))
                                translation += translated_sentence.rstrip()
                                if tgt_lang not in ['ja', 'zh']:
                                    translation += ' '
                            translation += "\n\n"

                        if 'post_processing_cmd' in config and lang_pair in config['post_processing_cmd']:
                            translation_file = tempfile.NamedTemporaryFile()
                            try:
                                translation_file.write(translation.encode('utf-8'))
                                translation_file.seek(0)
                                cmd = config['post_processing_cmd'][lang_pair].replace("$FILE", translation_file.name)
                                translation = subprocess.check_output(cmd, shell=True).decode('utf-8')
                            finally:
                                translation_file.close()

                        self.logger.info("Translation: {0}".format(translation))

                        reply = _("{0}\n\n---------- Original text ----------\n\n{1}\n\n-- The content of this message has been generated by KNMT {2}. --").decode('utf-8')

                        subject = _('Translation result: {0}').format(req['subject'])
                        self._send_mail(req['ffrom'], subject.decode('utf-8'),
                                        reply.format(translation, req['body'], knmt_version.decode('utf-8')))

                        self.logger.info("Moving message to Processed mailbox.\n\n")
                        mail.uid('COPY', req['uid'], config['imap']['processed_request_mailbox'])
                        mail.uid('STORE', req['uid'], '+FLAGS', '\\Deleted')
                        mail.expunge()

                    except Exception, ex_msg:
                        self.logger.info('Error: {0}'.format(ex_msg))
                        self.logger.info("Moving message to Ignored mailbox.\n\n")
                        mail.uid('COPY', req['uid'], config['imap']['ignored_request_mailbox'])
                        mail.uid('STORE', req['uid'], '+FLAGS', '\\Deleted')
                        mail.expunge()

                        reply = "Error: {0}\n\nYour message has been discarded.\n\n-- The content of this message has been generated by KNMT {1}. --"
                        self._send_mail(req['ffrom'],
                                        'Translation result for {0}_{1} request'.format(src_lang, tgt_lang),
                                        reply.format(ex_msg, knmt_version))

                    self._dequeue_request()

            except Exception, ex:
                self.logger.error(ex)

            finally:
                if mail is not None:
                    mail.close()
                    mail.logout()

            self._first_time = False
            time.sleep(int(config['next_mail_handling_delay']))

    def run(self):
        logging.config.fileConfig(self.log_config_file)
        self.logger = logging.getLogger('default')

        context = daemon.DaemonContext(working_directory='.')
        context.files_preserve = [self.logger.root.handlers[1].stream.fileno()]

        with context:
            try:
                self.logger.info("Starting mail_handler daemon...")
                self._init_localization()
                self._prepare_mailboxes()
                self._process_mail()
            except Exception, ex:
                self.logger.error(ex)
            finally:
                self.logger.info("Stopping mail_handler daemon.")


def command_line(arguments=None):
    parser = argparse.ArgumentParser(description="Launch a translation mail handler daemon.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config", nargs="?", help="Main configuration file.", default="conf/mail_handler.json")
    parser.add_argument("--log_config", nargs="?", help="Log configuration file.", default="conf/mail_handler_logging.conf")

    args = parser.parse_args(args=arguments)

    handler = MailHandler(args.config, args.log_config)
    handler.run()


if __name__ == "__main__":
    command_line()
