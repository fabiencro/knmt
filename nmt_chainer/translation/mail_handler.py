#!/usr/bin/env python
"""mail_handler.py: Daemon that monitors an IMAP account for messages containing translation requests."""
__author__ = "Frederic Bergeron"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "bergeron@pa.jst.jp"
__status__ = "Development"

import daemon
import email
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
import httplib
import imaplib
import json
import logging
import logging.config
import re
import smtplib
import time
import urllib

from nmt_chainer._version import __version__ as knmt_version
from nmt_chainer.translation.client import Client

logging.config.fileConfig('conf/mail_handler_logging.conf')
logger = logging.getLogger('default')

CONFIG_FILE = 'conf/mail_handler.json'

list_resp_pattern = re.compile('(.*?) "(.*)" (.*)')


def send_mail(to, subject, text):
    config = json.load(open(CONFIG_FILE))
    resp_msg = MIMEMultipart()
    resp_msg['From'] = config['smtp']['from']
    resp_msg['To'] = to
    resp_msg['Subject'] = subject

    resp_msg.attach(MIMEText(text, 'plain', 'utf-8'))

    smtp_server = smtplib.SMTP(config['smtp']['host'], config['smtp']['port'])
    smtp_server.starttls()
    smtp_server.login(config['smtp']['user'], config['smtp']['password'])
    text = resp_msg.as_string().encode('ascii')
    smtp_server.sendmail(config['smtp']['from'], to, text)
    smtp_server.quit()


def split_text_into_sentences(src_lang, tgt_lang, text):
    config = json.load(open(CONFIG_FILE))
    sentences = None
    params = urllib.urlencode({'lang_source': src_lang, 'lang_target': tgt_lang, 'text': text})
    headers = {'Content-type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'}
    conn = httplib.HTTPConnection("{0}".format(config['text_splitter']['host']))
    try:
        conn.request('POST', config['text_splitter']['path'], params, headers)
        resp = conn.getresponse()
        data = resp.read()
        json_data = json.loads(data)
        sentences = json_data['sentences']
    except Exception as ex:
        logger.error(ex)
    finally:
        conn.close()
    return sentences


def translate_sentence(src_lang, tgt_lang, sentence):
    config = json.load(open(CONFIG_FILE))
    server_data = config['languages']['{0}-{1}'.format(src_lang, tgt_lang)]['server']
    client = Client(server_data['host'], server_data['port'])
    resp = client.query(sentence)
    json_resp = json.loads(resp)
    translation = json_resp['out'].encode('utf-8')
    if tgt_lang in ['ja', 'zh']:
        translation = translation.replace(' ', '')
    translation = translation.replace('&quot;', '"')
    translation = translation.replace('&apos;', "'")
    return translation


def parse_list_response(line):
    flags, delimiter, mailbox_name = list_resp_pattern.match(line).groups()
    mailbox_name = mailbox_name.strip('"')
    return (flags, delimiter, mailbox_name)


def prepare_mailboxes():
    logger.info("Preparing mailboxes.")
    config = json.load(open(CONFIG_FILE))
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
            flags, delimiter, mailbox_name = parse_list_response(line)
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


def process_mail():
    while True:
        config = json.load(open(CONFIG_FILE))
        mail = None
        try:
            mail = imaplib.IMAP4_SSL(config['imap']['host'], config['imap']['port'])
            mail.login(config['imap']['user'], config['imap']['password'])
            mail.select(config['imap']['incoming_request_mailbox'])

            type, data = mail.search(None, 'ALL')
            mail_ids = data[0]
            id_list = mail_ids.split()
            if (len(id_list) > 0):
                first_email_id = int(id_list[0])
                latest_email_id = int(id_list[-1])
                for i in range(first_email_id, latest_email_id + 1):
                    try:
                        type, data = mail.fetch(i, '(RFC822)')

                        for response_part in data:
                            if isinstance(response_part, tuple):
                                msg = email.message_from_string(response_part[1])
                                email_date = msg['date']
                                email_from = msg['from']
                                email_subject = msg['subject']
                                if msg.is_multipart():
                                    for child_msg in msg.get_payload():
                                        if child_msg.get_content_type() == 'text/plain':
                                            email_body = child_msg.get_payload(decode=True)
                                else:
                                    email_body = msg.get_payload(decode=True)
                                logger.info("Processing mail...")
                                logger.info("Date: {0}".format(email_date))
                                logger.info("From: {0}".format(email_from))
                                logger.info("Subject: {0}".format(email_subject))

                                is_request_ignored = True

                                subject_pattern = re.compile('([a-zA-Z][a-zA-Z])_([a-zA-Z][a-zA-Z])')
                                subject_match = subject_pattern.match(email_subject)
                                if subject_match:
                                    src_lang = subject_match.group(1).lower()
                                    tgt_lang = subject_match.group(2).lower()
                                    logger.info('Text: {0}\n'.format(email_body))

                                    if not "{0}-{1}".format(src_lang, tgt_lang) in config['languages']:
                                        raise Exception('Unsupported language pair: {0}-{1}'.format(src_lang, tgt_lang))

                                    sentences = split_text_into_sentences(src_lang, tgt_lang, email_body)

                                    translation = ''
                                    for sentence in sentences:
                                        translated_sentence = translate_sentence(src_lang, tgt_lang, sentence.encode('utf-8'))
                                        translation += translated_sentence
                                    logger.info("Translation: {0}".format(translation))

                                    reply = """
{0}

---------- Original text submitted on {1} ----------

{2}

-- The content of this message has been generated by KNMT {3}. --
"""
                                    send_mail(email_from,
                                              'Translation result for {0}_{1} request'.format(src_lang, tgt_lang),
                                              reply.format(translation, email_date, email_body, knmt_version))

                                    logger.info("Moving message to Processed mailbox.\n\n")
                                    mail.copy(i, config['imap']['processed_request_mailbox'])
                                    mail.store(i, '+FLAGS', '\\Deleted')
                                    mail.expunge()

                    except Exception, ex_msg:
                        logger.info('Error: {0}'.format(ex_msg))
                        logger.info("Moving message to Ignored mailbox.\n\n")
                        mail.copy(i, config['imap']['ignored_request_mailbox'])
                        mail.store(i, '+FLAGS', '\\Deleted')
                        mail.expunge()

                        reply = """
Error: {0}

Your message has been discarded.

-- The content of this message has been generated by KNMT {1}. --
"""
                        send_mail(email_from,
                                  'Translation result for {0}_{1} request'.format(src_lang, tgt_lang),
                                  reply.format(ex_msg, knmt_version))

        except Exception, ex:
            logger.error(ex)

        finally:
            if mail is not None:
                mail.close()
                mail.logout()

        time.sleep(int(config['next_mail_handling_delay']))


def run():
    context = daemon.DaemonContext(working_directory='.')
    context.files_preserve = [logger.root.handlers[1].stream.fileno()]

    with context:
        try:
            logger.info("Starting mail_handler daemon...")
            prepare_mailboxes()
            process_mail()
        except Exception, ex:
            logger.error(ex)
        finally:
            logger.info("Stopping mail_handler daemon.")

if __name__ == "__main__":
    run()
