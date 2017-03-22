import os
import json
import urllib
import sqlite3
import time
from nmt_chainer.utilities.utils import ensure_path
from nmt_chainer.utilities import bleu_computer
from nmt_chainer.utilities import graph_training
from nmt_chainer.training_module.train_config import load_config_train
from nmt_chainer.translation.eval_config import load_config_eval
from collections import defaultdict
import logging
logging.basicConfig()
log = logging.getLogger("rnns:recap")
log.setLevel(logging.INFO)

data_config_suffix = ".data.config"
train_config_suffix = ".train.config"
eval_config_suffix = ".eval.config.json"


def filenamize(path):
    return urllib.quote_plus(path.replace("/", "xDIRx"))


def process_eval_config(config_fn, dest_dir):
    log.info("processing eval config %s " % config_fn)
    assert config_fn.endswith(eval_config_suffix)

    eval_prefix = config_fn[:-len(eval_config_suffix)]
    log.info("eval prefix: %s " % eval_prefix)

    urlname = filenamize(eval_prefix) + ".html"
    log.info("urlname: %s " % eval_prefix)

    time_config_created = os.path.getmtime(config_fn)

    dest_filename = os.path.join(dest_dir, urlname)

    log.info("create file: %s" % dest_filename)
    f = open(dest_filename, "w")

    config = load_config_eval(config_fn)
#     config = json.load(open(config_fn))
#     ref_fn = config["ref"]

    ref_fn = config.output.ref
    if ref_fn is not None:
        bc_with_unk = bleu_computer.get_bc_from_files(ref_fn, eval_prefix)
        bc_unk_replaced = bleu_computer.get_bc_from_files(
            ref_fn, eval_prefix + ".unk_replaced")
        bleu_unk_replaced = bc_unk_replaced.bleu()
    else:
        bc_with_unk = "#No Ref#"
        bc_unk_replaced = "#No Ref#"
        bleu_unk_replaced = -1

    f.write("<html><body>")
    f.write("Config Filename: %s <p>" % config_fn)
    f.write("eval prefix: %s <p>" % eval_prefix)
    f.write("config file created/modified:%s <p>" % time.ctime(time_config_created))
    f.write("BLEU: %s <p>" % bc_with_unk)
    f.write("BLEU with unk replaced: %s <p>" % bc_unk_replaced)
    for line in open(config_fn):
        f.write(line)
    f.write("</body></html>")

    return urlname, bleu_unk_replaced


def process_data_config(config_fn, dest_dir, data_to_train):
    log.info("processing data config %s " % config_fn)
    assert config_fn.endswith(data_config_suffix)

    data_prefix = config_fn[:-len(data_config_suffix)]
    log.info("data prefix: %s " % data_prefix)

    urlname = filenamize(data_prefix) + ".html"
    log.info("urlname: %s " % data_prefix)

    time_config_created = os.path.getmtime(config_fn)

    dest_filename = os.path.join(dest_dir, urlname)

    log.info("create file: %s" % dest_filename)
    f = open(dest_filename, "w")

    config = json.load(open(config_fn))

    f.write("<html><body>")
    f.write("Config Filename: %s <p>" % config_fn)
    f.write("data prefix: %s <p>" % data_prefix)
    f.write("config file created/modified:%s <p>" % time.ctime(time_config_created))
    for train_urlname in data_to_train[data_prefix]:
        f.write("train: <a href = '../train/%s'>%s</a><p>" % (train_urlname, train_urlname))

    f.write("<pre><code>")
    for line in open(config_fn):
        f.write(line)
    f.write("</code></pre>")
    f.write("</body></html>")

    return urlname, data_prefix, (config["src_fn"], config["tgt_fn"]), time_config_created, config["src_voc_size"], config["tgt_voc_size"]


def process_train_config(config_fn, dest_dir):
    log.info("processing train config %s " % config_fn)
    assert config_fn.endswith(train_config_suffix)

    train_prefix = config_fn[:-len(train_config_suffix)]
    log.info("train prefix: %s " % train_prefix)

    urlname = filenamize(train_prefix) + ".html"
    log.info("urlname: %s " % train_prefix)

    dest_filename = os.path.join(dest_dir, urlname)

    if os.path.exists(dest_filename):
        time_file_modif = os.path.getmtime(dest_filename)
    else:
        time_file_modif = None
    try:
        time_last_exp = os.path.getmtime(train_prefix + ".result.sqlite")
    except OSError:
        time_last_exp = 1

    time_config_created = os.path.getmtime(config_fn)

    if time_file_modif is not None and (time_last_exp + 20) < time_file_modif:
        log.info("skipping creation for %s" % train_prefix)

    log.info("create file: %s" % dest_filename)
    f = open(dest_filename, "w")

    try:
        db = sqlite3.connect(train_prefix + ".result.sqlite")
        c = db.cursor()

        c.execute("SELECT date, bleu_info, iteration, loss, bleu, dev_loss, dev_bleu, avg_time, avg_training_loss FROM exp_data")
        date, bleu_info, iteration, test_loss, test_bleu, dev_loss, dev_bleu, avg_time, avg_training_loss = zip(*list(c.fetchall()))

        infos = dict(
            last_nb_iterations=iteration[-1],
            nb_records=len(test_loss),
            date_last_exp=date[-1],
            best_test_loss=min(test_loss),
            best_dev_loss=min(dev_loss),
            max_test_bleu=max(test_bleu * 100),
            max_dev_bleu=max(dev_bleu * 100),
            best_avg_training_loss=min(avg_training_loss[1:]) if len(avg_training_loss) > 1 else 0,
            avg_time_all=sum(t for t in avg_time if t is not None) / len(avg_time)
        )

    except sqlite3.OperationalError:
        infos = None

    config = load_config_train(config_fn, no_error=True)
#     config = json.load(open(config_fn))

#     data_prefix = config["command_line"]["data_prefix"]
    data_prefix = config.training_management.data_prefix

    try:
        graph_training.generate_graph(sqldb=train_prefix + ".result.sqlite",
                                      #                                   lib = "plotly",
                                      dest=os.path.join(dest_dir, filenamize(train_prefix) + ".graph.html"),
                                      title=train_prefix)
        has_graph = True
    except sqlite3.OperationalError as e:
        log.warn("could not create graph for %s" % train_prefix)
        print e
        has_graph = False

    f.write("<html><body>")
    f.write("Config Filename: %s <p>" % config_fn)
    f.write("train prefix: %s <p>" % train_prefix)

    f.write("data: <a href = '../data/%s'>%s</a><p>" % (filenamize(data_prefix) + ".html", filenamize(data_prefix)))
    if has_graph:
        f.write("<a href = '%s'>graph</a><p>" % (filenamize(train_prefix) + ".graph.html"))

    f.write("config file created/modified:%s <p>" % time.ctime(time_config_created))
    f.write("last exp:%s <p>" % time.ctime(time_last_exp))
    time_file_modif_new = os.path.getmtime(dest_filename)
    f.write("this file created/modified:%s <p>" % time.ctime(time_file_modif_new))

    if infos is not None:
        for key in sorted(infos.keys()):
            f.write("%s : %r <p>" % (key, infos[key]))

    f.write("<pre><code>")
    for line in open(config_fn):
        f.write(line)
    f.write("</code></pre>")
    f.write("</body></html>")

    return urlname, data_prefix, time_last_exp, infos


def define_parser(parser):
    parser.add_argument("source_dir", help="directory containing experiments results (also look into subdirectories)")
    parser.add_argument("target_dir", help="directory where html files will be generated")


def do_recap(args):
    data_dir = os.path.join(args.target_dir, "data")
    train_dir = os.path.join(args.target_dir, "train")
    eval_dir = os.path.join(args.target_dir, "eval")

    ensure_path(args.target_dir)
    ensure_path(data_dir)
    ensure_path(train_dir)
    ensure_path(eval_dir)

    index = open(os.path.join(args.target_dir, "index.html"), "w")
    index.write("<html><body>")

    data_urlname_list = defaultdict(list)
    train_urlname_list = defaultdict(list)

    data_config_fn_list = []
    eval_config_fn_list = []

    itdir = os.walk(args.source_dir)

    data_to_train = defaultdict(list)
    train_to_data = {}
    for current_dir, dirs, files in itdir:
        for fn in files:
            if fn.endswith(train_config_suffix):
                fn_full = os.path.join(current_dir, fn)
                urlname, data_prefix, time_last_exp, infos = process_train_config(
                    fn_full, train_dir)
                data_to_train[data_prefix].append(urlname)
                train_to_data[urlname] = data_prefix
                train_urlname_list[data_prefix].append(
                    (time_last_exp, urlname, infos))
            elif fn.endswith(data_config_suffix):
                fn_full = os.path.join(current_dir, fn)
                data_config_fn_list.append(fn_full)
            elif fn.endswith(eval_config_suffix):
                fn_full = os.path.join(current_dir, fn)
                eval_config_fn_list.append(fn_full)
            else:
                pass

    data_to_srctgt = {}
    for fn_full in data_config_fn_list:
        urlname, data_prefix, src_tgt_fn, time_config_created, src_voc_size, tgt_voc_size = process_data_config(fn_full, data_dir, data_to_train)
        data_urlname_list[src_tgt_fn].append((time_config_created, urlname, data_prefix, src_voc_size, tgt_voc_size))
        data_to_srctgt[data_prefix] = src_tgt_fn

    index.write("<h1>DATA</h1><p>")
    for src_tgt_fn, urlname_list in data_urlname_list.iteritems():
        index.write("<h3>** src: %s | tgt: %s **</h3>" % src_tgt_fn)
        urlname_list.sort(reverse=True)
        for time_config_created, urlname, data_prefix, src_voc_size, tgt_voc_size in urlname_list:
            index.write('%s s:%i t:%i \t<a href = "data/%s">%s</a><p/>' % (time.ctime(time_config_created),
                                                                           src_voc_size, tgt_voc_size, urlname, data_prefix))
    train_urlname_list_src_tgt = defaultdict(list)
    for data_path, urlname_list in train_urlname_list.iteritems():
        if data_path in data_to_srctgt:
            train_urlname_list_src_tgt[data_to_srctgt[data_path]
                                       ] += urlname_list
        else:
            train_urlname_list_src_tgt[("unk", "unk")] += urlname_list

    current_time = time.time()
    index.write("<h1>TRAIN</h1><p>")
    for src_tgt_fn in sorted(train_urlname_list_src_tgt.keys(), key=lambda x: max(train_urlname_list_src_tgt[x][i][0] for i in xrange(len(train_urlname_list_src_tgt[x]))), reverse=True):
        urlname_list = train_urlname_list_src_tgt[src_tgt_fn]
        index.write("<h3>** src: %s | tgt: %s **</h3>" % src_tgt_fn)
        urlname_list.sort(reverse=True)
        for time_last_exp, urlname, infos in urlname_list:
            if abs(time_last_exp - current_time) < 3000:
                recently_updated = True
            else:
                recently_updated = False
            if recently_updated:
                timestring = "<b>%s [RCT]</b>" % time.ctime(time_last_exp)
            else:
                timestring = "%s" % time.ctime(time_last_exp)
            index.write('%s <a href = "train/%s">%s</a><p/>' % (timestring, urlname, urlname.split("xDIRx")[-1]))
            if infos is not None:
                for key in sorted(infos.keys()):
                    index.write("%s : %r  ||| " % (key, infos[key]))
            index.write("<p>")

    index.write("<h1>EVAL</h1><p>")
    for fn_full in eval_config_fn_list:
        urlname, bleu = process_eval_config(fn_full, eval_dir)
        index.write('<a href = "eval/%s">%s</a> %f <p/>' % (urlname, urlname.split("xDIRx")[-1], bleu))

    index.write("</body></html>")


def commandline():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    define_parser(parser)
    args = parser.parse_args()
    do_recap(args)


if __name__ == '__main__':
    commandline()
