import argparse
from nmt_chainer.utilities import graph_training
from nmt_chainer.utilities import replace_tgt_unk
from nmt_chainer.utilities import expe_recap
from nmt_chainer.utilities import bleu_computer

# from nmt_chainer.models import char_encdec
from nmt_chainer.utilities import extract_processed_data

def define_parser(parser):
    subparsers = parser.add_subparsers(dest="__sub_subcommand_name")

    graph_parser = subparsers.add_parser('graph', description="Create a graph of the training data.",
                                         help="Create a graph of the training data.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    graph_training.define_parser(graph_parser)

    replace_tgt_parser = subparsers.add_parser('replace_tgt_unk', description="Replace UNK tags using a dictionnary.",
                                               help="Replace UNK tags using a dictionnary.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    replace_tgt_unk.define_parser(replace_tgt_parser)

    recap_parser = subparsers.add_parser('recap', description="Generate recap of experiments.",
                                         help="Generate recap of experimets.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    expe_recap.define_parser(recap_parser)

    bleu_parser = subparsers.add_parser('bleu', description="Compute BLEU score.",
                                        help="Compute BLEU score.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    bleu_computer.define_parser(bleu_parser)

#     charenc_parser = subparsers.add_parser('charenc', description="Compute CharEnc.",
#                                         help="Compute CharEnc.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     char_encdec.define_parser(charenc_parser)
    
    extract_data_parser = subparsers.add_parser('extract_data', description="Extract processed data.",
                                         help="Extract processed data.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    extract_processed_data.define_parser(extract_data_parser)    

def do_utils(args):
    func = {"graph": graph_training.do_graph,
            "replace_tgt_unk": replace_tgt_unk.do_replace,
            "recap": expe_recap.do_recap,
            "bleu": bleu_computer.do_bleu,
#             "charenc": char_encdec.do_command,
            "extract_data": extract_processed_data.do_extract
            }[args.__sub_subcommand_name]
    func(args)
