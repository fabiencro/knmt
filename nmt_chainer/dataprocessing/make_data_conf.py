from nmt_chainer.utilities import argument_parsing_tools
import logging

logging.basicConfig()
log = logging.getLogger("rnns:make_data_config")
log.setLevel(logging.INFO)

def define_parser(parser):
    parser.add_argument(
        "src_fn", help="source language text file for training data")
    parser.add_argument(
        "tgt_fn", help="target language text file for training data")
    parser.add_argument(
        "save_prefix", help="created files will be saved with this prefix")
    parser.add_argument("--align_fn", help="align file for training data")
    parser.add_argument("--src_voc_size", type=int, default=32000,
                        help="limit source vocabulary size to the n most frequent words")
    parser.add_argument("--tgt_voc_size", type=int, default=32000,
                        help="limit target vocabulary size to the n most frequent words")
#     parser.add_argument("--add_to_valid_set_every", type = int)
#     parser.add_argument("--shuffle", default = False, action = "store_true")
#     parser.add_argument("--enable_fast_shuffle", default = False, action = "store_true")

    parser.add_argument("--max_nb_ex", type = int, help = "only use the first MAX_NB_EX examples")
    
    parser.add_argument("--test_src", help = "specify a source test set")
    parser.add_argument("--test_tgt", help = "specify a target test set")
    
    parser.add_argument("--dev_src", help = "specify a source dev set")
    parser.add_argument("--dev_tgt", help = "specify a target dev set")
    parser.add_argument("--mode_align", choices = ["unk_align", "all_align"])
    parser.add_argument("--use_voc", help = "specify an exisiting vocabulary file")
    
    parser.add_argument("--tgt_segmentation_type", choices = ["word", "word2char", "char"], default = "word")
    parser.add_argument("--src_segmentation_type", choices = ["word", "word2char", "char"], default = "word")
    
    parser.add_argument("--bpe_src", type = int, help = "Apply BPE to source side, with this many merges")
    parser.add_argument("--bpe_tgt", type = int, help = "Apply BPE to target side, with this many merges")
    
    parser.add_argument("--joint_bpe", type = int, help = "Apply joint BPE, with this many merges")
    
    parser.add_argument("--latin_src", default = False, action = "store_true", help = "apply preprocessing for latin scripts to source")
    parser.add_argument("--latin_tgt", default = False, action = "store_true", help = "apply preprocessing for latin scripts to target")
    
    parser.add_argument("--latin_type", choices = "all_adjoint caps_isolate".split(), default = "all_adjoint", help = "choose preprocessing for latin scripts to source")

class CommandLineValuesException(Exception):
    pass

def cmdline(arguments=None):
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Prepare training data.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    define_parser(parser)
    
    args = parser.parse_args(args = arguments)
    
    do_make_data(args)
    
def get_parse_option_orderer():
    por = argument_parsing_tools.ParseOptionRecorder()
    define_parser(por)
    return por
    
def make_data_config(args):
    parse_option_orderer = get_parse_option_orderer()
    config = parse_option_orderer.convert_args_to_ordered_dict(args)
    
    config.set_readonly()
    
    if not ((config.test_src is None) == (config.test_tgt is None)):
        raise CommandLineValuesException("Command Line Error: either specify both --test_src and --test_tgt or neither")

    if not ((config.dev_src is None) == (config.dev_tgt is None)):
        raise CommandLineValuesException("Command Line Error: either specify both --dev_src and --dev_tgt or neither")
    
    if config.joint_bpe is not None and (config.bpe_src is not None or config.bpe_tgt is not None):
        raise CommandLineValuesException("Command Line Error: --joint_bpe is incompatible with --bpe_src and --bpe_tgt")
    return config

def do_make_data(args):
    import nmt_chainer.dataprocessing.make_data
    config = make_data_config(args)
    nmt_chainer.dataprocessing.make_data.do_make_data(config)
    
if __name__ == '__main__':
    cmdline()

