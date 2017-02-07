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
    
    
    
def cmdline(arguments=None):
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Prepare training data.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    define_parser(parser)
    
    args = parser.parse_args(args = arguments)
    
    do_make_data(args)
    
def do_make_data(args):
    import nmt_chainer.dataprocessing.make_data
    nmt_chainer.dataprocessing.make_data.do_make_data(args)