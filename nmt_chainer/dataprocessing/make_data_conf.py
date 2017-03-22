from nmt_chainer.utilities import argument_parsing_tools
import logging

logging.basicConfig()
log = logging.getLogger("rnns:make_data_config")
log.setLevel(logging.INFO)

_CONFIG_SECTION_TO_DESCRIPTION = {"data": "Files used for training, dev, testing, ...",
                                  "processing": "Processing applied to the data."}


def define_parser(parser):
    parser.add_argument(
        "src_fn", nargs="?",
        action=argument_parsing_tools.ArgumentActionNotOverwriteWithNone, help="source language text file for training data")
    parser.add_argument(
        "tgt_fn", nargs="?",
        action=argument_parsing_tools.ArgumentActionNotOverwriteWithNone, help="target language text file for training data")
    parser.add_argument(
        "save_prefix", nargs="?",
        action=argument_parsing_tools.ArgumentActionNotOverwriteWithNone, help="created files will be saved with this prefix")

    parser.add_argument("--description", help="Optional message to be stored in the configuration file")

    data_group = parser.add_argument_group(_CONFIG_SECTION_TO_DESCRIPTION["data"])

    data_group.add_argument("--src_fn", dest="src_fn",
                            action=argument_parsing_tools.ArgumentActionNotOverwriteWithNone,
                            help="same as positional argument src_fn")

    data_group.add_argument("--tgt_fn", dest="tgt_fn",
                            action=argument_parsing_tools.ArgumentActionNotOverwriteWithNone,
                            help="same as positional argument tgt_fn")

    data_group.add_argument("--save_prefix", dest="save_prefix",
                            action=argument_parsing_tools.ArgumentActionNotOverwriteWithNone,
                            help="same as positional argument save_prefix")

    data_group.add_argument("--max_nb_ex", type=int, help="only use the first MAX_NB_EX examples")

    data_group.add_argument("--dev_src", help="specify a source dev set")
    data_group.add_argument("--dev_tgt", help="specify a target dev set")

    data_group.add_argument("--test_src", help="specify a source test set")
    data_group.add_argument("--test_tgt", help="specify a target test set")

    data_group.add_argument("--mode_align", choices=["unk_align", "all_align"])
    data_group.add_argument("--align_fn", help="align file for training data")

    processing_group = parser.add_argument_group(_CONFIG_SECTION_TO_DESCRIPTION["processing"])
    processing_group.add_argument("--src_voc_size", type=int, default=32000,
                                  help="limit source vocabulary size to the n most frequent words")
    processing_group.add_argument("--tgt_voc_size", type=int, default=32000,
                                  help="limit target vocabulary size to the n most frequent words")
    processing_group.add_argument("--use_voc", help="specify an exisiting vocabulary file")
#     parser.add_argument("--add_to_valid_set_every", type = int)
#     parser.add_argument("--shuffle", default = False, action = "store_true")
#     parser.add_argument("--enable_fast_shuffle", default = False, action = "store_true")
    processing_group.add_argument("--tgt_segmentation_type", choices=["word", "word2char", "char"], default="word")
    processing_group.add_argument("--src_segmentation_type", choices=["word", "word2char", "char"], default="word")

    processing_group.add_argument("--bpe_src", type=int, help="Apply BPE to source side, with this many merges")
    processing_group.add_argument("--bpe_tgt", type=int, help="Apply BPE to target side, with this many merges")

    processing_group.add_argument("--joint_bpe", type=int, help="Apply joint BPE, with this many merges")

    processing_group.add_argument("--latin_src", default=False, action="store_true", help="apply preprocessing for latin scripts to source")
    processing_group.add_argument("--latin_tgt", default=False, action="store_true", help="apply preprocessing for latin scripts to target")

    processing_group.add_argument("--latin_type", choices="all_adjoint caps_isolate".split(), default="all_adjoint", help="choose preprocessing for latin scripts to source")


class CommandLineValuesException(Exception):
    pass


def cmdline(arguments=None):
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Prepare training data.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    define_parser(parser)

    args = parser.parse_args(args=arguments)

    do_make_data(args)

# def get_parse_option_orderer():
#     por = argument_parsing_tools.ParseOptionRecorder()
#     define_parser(por)
#     return por


def get_parse_option_orderer():
    description_to_config_section = dict((v, k) for (k, v) in _CONFIG_SECTION_TO_DESCRIPTION.iteritems())
    por = argument_parsing_tools.ParseOptionRecorder(group_title_to_section=description_to_config_section,
                                                     ignore_positional_arguments=set(["save_prefix", "data_prefix"]))
    define_parser(por)
    return por


def make_data_config(args):
    parse_option_orderer = get_parse_option_orderer()
    config = parse_option_orderer.convert_args_to_ordered_dict(args)
    config.add_metadata_infos(1.0, overwrite=True)
    config.set_readonly()

    if config["data"]["src_fn"] is None or config["data"]["tgt_fn"] is None or config["data"]["save_prefix"] is None:
        raise CommandLineValuesException("src_fn, tgt_fn and save_prefix need to be set either on the command line or in a config file")

    if not ((config.data.test_src is None) == (config.data.test_tgt is None)):
        raise CommandLineValuesException(
            "Command Line Error: either specify both --test_src and --test_tgt or neither")

    if not ((config.data.dev_src is None) == (config.data.dev_tgt is None)):
        raise CommandLineValuesException("Command Line Error: either specify both --dev_src and --dev_tgt or neither")

    if config.processing.joint_bpe is not None and (config.processing.bpe_src is not None or config.processing.bpe_tgt is not None):
        raise CommandLineValuesException("Command Line Error: --joint_bpe is incompatible with --bpe_src and --bpe_tgt")
    return config


def load_config(filename):
    config = argument_parsing_tools.OrderedNamespace.load_from(filename)
    if "metadata" not in config:  # older config file
        parse_option_orderer = get_parse_option_orderer()
        config = parse_option_orderer.convert_args_to_ordered_dict(config, args_is_namespace=False)
    config.set_readonly()
    return config


def do_make_data(args):
    import nmt_chainer.dataprocessing.make_data
    config = make_data_config(args)
    nmt_chainer.dataprocessing.make_data.do_make_data(config)


if __name__ == '__main__':
    cmdline()
