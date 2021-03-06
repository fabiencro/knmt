"""eval_config.py: parse evaluation argument and create configuration dictionnary"""
from __future__ import absolute_import, division, print_function, unicode_literals
from nmt_chainer.utilities import argument_parsing_tools
import six

_CONFIG_SECTION_TO_DESCRIPTION = {"method": "Translation Method",
                                  "output": "Output Options",
                                  "process": "Translation Process Options"}

# def add_arguments_from_dataclass(parser, dataclass):
#     for key, field in dataclass.__dataclass_fields__:
#         assert field.default != dataclasses._MISSING_TYPE
#         parser.add_argument("--"+key, type = field.type, default = field.default)

def define_parser(parser):
    parser.add_argument("training_config", nargs="?", help="prefix of the trained model",
                        action=argument_parsing_tools.ArgumentActionNotOverwriteWithNone)
    parser.add_argument("trained_model", nargs="?", help="prefix of the trained model",
                        action=argument_parsing_tools.ArgumentActionNotOverwriteWithNone)
    parser.add_argument("src_fn", nargs="?", help="source text",
                        action=argument_parsing_tools.ArgumentActionNotOverwriteWithNone)
    parser.add_argument("dest_fn", nargs="?", help="destination file",
                        action=argument_parsing_tools.ArgumentActionNotOverwriteWithNone)

    translation_method_group = parser.add_argument_group(_CONFIG_SECTION_TO_DESCRIPTION["method"])
    translation_method_group.add_argument("--mode", default="translate",
                                          choices=["translate", "align", "translate_attn", "beam_search", "eval_bleu",
                                                   "score_nbest", "astar_search", "astar_eval_bleu"], help="target text")
    translation_method_group.add_argument("--beam_width", type=int, default=20, help="beam width")
    translation_method_group.add_argument("--beam_pruning_margin", type=float, default=None, help="beam pruning margin")
    parser.add_argument("--beam_score_coverage_penalty", choices=['none', 'google'], default='none')
    parser.add_argument("--beam_score_coverage_penalty_strength", type=float, default=0.2)
    translation_method_group.add_argument("--nb_steps", type=int, default=50, help="nb_steps used in generation")
    translation_method_group.add_argument("--nb_steps_ratio", type=float, help="nb_steps used in generation as a ratio of input length")
#     translation_method_group.add_argument("--beam_opt", default = False, action = "store_true")
    translation_method_group.add_argument("--groundhog", default=False, action="store_true")
    translation_method_group.add_argument("--force_finish", default=False, action="store_true")
    translation_method_group.add_argument("--beam_score_length_normalization", choices=['none', 'simple', 'google'], default='none')
    translation_method_group.add_argument("--beam_score_length_normalization_strength", type=float, default=0.2)
    translation_method_group.add_argument("--post_score_length_normalization", choices=['none', 'simple', 'google'], default='simple')
    translation_method_group.add_argument("--post_score_length_normalization_strength", type=float, default=0.2)
    translation_method_group.add_argument("--post_score_coverage_penalty", choices=['none', 'google'], default='none')
    translation_method_group.add_argument("--post_score_coverage_penalty_strength", type=float, default=0.2)
    translation_method_group.add_argument("--prob_space_combination", default=False, action="store_true")

    translation_method_group.add_argument("--astar_batch_size", type=int, default=32)
    translation_method_group.add_argument("--astar_max_queue_size", type=int, default=1000)
    translation_method_group.add_argument("--astar_prune_margin", type=float)
    translation_method_group.add_argument("--astar_prune_ratio", type=float)
    translation_method_group.add_argument("--astar_length_normalization_exponent", type=float, default=1)
    translation_method_group.add_argument("--astar_length_normalization_constant", type=float, default=0)
    translation_method_group.add_argument("--astar_priority_eval_string")
    translation_method_group.add_argument("--astar_max_length_diff", type=float)
    

    translation_method_group.add_argument("--always_consider_eos_and_placeholders", default=False, action="store_true",
                        help="consider the possibility of outputing eos and placeholders at each steps during beam search. Deprecated. True by default.")
    

    output_group = parser.add_argument_group(_CONFIG_SECTION_TO_DESCRIPTION["output"])
    output_group.add_argument("--tgt_fn", help="target text")
    output_group.add_argument("--nbest_to_rescore", help="nbest list in moses format")
    output_group.add_argument("--nbest", help="list of nbest translations instead of best translation", type=int, default=None)
    output_group.add_argument("--ref", help="target text")
    output_group.add_argument("--tgt_unk_id", choices=["align", "id"], default="align")
    output_group.add_argument("--generate_attention_html", help="generate a html file with attention information")
    output_group.add_argument("--rich_output_filename", help="generate a JSON file with attention information")
    # arguments for unk replace
    output_group.add_argument("--dic")
    output_group.add_argument("--remove_unk", default=False, action="store_true")
    output_group.add_argument("--normalize_unicode_unk", default=False, action="store_true")
    output_group.add_argument("--attempt_to_relocate_unk_source", default=False, action="store_true")
    output_group.add_argument("--log_config", nargs="?", help="Log configuration file.")

    output_group.add_argument("--no_attention_map_in_server_mode", default=False, action="store_true")

    management_group = parser.add_argument_group(_CONFIG_SECTION_TO_DESCRIPTION["process"])
    management_group.add_argument("--gpu", type=int, help="specify gpu number to use, if any")
    management_group.add_argument("--max_nb_ex", type=int, help="only use the first MAX_NB_EX examples")
    management_group.add_argument("--mb_size", type=int, default=80, help="Minibatch size")
    management_group.add_argument("--nb_batch_to_sort", type=int, default=20, help="Sort this many batches by size.")
    management_group.add_argument("--load_model_config", nargs="+", help="gives a list of models to be used for translation")
    management_group.add_argument("--src_fn", nargs="?", help="source text",
                                  action=argument_parsing_tools.ArgumentActionNotOverwriteWithNone)
    management_group.add_argument("--dest_fn", nargs="?", help="destination file",
                                  action=argument_parsing_tools.ArgumentActionNotOverwriteWithNone)
    management_group.add_argument("--additional_training_config", nargs="*", help="prefix of the trained model")
    management_group.add_argument("--additional_trained_model", nargs="*", help="prefix of the trained model")
    management_group.add_argument("--reverse_training_config", help="prefix of the trained model")
    management_group.add_argument("--reverse_trained_model", help="prefix of the trained model")
#     management_group.add_argument("--config", help = "load eval config file")
    management_group.add_argument("--server", help="host:port for listening request")
    management_group.add_argument("--multiserver_config", help="config file for multiserver")
    management_group.add_argument("--segmenter_command", help="command to communicate with the segmenter server")
    management_group.add_argument("--segmenter_format", help="format to expect from the segmenter (parse_server, morph)", default='plain')
    management_group.add_argument("--description", help="Optional message to be stored in the configuration file")
    management_group.add_argument("--pp_command", help="command to call on the translation before sending the response to the client.")
    management_group.add_argument("--force_placeholders", default=False, action="store_true", 
                help="force the generation of translations with placeholders")
    management_group.add_argument("--units_placeholders", default=False, action="store_true", help="Use in conjunction with --force_placeholders if each placeholder is a subword unit in the vocabulary (deprecated, now True by default)")
    management_group.add_argument("--use_chainerx", default=False, action="store_true", 
                                    help="Use the chainerx library instead of chainer. Can provide speed improvement (if chainerx is installed)")


    management_group.add_argument("--bilingual_dic_for_reranking", help="Dictionary file in TSV format. Will be used to create additional constraints during beam search.")
    management_group.add_argument("--invert_bilingual_dic_for_reranking", default=False, action="store_true")

    management_group.add_argument("--do_hyper_param_search", nargs=3)

class CommandLineValuesException(Exception):
    pass


def get_parse_option_orderer():
    description_to_config_section = dict((v, k) for (k, v) in six.iteritems(_CONFIG_SECTION_TO_DESCRIPTION))
    por = argument_parsing_tools.ParseOptionRecorder(group_title_to_section=description_to_config_section,
                                                     # ignore_positional_arguments = set(["src_fn", "dest_fn"])
                                                     )
    define_parser(por)
    return por


def make_config_eval(args):
    parse_option_orderer = get_parse_option_orderer()
    config_eval = parse_option_orderer.convert_args_to_ordered_dict(args)
    config_eval.add_metadata_infos(version_num=1)
    config_eval.set_readonly()

    if config_eval.process.server is None and config_eval.process.multiserver_config is None:
        if config_eval.process.dest_fn is None:
            raise CommandLineValuesException("dest_fn need to be set either on the command line or in a config file")

    return config_eval


def load_config_eval(filename, readonly=True):
    config = argument_parsing_tools.OrderedNamespace.load_from(filename)
    if "metadata" not in config:  # older config file
        parse_option_orderer = get_parse_option_orderer()
        config_eval = parse_option_orderer.convert_args_to_ordered_dict(config, args_is_namespace=False)
        assert "metadata" not in config_eval
        config_eval["metadata"] = argument_parsing_tools.OrderedNamespace()
        config_eval["metadata"]["config_version_num"] = 0.9
        config_eval["metadata"]["command_line"] = None
        config_eval["metadata"]["knmt_version"] = None
        config = config_eval
    elif config["metadata"]["config_version_num"] != 1.0:
        raise ValueError("The config version of %s is not supported by this version of the program" % filename)
    if readonly:
        config.set_readonly()
    return config


def command_line(arguments=None):

    import argparse
    parser = argparse.ArgumentParser(description="Use a RNNSearch model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    define_parser(parser)

    args = parser.parse_args(args=arguments)

    do_eval(args)


def do_eval(args):
    import nmt_chainer.translation.eval
    config_eval = make_config_eval(args)

    if config_eval.training_config is not None:
        if config_eval.trained_model is None:
            raise CommandLineValuesException(
                "If specifying a model via the training_config argument, you also need to specify the trained_model argument")
    else:
        if config_eval.process.multiserver_config is None and config_eval.process.load_model_config is None:
            raise CommandLineValuesException(
                "You need to specify either the training_config positional argument, or the load_model_config option, or both")

    nmt_chainer.translation.eval.do_eval(config_eval)


if __name__ == '__main__':
    command_line()
