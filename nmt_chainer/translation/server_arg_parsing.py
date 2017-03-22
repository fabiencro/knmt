"""server_arg_parsing.py: parse evaluation argument and create configuration dictionnary for server"""

from nmt_chainer.utilities import argument_parsing_tools

_CONFIG_SECTION_TO_DESCRIPTION = {"output": "Output Options",
                                  "process": "Translation Process Options"}


def define_parser(parser):
    parser.add_argument("training_config", nargs="?", help="prefix of the trained model",
                        action=argument_parsing_tools.ArgumentActionNotOverwriteWithNone)
    parser.add_argument("trained_model", nargs="?", help="prefix of the trained model",
                        action=argument_parsing_tools.ArgumentActionNotOverwriteWithNone)
    parser.add_argument("--host", help="host for listening request", default='localhost')
    parser.add_argument("--port", help="port for listening request", default=44666)
    parser.add_argument("--segmenter_command", help="command to communicate with the segmenter server")
    parser.add_argument("--segmenter_format", help="format to expect from the segmenter (parse_server, morph)", default='plain')

    output_group = parser.add_argument_group(_CONFIG_SECTION_TO_DESCRIPTION["output"])
    output_group.add_argument("--tgt_fn", help="target text")
    output_group.add_argument("--nbest_to_rescore", help="nbest list in moses format")
    output_group.add_argument("--ref", help="target text")
    output_group.add_argument("--tgt_unk_id", choices=["attn", "id"], default="align")
    # arguments for unk replace
    output_group.add_argument("--dic")

    management_group = parser.add_argument_group(_CONFIG_SECTION_TO_DESCRIPTION["process"])
    management_group.add_argument("--gpu", type=int, help="specify gpu number to use, if any")
    management_group.add_argument("--additional_training_config", nargs="*", help="prefix of the trained model")
    management_group.add_argument("--additional_trained_model", nargs="*", help="prefix of the trained model")
    management_group.add_argument("--max_nb_ex", type=int, help="only use the first MAX_NB_EX examples")
    management_group.add_argument("--mb_size", type=int, default=80, help="Minibatch size")
    management_group.add_argument("--nb_batch_to_sort", type=int, default=20, help="Sort this many batches by size.")
    management_group.add_argument("--reverse_training_config", help="prefix of the trained model")
    management_group.add_argument("--reverse_trained_model", help="prefix of the trained model")


def do_start_server(args):
    import nmt_chainer.translation.server
    nmt_chainer.translation.server.do_start_server(args)


def get_parse_option_orderer():
    description_to_config_section = dict((v, k) for (k, v) in _CONFIG_SECTION_TO_DESCRIPTION.iteritems())
    por = argument_parsing_tools.ParseOptionRecorder(group_title_to_section=description_to_config_section)
    define_parser(por)
    return por


def make_config_server(args):
    parse_option_orderer = get_parse_option_orderer()
    config_eval = parse_option_orderer.convert_args_to_ordered_dict(args)
    config_eval.add_metadata_infos(version_num=1)
    config_eval.set_readonly()

    return config_eval
