"""train_config.py: Parse training arguments and create config dictionnary."""

import logging
import sys

from nmt_chainer.utilities import argument_parsing_tools

logging.basicConfig()
log = logging.getLogger("rnns:train_config")
log.setLevel(logging.INFO)

_CONFIG_SECTION_TO_DESCRIPTION = {"model": "Model Description",
                                  "training": "Training Parameters",
                                  "training_management": "Training Management and Monitoring"}


def define_parser(parser):
    parser.add_argument("data_prefix", nargs="?",
                        action=argument_parsing_tools.ArgumentActionNotOverwriteWithNone,
                        help="prefix of the training data created by make_data.py")
    parser.add_argument("save_prefix", nargs="?",
                        action=argument_parsing_tools.ArgumentActionNotOverwriteWithNone,
                        help="prefix to be added to all files created during the training")

    model_description_group = parser.add_argument_group(_CONFIG_SECTION_TO_DESCRIPTION["model"])
    model_description_group.add_argument("--Ei", type=int, default=640, help="Source words embedding size.")
    model_description_group.add_argument("--Eo", type=int, default=640, help="Target words embedding size.")
    model_description_group.add_argument("--Hi", type=int, default=1024, help="Source encoding layer size.")
    model_description_group.add_argument("--Ho", type=int, default=1024, help="Target hidden layer size.")
    model_description_group.add_argument("--Ha", type=int, default=1024, help="Attention Module Hidden layer size.")
    model_description_group.add_argument("--Hl", type=int, default=512, help="Maxout output size.")
    model_description_group.add_argument("--encoder_cell_type", default="lstm", help="cell type of encoder. format: type,param1:val1,param2:val2,...")  # where type is in [%s]"%(" ".join(rnn_cells.cell_dict.keys())))
    model_description_group.add_argument("--decoder_cell_type", default="lstm", help="cell type of decoder. format same as for encoder")
    model_description_group.add_argument("--lexical_probability_dictionary", help="lexical translation probabilities in zipped JSON format. Used to implement https://arxiv.org/abs/1606.02006")
    model_description_group.add_argument("--lexicon_prob_epsilon", default=1e-3, type=float, help="epsilon value for combining the lexical probabilities")
    model_description_group.add_argument("--use_deep_attn", default=False, action="store_true")
    model_description_group.add_argument("--use_accumulated_attn", default=False, action="store_true")
    model_description_group.add_argument("--init_orth", default=False, action="store_true")
    model_description_group.add_argument("--use_bn_length", default=0, type=int)
    model_description_group.add_argument("--use_goto_attention", default=False, action="store_true")
    
    model_description_group.add_argument("--use_ff_model", default=False, action="store_true")
    model_description_group.add_argument("--ff_d_model", type=int, default=512, help="FF model d_model")
    model_description_group.add_argument("--ff_n_heads", type=int, default=8, help="FF model number of attention heads")
    model_description_group.add_argument("--ff_nb_layers_src", type=int, default=6, help="FF model number of source layers")
    model_description_group.add_argument("--ff_nb_layers_tgt", type=int, default=6, help="FF model number of target layers")
    model_description_group.add_argument("--ff_dropout", type=float, help="FF model dropout")
    model_description_group.add_argument("--ff_d_ff", type=int, default=2048, help="FF model d_ff")
    model_description_group.add_argument("--ff_use_exp_relu", default=False, action="store_true")
    model_description_group.add_argument("--ff_residual_mode", default="normal", choices="normal none after".split())
    model_description_group.add_argument("--ff_no_normalize", default=False, action="store_true")
    model_description_group.add_argument("--use_own_layer_normalization", default=False, action="store_true")
    
    training_paramenters_group = parser.add_argument_group(_CONFIG_SECTION_TO_DESCRIPTION["training"])
    training_paramenters_group.add_argument("--mb_size", type=int, default=64, help="Minibatch size")
    training_paramenters_group.add_argument("--nb_batch_to_sort", type=int, default=20, help="Sort this many batches by size.")
    training_paramenters_group.add_argument("--noise_on_prev_word", default=False, action="store_true")
    training_paramenters_group.add_argument("--l2_gradient_clipping", type=float, default=1, help="L2 gradient clipping. 0 for None")
    training_paramenters_group.add_argument("--hard_gradient_clipping", type=float, nargs=2, help="hard gradient clipping.")
    training_paramenters_group.add_argument("--weight_decay", type=float, help="Weight decay value. ")
    training_paramenters_group.add_argument("--optimizer", choices=["sgd", "rmsprop", "rmspropgraves",
                                                                    "momentum", "nesterov", "adam", "scheduled_adam", "adagrad", "adadelta"],
                                            default="adam", help="Optimizer type.")
    training_paramenters_group.add_argument("--learning_rate", type=float, default=0.01, help="Learning Rate")
    training_paramenters_group.add_argument("--momentum", type=float, default=0.9, help="Momentum term")
    training_paramenters_group.add_argument("--randomized_data", default=False, action="store_true")
    training_paramenters_group.add_argument("--no_shuffle_of_training_data", default=False, action="store_true")
    training_paramenters_group.add_argument("--use_reinf", default=False, action="store_true")
    training_paramenters_group.add_argument("--use_previous_prediction", default=0, type=float)
    training_paramenters_group.add_argument("--curiculum_training", default=False, action="store_true")
    training_paramenters_group.add_argument("--reverse_src", default=False, action="store_true")
    training_paramenters_group.add_argument("--reverse_tgt", default=False, action="store_true")
    
    training_paramenters_group.add_argument("--use_soft_prediction_feedback", default=False, action="store_true")
    training_paramenters_group.add_argument("--use_gumbel_for_soft_predictions", default=False, action="store_true")
    training_paramenters_group.add_argument("--temperature_for_soft_predictions", type=float, default=1.0)


    training_paramenters_group.add_argument("--dynamic_batching", default=False, action="store_true")
    training_paramenters_group.add_argument("--dynamic_batching_max_elems", type=int, default=10000)
    training_paramenters_group.add_argument("--dynamic_batching_nb_sent_to_sort", type=int, default=5000)
    
    training_paramenters_group.add_argument("--load_initial_source_embeddings")
    training_paramenters_group.add_argument("--load_initial_target_embeddings")

    training_monitoring_group = parser.add_argument_group(_CONFIG_SECTION_TO_DESCRIPTION["training_management"])
    training_monitoring_group.add_argument("--config", help="load a training config file")
    training_monitoring_group.add_argument("--data_prefix", dest="data_prefix",
                                           action=argument_parsing_tools.ArgumentActionNotOverwriteWithNone,
                                           help="same as positional argument --data_prefix")
    training_monitoring_group.add_argument("--save_prefix", dest="save_prefix",
                                           action=argument_parsing_tools.ArgumentActionNotOverwriteWithNone,
                                           help="same as positional argument --save_prefix")
    training_monitoring_group.add_argument("--gpu", type=int, help="specify gpu number to use, if any")
    training_monitoring_group.add_argument("--load_model", help="load the parameters of a previously trained model")
    training_monitoring_group.add_argument("--load_optimizer_state", help="load previously saved optimizer states")
    training_monitoring_group.add_argument("--load_trainer_snapshot", help="load previously saved trainer states")
    training_monitoring_group.add_argument("--use_memory_optimization", default=False, action="store_true",
                                           help="Experimental option that could strongly reduce memory used.")
    training_monitoring_group.add_argument("--max_nb_iters", type=int, default=None, help="maximum number of iterations")
    training_monitoring_group.add_argument("--max_nb_epochs", type=int, default=None, help="maximum number of epochs")
    training_monitoring_group.add_argument("--max_src_tgt_length", type=int, help="Limit length of training sentences")
    training_monitoring_group.add_argument("--report_every", type=int, default=200, help="report every x iterations")
    training_monitoring_group.add_argument("--no_resume", default=False, action="store_true")
    training_monitoring_group.add_argument("--no_report_or_save", default=False, action="store_true")
    training_monitoring_group.add_argument("--sample_every", default=200, type=int)
    training_monitoring_group.add_argument("--save_ckpt_every", default=4000, type=int)
    training_monitoring_group.add_argument("--save_initial_model_to", help="save the initial model parameters to given file in npz format")
    training_monitoring_group.add_argument("--reshuffle_every_epoch", default=False, action="store_true", help="reshuffle training data at the end of each epoch")
    training_monitoring_group.add_argument("--resume", default=False, action="store_true", help="resume training from checkpoint config")
    training_monitoring_group.add_argument("--timer_hook", default=False, action="store_true", help="activate timer hook for profiling")
    training_monitoring_group.add_argument("--force_overwrite", default=False, action="store_true", help="Do not ask before overwiting existing files")
    training_monitoring_group.add_argument("--description", help="Optional message to be stored in the configuration file")

    training_monitoring_group.add_argument("--set_false_in_config", nargs="*", help="Forcing some options to be false")
    
    training_monitoring_group.add_argument("--update_old_config_file_with_default_values", 
                                           default=False, action="store_true", help="When using older config files")

    training_monitoring_group.add_argument("--generate_computation_graph", help="will generate computation graph of the first loss computed")

    training_monitoring_group.add_argument("--disable_cudnn_softmax", default=False, action="store_true")

class CommandLineValuesException(Exception):
    pass

#
# def load_training_config_file(filename):
#     file_content = json.load(open(filename))


def get_parse_option_orderer():
    description_to_config_section = dict((v, k) for (k, v) in _CONFIG_SECTION_TO_DESCRIPTION.iteritems())
    por = argument_parsing_tools.ParseOptionRecorder(group_title_to_section=description_to_config_section,
                                                     ignore_positional_arguments=set(["save_prefix", "data_prefix"]))
    define_parser(por)
    return por


def convert_cell_string(config_training, no_error=False):
    import nmt_chainer.models.rnn_cells_config

    try:
        if "encoder_cell_type" in config_training["model"] and config_training["model"]["encoder_cell_type"] is not None:
            config_training["model"]["encoder_cell_type"] = nmt_chainer.models.rnn_cells_config.create_cell_config_from_string(
                config_training["model"]["encoder_cell_type"])

        if "decoder_cell_type" in config_training["model"] and config_training["model"]["decoder_cell_type"] is not None:
            config_training["model"]["decoder_cell_type"] = nmt_chainer.models.rnn_cells_config.create_cell_config_from_string(
                config_training["model"]["decoder_cell_type"])
    except BaseException:
        if not no_error:
            raise


def load_config_train(filename, readonly=True, no_error=False):
    config = argument_parsing_tools.OrderedNamespace.load_from(filename)
    if "metadata" not in config:  # older config file
        parse_option_orderer = get_parse_option_orderer()
        config_training = parse_option_orderer.convert_args_to_ordered_dict(config["command_line"], args_is_namespace=False)

        convert_cell_string(config_training, no_error=no_error)

        assert "data" not in config_training
        config_training["data"] = argument_parsing_tools.OrderedNamespace()
        config_training["data"]["data_fn"] = config["data"]
        config_training["data"]["Vi"] = config["Vi"]
        config_training["data"]["Vo"] = config["Vo"]
        config_training["data"]["voc"] = config["voc"]

        assert "metadata" not in config_training
        config_training["metadata"] = argument_parsing_tools.OrderedNamespace()
        config_training["metadata"]["config_version_num"] = 0.9
        config_training["metadata"]["command_line"] = None
        config_training["metadata"]["knmt_version"] = None
        config = config_training
    elif config["metadata"]["config_version_num"] != 1.0:
        raise ValueError("The config version of %s is not supported by this version of the program" % filename)

    # Compatibility with intermediate verions of config file
    if "data_prefix" in config and "data_prefix" not in config["training_management"]:
        config["training_management"]["data_prefix"] = config["data_prefix"]
        del config["data_prefix"]

    if "train_prefix" in config and "train_prefix" not in config["training_management"]:
        config["training_management"]["train_prefix"] = config["train_prefix"]
        del config["train_prefix"]

    if readonly:
        config.set_readonly()
    return config


def find_which_command_line_arguments_were_given(argument_list):
    pwndan = argument_parsing_tools.ParserWithNoneDefaultAndNoGroup()
    define_parser(pwndan)
    args_given_set = pwndan.get_args_given(argument_list)
    return args_given_set


def make_config_from_args(args, readonly=True):
    config_base = None
    if args.config is not None:
        log.info("loading training config file %s", args.config)
        config_base = load_config_train(args.config, readonly=False)

        if args.set_false_in_config is not None:
            for option_name in args.set_false_in_config:
                path_option = option_name.split(".")
                last_dict = config_base
                for level in range(len(path_option) -1):
                    last_dict = config_base[path_option[level]]
                last_dict[path_option[-1]] = False
            

    parse_option_orderer = get_parse_option_orderer()
    config_training = parse_option_orderer.convert_args_to_ordered_dict(args)

    convert_cell_string(config_training)

    if config_base is not None:
        args_given_set = find_which_command_line_arguments_were_given(
            args.__original_argument_list)
        for argname in set(args_given_set):
            if getattr(args, argname) is None:
                args_given_set.remove(argname)

        print "args_given_set", args_given_set
        config_base.update_recursive(config_training, valid_keys=args_given_set, add_absent_keys=args.update_old_config_file_with_default_values)
        config_training = config_base
    else:
        assert "data" not in config_training
        assert "metadata" not in config_training

#     config_data_fn = config_training["data_prefix"] + ".data.config"

    if config_training["training_management"]["data_prefix"] is None or config_training["training_management"]["save_prefix"] is None:
        raise CommandLineValuesException("save_prefix and data_prefix need to be set either on the command line or in a config file")

    config_training.add_metadata_infos(version_num=1, overwrite=args.config is not None)

    if readonly:
        config_training.set_readonly()

    return config_training


# def load_config_train(filename, readonly = True):
#     config_as_ordered_dict = json.load(open(filename), object_pairs_hook=OrderedDict)
#
#     config = OrderedNamespace.load_from(filename)
#     if "metadata" not in config_as_ordered_dict: # older config file
#         parse_option_orderer = get_parse_option_orderer()
#         config_training = parse_option_orderer.convert_args_to_ordered_dict(config_as_ordered_dict["command_line"], args_is_namespace = False)
#
#         assert "data" not in config_training
#         config_training["data"] = argument_parsing_tools.OrderedNamespace()
#         config_training["data"]["data_fn"] = config_as_ordered_dict["data"]
#         config_training["data"]["Vi"] = config_as_ordered_dict["Vi"]
#         config_training["data"]["Vo"] = config_as_ordered_dict["Vo"]
#         config_training["data"]["voc"] = config_as_ordered_dict["voc"]
#
#         assert "metadata" not in config_training
#         config_training["metadata"] = argument_parsing_tools.OrderedNamespace()
#         config_training["metadata"]["config_version_num"] = 0.9
#         config_training["metadata"]["command_line"] = None
#         config_training["metadata"]["knmt_version"] = None
#     elif config_as_ordered_dict["metadata"]["config_version_num"] == 1.0:
#         argument_parsing_tools.OrderedNamespace.convert_to_ordered_namespace(config_as_ordered_dict)
#         config_training = config_as_ordered_dict
#     else:
#         raise ValueError("The config version of %s is not supported by this version of the program" % filename)
#
#     if readonly:
#         config_training.set_readonly()
#     return config_training

def command_line(arguments=None):
    import argparse
    parser = argparse.ArgumentParser(description="Train a RNNSearch model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    define_parser(parser)
    args = parser.parse_args(args=arguments)

    do_train(args)


def do_train(args):
    import nmt_chainer.training_module.train
    config = make_config_from_args(args, readonly=False)
    nmt_chainer.training_module.train.do_train(config)


if __name__ == '__main__':
    command_line()
