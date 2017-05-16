"""convert_config.py: Read an old config file and convert it to new format."""

import logging
import sys

from nmt_chainer.utilities import argument_parsing_tools
from nmt_chainer.training_module import train_config

logging.basicConfig()
log = logging.getLogger("rnns:train_config")
log.setLevel(logging.INFO)


def convert_config(old_config_fn, new_config_fn):
    config = argument_parsing_tools.OrderedNamespace.load_from(old_config_fn)
    if "metadata" not in config:  # older config file
        parse_option_orderer = train_config.get_parse_option_orderer()
        config_training = parse_option_orderer.convert_args_to_ordered_dict(config["command_line"], args_is_namespace=False)

        train_config.convert_cell_string(config_training, no_error=False)

        assert "data" not in config_training
        config_training["data"] = argument_parsing_tools.OrderedNamespace()
        config_training["data"]["data_fn"] = config["data"]
        config_training["data"]["Vi"] = config["Vi"]
        config_training["data"]["Vo"] = config["Vo"]
        config_training["data"]["voc"] = config["voc"]

        assert "metadata" not in config_training
        config_training["metadata"] = argument_parsing_tools.OrderedNamespace()
        config_training["metadata"]["config_version_num"] = 1.0
        config_training["metadata"]["command_line"] = None
        config_training["metadata"]["knmt_version"] = None
        config = config_training
    config.save_to(new_config_fn)
    print "Remember that the model_parameters section must be added manually."

if len(sys.argv) != 3:
    print "Usage: python nmt_chainer/utilities/convert_config.py config_filename converted_config_filename"
    exit(-1)

config_filename = sys.argv[1]
converted_config_filename = sys.argv[2]
convert_config(config_filename, converted_config_filename)
