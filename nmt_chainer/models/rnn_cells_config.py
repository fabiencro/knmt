from collections import OrderedDict
import nmt_chainer.utilities.argument_parsing_tools

cell_description_keywords = OrderedDict([
    ("cell_type", str),
    ("sub_cell_type", str),
    ("nb_stacks", int),
    ("dropout", float),
    ("residual_connection", int),
    ("no_dropout_on_input", int),
    ("no_residual_connection_on_output", int),
    ("no_residual_connection_on_input", int),
    ("init_type", str),
    ("inner_init_type", str),
    ("lateral_init_type", str),
    ("upward_init_type", str),
    ("bias_init_type", str),
    ("forget_bias_init_type", str),
    ("init_scale", float),
    ("inner_init_scale", float),
    ("lateral_init_scale", float),
    ("upward_init_scale", float),
    ("bias_init_scale", float),
    ("forget_bias_init_scale", float),
    ("init_fillvalue", float),
    ("inner_init_fillvalue", float),
    ("lateral_init_fillvalue", float),
    ("upward_init_fillvalue", float),
    ("bias_init_fillvalue", float),
    ("forget_bias_init_fillvalue", float)
])


def create_cell_config_from_string(model_str):
    components = model_str.split(",")
    if ":" not in components[0]:
        components = ["cell_type:%s" % components[0]] + components[1:]
    keywords = {}
    for comp in components:
        assert ":" in comp
        key, val = comp.split(":")
        if key in cell_description_keywords:
            keywords[key] = cell_description_keywords[key](val)
        else:
            raise ValueError("bad cell parameter: %s (possible parameters: %s)" %
                             (comp, " ".join(cell_description_keywords.keys())))

    ordered_keywords = OrderedDict()
    for key in cell_description_keywords.keys():
        if key in keywords:
            ordered_keywords[key] = keywords[key]
    nmt_chainer.utilities.argument_parsing_tools.OrderedNamespace.convert_to_ordered_namespace(
        ordered_keywords)
#     ordered_keywords.set_readonly()
    return ordered_keywords
