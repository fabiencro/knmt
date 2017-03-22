""" Some utilities to leverage the argparse module and produce nicer config files."""

from collections import OrderedDict
import argparse
import json
import datetime
import sys
from nmt_chainer.utilities import versioning_tools


class OrderedNamespace(OrderedDict):
    """A class that act as a configuration dictionnary.
    Inherit from OrderedDict. When set to readonly mode, fields can be accessed by attributes:
        config["name"] == config.name
    """

    def __init__(self, *args, **kwargs):
        self.readonly = False
        super(OrderedNamespace, self).__init__(*args, **kwargs)

    def set_readonly(self):
        self.readonly = True
        for v in self.itervalues():
            if isinstance(v, OrderedNamespace):
                v.set_readonly()

    def __setitem__(self, *args, **kwargs):
        if self.readonly:
            raise KeyError("object in read-only mode")
        else:
            return OrderedDict.__setitem__(self, *args, **kwargs)

    def __getattr__(self, key):
        if self.readonly:
            try:
                return getattr(super(OrderedNamespace, self), key)
            except AttributeError:
                return self[key]
        else:
            return getattr(super(OrderedNamespace, self), key)

    def save_to(self, filename):
        json.dump(self, open(filename, "w"), indent=2, separators=(',', ': '))

    @classmethod
    def load_from(cls, filename):
        d = json.load(open(filename), object_pairs_hook=OrderedDict)
        cls.convert_to_ordered_namespace(d)
        return d

    @classmethod
    def convert_to_ordered_namespace(cls, ordered_dict):
        for v in ordered_dict.itervalues():
            if isinstance(v, OrderedDict):
                cls.convert_to_ordered_namespace(v)
        if not isinstance(ordered_dict, OrderedDict):
            raise ValueError()
        ordered_dict.__class__ = cls
        ordered_dict.readonly = False

    def update_recursive(self, other, valid_keys):
        if not isinstance(other, OrderedNamespace):
            raise ValueError()
        if self.readonly:
            raise ValueError()
        for key, val in other.iteritems():
            if isinstance(val, OrderedNamespace):
                if key in self:
                    if not (isinstance(self[key], OrderedNamespace)):
                        raise ValueError()
                else:
                    self[key] = OrderedNamespace()
                self[key].update_recursive(val, valid_keys)
            else:
                if key in self:
                    if (isinstance(self[key], OrderedNamespace)):
                        raise ValueError()
                if key in valid_keys:
                    self[key] = val

    def pretty_print(self, indent=4, discard_section=("metadata",)):
        for k, v in self.iteritems():
            if isinstance(v, OrderedNamespace) and k not in discard_section:
                print " " * indent, k, ":"
                v.pretty_print(indent=indent + 4, discard_section=())
            else:
                print " " * indent, k, v

    def copy(self, readonly=None):
        res = OrderedNamespace()
        for k, v in self.iteritems():
            if isinstance(v, OrderedNamespace):
                res[k] = v.copy(readonly)
            else:
                res[k] = v
        if readonly is None:
            res.readonly = self.readonly
        else:
            res.readonly = readonly
        return res

    def add_section(self, name, keep_at_bottom=None, overwrite=False):
        if self.readonly:
            raise ValueError()
        if name in self and not overwrite:
            return
        self[name] = OrderedNamespace()
        if keep_at_bottom is not None:
            metadata = self[keep_at_bottom]
            del self[keep_at_bottom]
            self[keep_at_bottom] = metadata

    def insert_section(self, name, content, even_if_readonly=False, keep_at_bottom=None, overwrite=False):
        if self.readonly and not even_if_readonly:
            raise ValueError()
        if name in self and not overwrite:
            raise ValueError()
        super(OrderedNamespace, self).__setitem__(name, content)
        if keep_at_bottom is not None:
            metadata = self[keep_at_bottom]
            del self[keep_at_bottom]
            super(OrderedNamespace, self).__setitem__(keep_at_bottom, metadata)

    def add_metadata_infos(self, version_num, overwrite=False):
        if self.readonly:
            raise ValueError()
        if "metadata" in self:
            if not overwrite:
                raise AssertionError()
        else:
            self["metadata"] = OrderedNamespace()

        self["metadata"]["config_version_num"] = version_num
        self["metadata"]["command_line"] = " ".join(sys.argv)
        self["metadata"]["knmt_version"] = versioning_tools.get_version_dict()
        self["metadata"]["creation_time"] = datetime.datetime.now().strftime("%I:%M%p %B %d, %Y")

    def set_metadata_modified_time(self):
        if self.readonly:
            raise ValueError()
        if "metadata" not in self:
            self["metadata"] = OrderedNamespace()
        self["metadata"]["modified_time"] = datetime.datetime.now().strftime("%I:%M%p %B %d, %Y")


class ParseOptionRecorder(object):
    """ A class whose main role is to remember the order in which argparse options have been defined, and to which subparser they belong.
    """

    def __init__(self, name=None, group_title_to_section=None, ignore_positional_arguments=set()):
        self.name = name
        self.argument_list = []
        self.group_title_to_section = group_title_to_section
        self.ignore_positional_arguments = ignore_positional_arguments

    def add_argument(self, argument_name, *args, **kwargs):
        positional = False
        if "dest" in kwargs:
            dest = kwargs["dest"]
            assert len(args) < 10
        elif len(args) >= 10:
            dest = args[9]
        elif argument_name.startswith("--"):
            dest = argument_name[2:]
        else:
            positional = True
            dest = argument_name
        if not (positional and dest in self.ignore_positional_arguments):
            self.argument_list.append(dest)

    def add_argument_group(self, title, desc=None):
        if self.group_title_to_section is not None:
            title = self.group_title_to_section[title]
        group = ParseOptionRecorder(title)
        self.argument_list.append(group)
        return group

    def convert_args_to_ordered_dict(self, args, args_is_namespace=True):
        result = OrderedNamespace()
        for arg in self.argument_list:
            if isinstance(arg, ParseOptionRecorder):
                if arg.name in result:
                    raise AssertionError
                result[arg.name] = arg.convert_args_to_ordered_dict(args, args_is_namespace=args_is_namespace)
            else:
                if arg in result:
                    raise AssertionError
                if arg in args:
                    if args_is_namespace:
                        result[arg] = getattr(args, arg)
                    else:
                        result[arg] = args[arg]
        return result


class ParserWithNoneDefaultAndNoGroup(object):
    """ A class whose main role is to help determine which arguments have been set on the command line."""

    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def add_argument(self, *args, **kwargs):
        #         print " add_arg ", args, kwargs
        if len(args) >= 5:
            #             print "changing default args", args, kwargs
            assert "default" not in kwargs
            args[4] = None
#             print " -> ", args, kwargs
        elif "default" in kwargs:
            #             print "changing default", args, kwargs
            kwargs["default"] = None
#             print " -> ", args, kwargs
        self.parser.add_argument(*args, **kwargs)

    def add_argument_group(self, *args, **kwargs):
        return self

    def get_args_given(self, arglist):
        args_given_set = set()
        args, remaining_args = self.parser.parse_known_args(arglist)
        for argname in args.__dict__:
            assert argname not in args_given_set
            if getattr(args, argname) is not None:
                args_given_set.add(argname)
        return args_given_set


class ArgumentActionNotOverwriteWithNone(argparse.Action):
    """An argparse action that will not overwrite a dest that is already set to a non-None value.
    Useful for options that are set by multiple arguments"""

    def __call__(self, parser, namespace, values, option_string=None):
        if self.dest in namespace and getattr(
                namespace, self.dest) is not None and values is None:
            return
        setattr(namespace, self.dest, values)
