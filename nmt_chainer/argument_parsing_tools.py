
from collections import OrderedDict
import argparse

class OrderedNamespace(OrderedDict):
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
                    
    def pretty_print(self, indent = 4, discard_section = ("metadata",)):
        for k, v in self.iteritems():
            if isinstance(v, OrderedNamespace) and k not in discard_section:
                print " " * indent, k, ":"
                v.pretty_print(indent = indent + 4, discard_section = ())
            else:
                print " " * indent, k, v
                
    def copy(self, readonly = None):
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
    
    def add_section(self, name, keep_at_bottom = None):
        if self.readonly:
            raise ValueError()
        self[name] = OrderedNamespace()
        if keep_at_bottom is not None:
            metadata = self[keep_at_bottom]
            del self[keep_at_bottom]
            self[keep_at_bottom] = metadata
                                 
class ParseOptionRecorder(object):
    def __init__(self, name = None, group_title_to_section = None, ignore_positional_arguments = set()):
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
        
    def add_argument_group(self, title, desc = None):
        if self.group_title_to_section is not None:
            title = self.group_title_to_section[title]
        group = ParseOptionRecorder(title)
        self.argument_list.append(group)
        return group
        
    def convert_args_to_ordered_dict(self, args, args_is_namespace = True):
        result = OrderedNamespace()
        for arg in self.argument_list:
            if isinstance(arg, ParseOptionRecorder):
                if arg.name in result:
                    raise AssertionError
                result[arg.name] = arg.convert_args_to_ordered_dict(args, args_is_namespace = args_is_namespace)
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
    def __init__(self):
        self.parser = argparse.ArgumentParser()
         
    def add_argument(self, *args, **kwargs):
#         print " add_arg ", args, kwargs
        if len(args) >=5:
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

        