
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

class ParseOptionRecorder(object):
    def __init__(self, name = None, group_title_to_section = None):
        self.name = name
        self.argument_list = []
        self.group_title_to_section = group_title_to_section
    
    def add_argument(self, argument_name, *args, **kwargs):
        if "dest" in kwargs:
            dest = kwargs["dest"]
            assert len(args) < 10
        elif len(args) >= 10:
            dest = args[9]
        elif argument_name.startswith("--"):
            dest = argument_name[2:]
        else:
            dest = argument_name
        self.argument_list.append(dest)
        
    def add_argument_group(self, title, desc = None):
        if self.group_title_to_section is not None:
            title = self.group_title_to_section[title]
        group = ParseOptionRecorder(title)
        self.argument_list.append(group)
        return group
        
    def convert_args_to_ordered_dict(self, args):
        result = OrderedNamespace()
        for arg in self.argument_list:
            if isinstance(arg, ParseOptionRecorder):
                if arg.name in result:
                    raise AssertionError
                result[arg.name] = arg.convert_args_to_ordered_dict(args)
            else:
                if arg in result:
                    raise AssertionError
                if arg in args:
                    result[arg] = getattr(args, arg)
        return result
    
# class ParserWithNoneDefaultAndNoGroup(object):
#     def __init__(self):
#         import argparse
#         self.parser = argparse.ArgumentParser()
#         
#     def add_argument(self, *args, **kwargs):
#         if len(args) >=5:
#             assert "default" not in kwargs
#             args[4] = None
#         elif "default" in kwargs:
#             kwargs["default"] = None
#         self.parser.add_argument(*args, **kwargs)
#         
#     def add_argument_group(self, *args, **kwargs):
#         return self.parser