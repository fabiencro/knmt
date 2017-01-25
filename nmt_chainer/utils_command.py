import argparse
import graph_training
import replace_tgt_unk

def define_parser(parser):
    subparsers = parser.add_subparsers(dest = "__sub_subcommand_name")
    
    graph_parser = subparsers.add_parser('graph', description= "Create a graph of the training data.", 
                           help = "Create a graph of the training data.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    graph_training.define_parser(graph_parser)
    
    replace_tgt_parser = subparsers.add_parser('replace_tgt_unk', description= "Replace UNK tags using a dictionnary.", 
                           help = "Replace UNK tags using a dictionnary.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    replace_tgt_unk.define_parser(replace_tgt_parser)
    
def do_utils(args):
    func = {"graph": graph_training.do_graph,
            "replace_tgt_unk": replace_tgt_unk.do_replace
            }[args.__sub_subcommand_name]
    func(args)
    