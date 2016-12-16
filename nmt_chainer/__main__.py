#!/usr/bin/env python -O

import argparse
import nmt_chainer.train as train
import nmt_chainer.eval as eval_module
import nmt_chainer.make_data as make_data

def main():
    # create the top-level parser
    parser = argparse.ArgumentParser(description = "Kyoto-NMT: an Implementation of the RNNSearch model", 
                                         formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers()
    
    # create the parser for the "make_data" command
    parser_make_data = subparsers.add_parser('make_data', description= "Prepare data for training.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    make_data.define_parser(parser_make_data)
    parser_make_data.set_defaults(func_str = "make_data")
    
    # create the parser for the "train" command
    parser_train = subparsers.add_parser('train', description= "Train a model.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train.define_parser(parser_train)
    parser_train.set_defaults(func_str = "train")
    
    # create the parser for the "eval" command
    parser_eval = subparsers.add_parser('eval', description= "Use a model.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    eval_module.define_parser(parser_eval)
    parser_eval.set_defaults(func_str = "eval")
    
    args = parser.parse_args()
    
    func = {"make_data": make_data.do_make_data, "train": train.do_train, "eval":eval_module.do_eval}[args.func_str]
    func(args)
    
if __name__ == "__main__":
    main()