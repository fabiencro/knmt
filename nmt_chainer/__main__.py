#!/usr/bin/env python -O
# PYTHON_ARGCOMPLETE_OK

import argparse
import nmt_chainer.training_module.train_config as train
import nmt_chainer.translation.eval_config as eval_module
import nmt_chainer.dataprocessing.make_data_conf as make_data

import nmt_chainer.utilities.utils_command as utils_command

from nmt_chainer.utilities import versioning_tools
import sys

try:
    import argcomplete
except ImportError:
    argcomplete = None


def run_in_pdb(func, args):
    import pdb as pdb_module
    import sys
    import traceback
    pdb = pdb_module.Pdb()
    while True:
        try:
            pdb.runcall(func, args)
            if pdb._user_requested_quit:
                break
            print "The program finished and will be restarted"
        except pdb_module.Restart:
            print "Restarting with arguments:"
            print "\t" + " ".join(sys.argv[1:])
        except SystemExit:
            # In most cases SystemExit does not warrant a post-mortem session.
            print "The program exited via sys.exit(). Exit status: ",
            print sys.exc_info()[1]
        except SyntaxError:
            traceback.print_exc()
            sys.exit(1)
        except BaseException:
            traceback.print_exc()
            print "Uncaught exception. Entering post mortem debugging"
            print "Running 'cont' or 'step' will restart the program"
            t = sys.exc_info()[2]
            pdb.interaction(None, t)
            print "Post mortem debugger finished. The program will be restarted"


def main(arguments=None):
    # create the top-level parser
    parser = argparse.ArgumentParser(description="Kyoto-NMT: an Implementation of the RNNSearch model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     prog="knmt")

    parser.add_argument("--run_in_pdb", default=False, action="store_true", help="run knmt in pdb (python debugger)")

    subparsers = parser.add_subparsers(dest="__subcommand_name")

    # create the parser for the "make_data" command
    parser_make_data = subparsers.add_parser('make_data', description="Prepare data for training.", help="Prepare data for training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    make_data.define_parser(parser_make_data)

    # create the parser for the "train" command
    parser_train = subparsers.add_parser('train', description="Train a model.", help="Train a model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train.define_parser(parser_train)

    # create the parser for the "eval" command
    parser_eval = subparsers.add_parser('eval', description="Use a model.", help="Use a model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    eval_module.define_parser(parser_eval)

    # create the parser for the "version" command
    parser_version = subparsers.add_parser('version', description="Get version infos.", help="Get version infos", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser_utils = subparsers.add_parser('utils', description="Call a utility script.", help="Call a utility script", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    utils_command.define_parser(parser_utils)

#     import argcomplete
    if argcomplete is not None:
        argcomplete.autocomplete(parser)

    args = parser.parse_args(args=arguments)
    if arguments is not None:
        args.__original_argument_list = arguments
    else:
        args.__original_argument_list = sys.argv

    func = {"make_data": make_data.do_make_data,
            "train": train.do_train,
            "eval": eval_module.do_eval,
            "version": versioning_tools.main,
            "utils": utils_command.do_utils}[args.__subcommand_name]

    if args.run_in_pdb:
        run_in_pdb(func, args)
#         import pdb
#         pdb.runcall(func, args)
    else:
        try:
            func(args)
        except train.CommandLineValuesException as e:
            parser_train.error(e.args[0])
        except eval_module.CommandLineValuesException as e:
            parser_eval.error(e.args[0])
        except make_data.CommandLineValuesException as e:
            parser_make_data.error(e.args[0])


if __name__ == "__main__":
    main()
