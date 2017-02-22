#def define_parser(parser):
#    parser.add_argument("training_config", help = "prefix of the trained model")
#    parser.add_argument("trained_model", help = "prefix of the trained model")
#    
#    parser.add_argument("--additional_training_config", nargs = "*", help = "prefix of the trained model")
#    parser.add_argument("--additional_trained_model", nargs = "*", help = "prefix of the trained model")
#    
#    parser.add_argument("--tgt_fn", help = "target text")
#    
#    parser.add_argument("--nbest_to_rescore", help = "nbest list in moses format")
#    
#    parser.add_argument("--ref", help = "target text")
#    
#    parser.add_argument("--gpu", type = int, help = "specify gpu number to use, if any")
#    
#    parser.add_argument("--max_nb_ex", type = int, help = "only use the first MAX_NB_EX examples")
#    parser.add_argument("--mb_size", type = int, default= 80, help = "Minibatch size")
#    parser.add_argument("--nb_batch_to_sort", type = int, default= 20, help = "Sort this many batches by size.")
#    parser.add_argument("--beam_opt", default = False, action = "store_true")
#    parser.add_argument("--tgt_unk_id", choices = ["attn", "id"], default = "align")
#    
#    # arguments for unk replace
#    parser.add_argument("--dic")
#    
#    parser.add_argument("--reverse_training_config", help = "prefix of the trained model")
#    parser.add_argument("--reverse_trained_model", help = "prefix of the trained model")
#    
#    parser.add_argument("--netiface", help = "network interface for listening request", default = 'eth0')
#    parser.add_argument("--port", help = "port for listening request", default = 44666)
#    parser.add_argument("--segmenter_command", help = "command to communicate with the segmenter server")
#    parser.add_argument("--segmenter_format", help = "format to expect from the segmenter (parse_server, morph)", default = 'parse_server')

def define_parser(parser):
    parser.add_argument("training_config", help = "prefix of the trained model")
    parser.add_argument("trained_model", help = "prefix of the trained model")
    
    parser.add_argument("--additional_training_config", nargs = "*", help = "prefix of the trained model")
    parser.add_argument("--additional_trained_model", nargs = "*", help = "prefix of the trained model")
    
    parser.add_argument("--tgt_fn", help = "target text")
    
    parser.add_argument("--nbest_to_rescore", help = "nbest list in moses format")
    
    parser.add_argument("--ref", help = "target text")
    
    parser.add_argument("--gpu", type = int, help = "specify gpu number to use, if any")
    
    parser.add_argument("--max_nb_ex", type = int, help = "only use the first MAX_NB_EX examples")
    parser.add_argument("--mb_size", type = int, default= 80, help = "Minibatch size")
    parser.add_argument("--nb_batch_to_sort", type = int, default= 20, help = "Sort this many batches by size.")
    parser.add_argument("--beam_opt", default = False, action = "store_true")
    parser.add_argument("--tgt_unk_id", choices = ["attn", "id"], default = "align")
    
    # arguments for unk replace
    parser.add_argument("--dic")
    
    parser.add_argument("--reverse_training_config", help = "prefix of the trained model")
    parser.add_argument("--reverse_trained_model", help = "prefix of the trained model")
    
    parser.add_argument("--netiface", help = "network interface for listening request", default = 'eth0')
    parser.add_argument("--port", help = "port for listening request", default = 44666)
    parser.add_argument("--segmenter_command", help = "command to communicate with the segmenter server")
    parser.add_argument("--segmenter_format", help = "format to expect from the segmenter (parse_server, morph)", default = 'parse_server')

def command_line(arguments = None):
    import argparse
    parser = argparse.ArgumentParser(description= "Launch a RNNSearch server", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    define_parser(parser)
    args = parser.parse_args(args = arguments)
    do_start_server(args)
    
def do_start_server(args):
    import nmt_chainer.translation.server
    nmt_chainer.translation.server.do_start_server(args)
