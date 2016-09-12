# ChaiNMT
## An implementation of Neural Machine Translation in Chainer.
This code mainly implements the Neural Machine Translation system described in (Bahdanau et al., 2015). Also known as "RNNSearch", or "Sequence-to-Sequence Modeling with Attention Mechanism", this is, as of 2016, the most commonly used approach in the recent field of Neural Machine Translation.

This implementation uses the Chainer Deep Learning Library (http://chainer.org/).

#Requirements:
* Python 2.7.x
* A recent version of Chainer (> 1.9). Install with:

    #! shell
    pip install chainer

* Optionally, the plotting libraries plotly and bokeh are used in some visualisation scripts:

    #! shell
    pip install plotly
    pip install bokeh

##Usage:

0. Gather some training data.

        - Training data is in the form of two files: a source language file and a target language file. Each is a utf-8 file containing one sentence per line, segmented into "words" by standard whitespace.

1. Convert Data

        #! shell
        python make_data.py /path/train/ja /path/train/en /path_to_experiment_dir/prefix --test_src /path/test/ja --test_tgt /path/test/en --dev_src /path/dev/ja --dev_tgt /path/dev/en

2. Train

        #! shell
        python train.py /path_to_experiment_dir/prefix /path_to_experiment_dir/prefix_train --gpu 0

## More options

run:

python make_data.py --help

python train.py --help

## Server Mode

Preparation: 

0. Copy your own config file from the provided sample file and adjust it in function of your environment.

cp server.conf.sample server.conf

1. Start a parse server if needed.

cd $PARSE_SERVER_HOME
./src/parse_server.pl -n 50 > parse.log 2>&1

2. Start the server.  Remember to specify the GPU number.

bin/server 0
