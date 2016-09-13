# RNNSearch
## An implementation of RNNSearch in Chainer.

##Usage:

1. Create Data

        #! shell
        python make_data.py /path/train/ja /path/train/en /path_to_experiment_dir/prefix --test_src /path/test/ja --test_tgt /path/test/en --dev_src /path/dev/ja --dev_tgt /path/dev/en

1. Train

        #! shell
        python train.py /path_to_experiment_dir/prefix /path_to_experiment_dir/prefix_train --gpu 0

1. Generate Training Graph

        #! shell
        pip install --user plotly
        python graph_training.py --lib plotly /path_to_experiment_dir/prefix_train/*.result.sqlite ~/public_html/graph.html

## Recommended Options

        #! shell
        python train.py /path_to_experiment_dir/prefix /path_to_experiment_dir/prefix_train --gpu 0
        --optimizer adam
        --weight_decay 0.000001
        --l2_gradient_clipping 1
        --mb_size 64
        --max_src_tgt_length 90
        // use smaller value if frequently running out of memory

## More options

run:

python make_data.py --help

python train.py --help
