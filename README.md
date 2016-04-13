# RNNSearch
## An implementation of RNNSearch in Chainer.

##Usage:

1. Create Data

 ```
#!shell
python make_data.py /path/train/ja /path/train/en /path_to_experiment_dir/prefix --test_src /path/test/ja --test_tgt /path/test/en --dev_src /path/dev/ja --dev_tgt /path/dev/en
```

2. Train

 ```
#!shell
python train.py /path_to_experiment_dir/prefix /path_to_experiment_dir/prefix_train --gpu 0
```

## More options

run:

python make_data.py --help

python train.py --help
