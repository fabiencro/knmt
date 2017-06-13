def create_reference_memory(encdec, biprocessor, retriever, src_sentence):
    examples_list = retriever.retrieve(src_sentence)
    
    reference_memory = []
    for ex_src, ex_tgt in examples_list:
        if isinstance(biprocessor, tuple):
            assert len(biprocessor) == 2
            idx_ex_src, idx_ex_tgt = biprocessor[0].convert(ex_src), biprocessor[1].convert(ex_tgt)
        else:
            idx_ex_src, idx_ex_tgt = biprocessor.convert(ex_src, ex_tgt)
            
        reference_memory.extend(encdec.compute_reference_memory(idx_ex_src, idx_ex_tgt))

    return reference_memory


import nmt_chainer.training_module.train as train
import nmt_chainer.training_module.train_config as train_config


def test_reference_memory(config_filename, search_engine, src_sentence):
    config_training = train_config.load_config_train(config_filename)
    (encdec, eos_idx, src_indexer, tgt_indexer), model_infos = \
        train.create_encdec_and_indexers_from_config_dict(config_training,
                                                          load_config_model="yes",
                                                          return_model_infos=True)
    
    ctxt_mem = create_reference_memory(encdec, (src_indexer, tgt_indexer), search_engine, src_sentence)
    
    # (later) do some tests with ctxt_mem
