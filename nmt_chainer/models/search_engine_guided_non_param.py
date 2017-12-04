

def create_context_memory(encdec, biprocessor, search_engine, src_sentence):
    examples_list = search_engine.search(src_sentence)
    
    for ex_src, ex_tgt in examples_list:
        if isinstance(biprocessor, tuple):
            assert len(biprocessor) == 2
            idx_ex_src, idx_ex_tgt = biprocessor[0].convert(ex_src), biprocessor[1].convert(ex_tgt)
        else:
            idx_ex_src, idx_ex_tgt = biprocessor.convert(ex_src, ex_tgt)
            
        state_context_list = encdec.compute_state_context_list(idx_ex_src, idx_ex_tgt)
                                        # encode idx_ex_src
                                        # generate conditionalized cell
                                        # apply conditionalized cell to idx_ex_tgt to generate sequence of (states, ci, yt)
        context_memory.add(state_context_list)
    return context_memory


import nmt_chainer.training_module.train as train
import nmt_chainer.training_module.train_config as train_config

def test_context_memory(config_filename, search_engine, src_sentence):
    config_training = train_config.load_config_train(config_filename)
    (encdec, eos_idx, src_indexer, tgt_indexer), model_infos = train.create_encdec_and_indexers_from_config_dict(config_training,
                                                                                                                                        load_config_model="yes",
                                                                                                                                        return_model_infos=True)
    
    ctxt_mem = create_context_memory(encdec, (src_indexer, tgt_indexer), search_engine, src_sentence)
    
    # (later) do some tests with ctxt_mem
    
    