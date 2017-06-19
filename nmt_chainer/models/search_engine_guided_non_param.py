import logging
logging.basicConfig()
log = logging.getLogger("rnns:models")
log.setLevel(logging.INFO)


def create_reference_memory(encdec, biprocessor, retriever, src_sentence):
    examples_list = retriever.retrieve(src_sentence)
    log.info("retrieved %s" % examples_list)
    
    reference_memory = []
    for ex_src, ex_tgt in examples_list:
        if isinstance(biprocessor, tuple):
            assert len(biprocessor) == 2
            idx_ex_src, idx_ex_tgt = biprocessor[0].convert(ex_src), biprocessor[1].convert(ex_tgt)
        else:
            idx_ex_src, idx_ex_tgt = biprocessor.convert(ex_src, ex_tgt)

        reference_memory.extend(encdec.compute_reference_memory(idx_ex_src, idx_ex_tgt))

    return reference_memory
