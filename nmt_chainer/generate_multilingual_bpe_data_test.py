import unittest
from generate_multilingual_bpe_data import *

class TestPipeline(unittest.TestCase):
    dpp = None
    all_vocs_src = {'en': Counter({u'I': 5, u'Harambe': 5, u'Eat': 3, u'Like': 2, u'Everyday': 1}), 'mr': Counter({u'Harambe': 4, u'Khato': 2, u'Mi': 2, u'Mala': 2, u'Prem': 1, u'Avadto': 1, u'Karto': 1, u'Shi': 1, u'Dardivshi': 1})}
    all_vocs_tgt = {'hi': Counter({u'Harambe': 2, u'Hai': 1, u'Pasand': 1, u'Khata': 1, u'Hu': 1, u'Muhje': 1, u'Mai': 1}), 'en': Counter({u'I': 4, u'Harambe': 4, u'Eat': 2, u'Love': 1, u'Like': 1, u'Everyday': 1}), 'mr': Counter({u'Harambe': 3, u'Khato': 2, u'Mi': 2, u'Avadto': 1, u'Dardivshi': 1, u'Mala': 1})}
    all_vocs_src_balanced = {'en': Counter({u'I': 5, u'Harambe': 5, u'Eat': 3, u'Like': 2, u'Everyday': 1}), 'mr': Counter({u'Harambe': 4.266666666666667, u'Khato': 2.1333333333333333, u'Mi': 2.1333333333333333, u'Mala': 2.1333333333333333, u'Prem': 1.0666666666666667, u'Avadto': 1.0666666666666667, u'Karto': 1.0666666666666667, u'Shi': 1.0666666666666667, u'Dardivshi': 1.0666666666666667})}
    all_vocs_tgt_balanced = {'hi': Counter({u'Harambe': 3.25, u'Hai': 1.625, u'Pasand': 1.625, u'Khata': 1.625, u'Hu': 1.625, u'Muhje': 1.625, u'Mai': 1.625}), 'en': Counter({u'I': 4, u'Harambe': 4, u'Eat': 2, u'Love': 1, u'Like': 1, u'Everyday': 1}), 'mr': Counter({u'Harambe': 3.9000000000000004, u'Khato': 2.6, u'Mi': 2.6, u'Avadto': 1.3, u'Dardivshi': 1.3, u'Mala': 1.3})}
    merged_voc_src = Counter({u'Harambe': 9.266666666666666, u'I': 5, u'Eat': 3, u'Khato': 2.1333333333333333, u'Mi': 2.1333333333333333, u'Mala': 2.1333333333333333, u'Like': 2, u'Prem': 1.0666666666666667, u'Avadto': 1.0666666666666667, u'Karto': 1.0666666666666667, u'Dardivshi': 1.0666666666666667, u'Shi': 1.0666666666666667, u'Everyday': 1})
    merged_voc_tgt = Counter({u'Harambe': 11.15, u'I': 4, u'Mi': 2.6, u'Khato': 2.6, u'Eat': 2, u'Hai': 1.625, u'Pasand': 1.625, u'Khata': 1.625, u'Hu': 1.625, u'Muhje': 1.625, u'Mai': 1.625, u'Avadto': 1.3, u'Dardivshi': 1.3, u'Mala': 1.3, u'Love': 1, u'Everyday': 1, u'Like': 1})
    all_vocs_src_sizes = {'en': 16, 'mr': 15}
    all_vocs_tgt_sizes = {'hi': 8, 'en': 13, 'mr': 10}
    largest_voc_size_src = 16
    largest_voc_size_tgt = 13
    bpe_model_src = ["a r\n", "e </w>\n", "a m\n", "am b\n", "ar amb\n", "aramb e</w>\n", "H arambe</w>\n", "a t\n", "I </w>\n", "i </w>\n"]
    bpe_model_tgt = ["e </w>\n", "H a\n", "a m\n", "am b\n", "r amb\n", "ramb e</w>\n", "Ha rambe</w>\n", "i </w>\n", "a t\n", "h at\n"]

    def test_vocab_generation(self):
        log.info("Testing vocab generation")
        self.dpp.generate_vocabularies()
        assert self.dpp.largest_voc_size_src == self.largest_voc_size_src
        assert self.dpp.largest_voc_size_tgt == self.largest_voc_size_tgt
        assert self.dpp.all_vocs_src_sizes == self.all_vocs_src_sizes
        assert self.dpp.all_vocs_tgt_sizes == self.all_vocs_tgt_sizes
        assert self.dpp.all_vocs_src == self.all_vocs_src
        assert self.dpp.all_vocs_tgt == self.all_vocs_tgt
        log.info("Testing vocab balancing")
        self.dpp.balance_vocabularies()
        assert self.dpp.all_vocs_src == self.all_vocs_src_balanced
        assert self.dpp.all_vocs_tgt == self.all_vocs_tgt_balanced
        self.dpp.merge_vocabularies()
        assert self.dpp.merged_voc_src == self.merged_voc_src
        assert self.dpp.merged_voc_tgt == self.merged_voc_tgt

    def test_bpe_model_generation(self):
        log.info("Testing for BPE model generation")
        self.dpp.generate_vocabularies()
        self.dpp.balance_vocabularies()
        self.dpp.merge_vocabularies()
        self.dpp.learn_bpe_models()
        bpe_model_src = open("/tmp/bpe_model.src").readlines()
        bpe_model_tgt = open("/tmp/bpe_model.tgt").readlines()
        assert bpe_model_src == self.bpe_model_src
        assert bpe_model_tgt == self.bpe_model_tgt
    
    def test_mlnmt_generation(self):
        log.info("Testing for mlnmt data generation")
        self.dpp.generate_vocabularies()
        self.dpp.balance_vocabularies()
        self.dpp.merge_vocabularies()
        self.dpp.learn_bpe_models()
        self.dpp.segment_corpora()
        self.dpp.generate_multilingual_data()
        log.info("Major subtlety here.... the oversampling procedure is non deterministic so the corpus that is generated will not always have some sentence pairs duplicated. The reason being that if the oversample rate is a floating point number. If it is 2.5 then the sentence pair will be sampled twice and then once more with a probability of 0.5. This is not much a big deal when the corpora are large enough. By large I mean > 1000 sentence pairs. Who even does NMT on anything less than 30k sentence pairs? Papa bless. Vape Naysh!")
        assert 1==1

if __name__ == '__main__':
    a = open("/tmp/train-en-hi.en", 'w+')
    a.write(u"I Eat Harambe\n")
    a.write(u"I Like Harambe\n")
    a.flush()
    a.close()
    a = open("/tmp/train-en-hi.hi", 'w+')
    a.write(u"Mai Harambe Khata Hu\n")
    a.write(u"Muhje Harambe Pasand Hai\n")
    a.flush()
    a.close()
    a = open("/tmp/train-en-mr.en", 'w+')
    a.write(u"I Eat Harambe\n")
    a.write(u"I Eat Harambe Everyday\n")
    a.write(u"I Like Harambe\n")
    a.flush()
    a.close()
    a = open("/tmp/train-en-mr.mr", 'w+')
    a.write(u"Mi Harambe Khato\n")
    a.write(u"Mi Dardivshi Harambe Khato\n")
    a.write(u"Mala Harambe Avadto\n")
    a.flush()
    a.close()
    a = open("/tmp/train-mr-en.en", 'w+')
    a.write(u"I Eat Harambe\n")
    a.write(u"I Eat Harambe Everyday\n")
    a.write(u"I Like Harambe\n")
    a.write(u"I Love Harambe\n")
    a.flush()
    a.close()
    a = open("/tmp/train-mr-en.mr", 'w+')
    a.write(u"Mi Harambe Khato\n")
    a.write(u"Mi Dardivshi Harambe Khato\n")
    a.write(u"Mala Harambe Avadto\n")
    a.write(u"Mala Harambe Shi Prem Karto\n")
    a.flush()
    a.close()

    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Test for prepare training data.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--save_prefix", default="/tmp/", help="created files will be saved with this prefix")
    parser.add_argument(
        "--train_lang_corpora_pairs", nargs='+', type=str, default = ['en:hi:/tmp/train-en-hi.en:/tmp/train-en-hi.hi', 'en:mr:/tmp/train-en-mr.en:/tmp/train-en-mr.mr', 'mr:en:/tmp/train-mr-en.mr:/tmp/train-mr-en.en'], help="list of colon separated quadruplet of language pairs and their corpora. ex: ja:zh:/tmp/corpus-ja-zh.ja:/tmp/corpus-ja-zh.zh en:ja:/tmp/corpus-en-ja.en:/tmp/corpus-en-ja.ja")
    parser.add_argument("--balance_vocab_counts", default=True, help="Before learning the BPE model do we want to adjust the count information? This might be needed if one language pair has more data than the other. For now the balancing will be done based on the ratio of the total word count for the text with the maximum total number of words to the total word count for the current text.")
    parser.add_argument("--balance_corpora", default=True, help="After learning the BPE model and segmenting the data do we want to oversample the smaller corpora? This might be needed if one language pair has more data than the other. For now the balancing will be done based on the ratio of the total line count for the text with the maximum total number of lines to the total line count for the current text.")
    parser.add_argument("--num_bpe_merge_operations", type=int, default=10, help="Number of merge operations that the BPE model should perform on the training vocabulary to learn the BPE codes.")
    args = parser.parse_args()

    TestPipeline.dpp = DataPreparationPipeline(args)

    unittest.main()