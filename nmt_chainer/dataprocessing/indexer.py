import logging

logging.basicConfig()
log = logging.getLogger("rnns:indexer")
log.setLevel(logging.INFO)


class Indexer(object):

    def __init__(self, unk_tag="#UNK#"):
        self.dic = {}
        self.lst = []
        self.unk_label_dictionary = None
        self.finalized = False

    def add_word(self, w, should_be_new=False, should_not_be_int=True):
        assert not self.finalized
        assert not (should_not_be_int and isinstance(w, int))
        if w not in self.dic:
            new_idx = len(self.lst)
            self.dic[w] = new_idx
            self.lst.append(w)
            assert len(self.lst) == len(self.dic)
        else:
            assert not should_be_new

#     def assign_index_to_voc(self, voc_iter, all_should_be_new=False):
#         assert not self.finalized
#         for w in voc_iter:
#             self.add_word(w, should_be_new=all_should_be_new)

    def finalize(self):
        assert not self.finalized
        assert len(self.dic) == len(self.lst)
        self.add_word(0, should_be_new=False, should_not_be_int=False)
        self.finalized = True

    def get_one_unk_idx(self, w):
        assert self.finalized
        assert w not in self.dic
        if self.unk_label_dictionary is not None:
            return self.dic[self.unk_label_dictionary.get(w, 1)]
        else:
            return self.dic[0]

    def is_unk_idx(self, idx):
        assert self.finalized
        assert idx < len(self.lst)
        return isinstance(self.lst[idx], int)

#     def add_unk_label_dictionary(self, unk_dic):
#         assert not self.finalized
#         self.unk_label_dictionary = unk_dic

    def convert(self, seq):
        assert self.finalized
        assert len(self.dic) == len(self.lst)
        res = [None] * len(seq)
        for pos, w in enumerate(seq):
            assert not isinstance(w, int)
            if w in self.dic:
                idx = self.dic[w]
            else:
                idx = self.get_one_unk_idx(w)
            res[pos] = idx
        return res

#     def convert_and_update_unk_tags(self, seq, give_unk_label):
#         assert not self.finalized
#         assert len(self.dic) == len(self.lst)
#         res = [None] * len(seq)
#         for pos, w in enumerate(seq):
#             assert not isinstance(w, int)
#             if w in self.dic:
#                 idx = self.dic[w]
#             else:
#                 aligned_pos = give_unk_label(pos, w)
#                 if aligned_pos not in self.dic:
#                     self.add_word(aligned_pos, should_be_new=True,
#                                   should_not_be_int=False)
#                 idx = self.dic[aligned_pos]
#             res[pos] = idx
#         return res

    def deconvert(self, seq, unk_tag="#UNK#", no_oov=True, eos_idx=None):
        assert self.finalized
        assert eos_idx is None or eos_idx >= len(self.lst)
        res = []
        for num, idx in enumerate(seq):
            if idx >= len(self.lst):
                if eos_idx is not None and eos_idx == idx:
                    w = "#EOS#"
                elif no_oov:
                    raise KeyError()
                else:
                    log.warn("unknown idx: %i / %i" % (idx, len(self.lst)))
                    continue
            else:
                w = self.lst[idx]

            if isinstance(w, int):
                if callable(unk_tag):
                    w = unk_tag(num, w)
                else:
                    w = unk_tag

            res.append(w)
        return res

    def __len__(self):
        assert self.finalized
        assert len(self.dic) == len(self.lst)
        return len(self.lst)

    def to_serializable(self):
        return {"type": "simple_indexer", "rev": 1, "voc_lst": self.lst, "unk_label_dic": self.unk_label_dictionary}

    @staticmethod
    def make_from_serializable(datas):
        if isinstance(datas, list):
            # legacy mode
            log.info("loading legacy voc")
            voc_lst = datas
            assert 0 not in voc_lst
            res = Indexer()
            res.lst = list(voc_lst)  # add UNK
            for idx, w in enumerate(res.lst):
                assert isinstance(w, basestring)
                res.dic[w] = idx
            assert len(res.dic) == len(res.lst)
            res.finalize()
            return res
        else:
            assert isinstance(datas, dict)
            assert datas["type"] == "simple_indexer"
            assert datas["rev"] == 1
            voc_lst = datas["voc_lst"]
            res = Indexer()
            res.lst = list(voc_lst)
            res.unk_label_dictionary = datas["unk_label_dic"]
            for idx, w in enumerate(voc_lst):
                res.dic[w] = idx
            res.finalized = True
            return res

    @staticmethod
    def check_if_data_indexer(datas):
        if isinstance(datas, list):
            return True
        elif isinstance(datas, dict) and "type" in datas and datas["type"] == "simple_indexer":
            return True

# class Indexer(object):
#     def __init__(self, unk_tag = "#UNK#"):
#         self.dic = {}
#         self.lst = []
#         self.finalized = False
#
#     def add_word(self, w, should_be_new = False):
#         assert not self.finalized
#         assert w is not None
#         if w not in self.dic:
#             new_idx = len(self.lst)
#             self.dic[w] = new_idx
#             self.lst.append(w)
#             assert len(self.lst) == len(self.dic)
#         else:
#             assert not should_be_new
#
#     def assign_index_to_voc(self, voc_iter, all_should_be_new = False):
#         assert not self.finalized
#         for w in voc_iter:
#             self.add_word(w, should_be_new = all_should_be_new)
#
#     def finalize(self):
#         assert not self.finalized
#         self.dic[None] = len(self.lst)
#         self.lst.append(None)
#         self.finalized = True
#
#     def get_unk_idx(self):
#         assert self.finalized
#         return self.dic[None]
#
#     def convert(self, seq):
#         assert self.finalized
#         assert len(self.dic) == len(self.lst)
#         unk_idx = self.get_unk_idx()
# #         res = np.empty( (len(seq),), dtype = np.int32)
#         res = [None] * len(seq)
#         for pos, w in enumerate(seq):
#             assert w is not None
#             idx = self.dic.get(w, unk_idx)
#             res[pos] = idx
#         return res
#
#     def deconvert(self, seq, unk_tag = "#UNK#", no_oov = True, eos_idx = None):
#         assert self.finalized
#         assert eos_idx is None or eos_idx >= len(self.lst)
#         res = []
#         for num, idx in enumerate(seq):
#             if idx >= len(self.lst):
#                 if eos_idx is not None and eos_idx == idx:
#                     w = "#EOS#"
#                 elif no_oov:
#                     raise KeyError()
#                 else:
#                     log.warn("unknown idx: %i / %i"%(idx, len(self.lst)))
#                     continue
#             else:
#                 w = self.lst[idx]
#             if w is None:
#                 if callable(unk_tag):
#                     w = unk_tag(num)
#                 else:
#                     w = unk_tag
#             res.append(w)
#         return res
#
#     def __len__(self):
#         assert self.finalized
#         assert len(self.dic) == len(self.lst)
#         return len(self.lst)
#
#     def to_serializable(self):
#         return {"type": "simple_indexer", "rev": 1,  "voc_lst" : self.lst}
#
#     @staticmethod
#     def make_from_serializable(datas):
#         assert datas["type"] == "simple_indexer"
#         assert datas["rev"] == 1
#         voc_lst = datas["voc_lst"]
#         res = Indexer()
#         res.lst = list(voc_lst)
#         for idx, w in enumerate(voc_lst):
#             res.dic[w] = idx
#         res.finalized = True
#         return res
