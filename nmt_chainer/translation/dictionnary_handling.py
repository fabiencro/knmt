
import math
import ahocorasick
import tqdm
from collections import defaultdict
import unicodedata
import os
import pickle

############################################################
# Trie loading and creation
# ja: source, en: target
#

def unsegment(s):
    res = []
    for w in s.split(" "):
        if w.startswith("▁"):
            w = " " + w[1:]
        res.append(w)
    return "".join(res)

def preproccess_line(line, do_unsegment=False):
    line = line.strip()
    if do_unsegment:
        line = unsegment(line)
    line = unicodedata.normalize('NFKC', line)
    line = line.lower()
    return line

def tqdm_utf_file_reader(fn):
    print("Reading", fn)
    with tqdm.tqdm(total=os.path.getsize(fn)) as pbar:
        with open(fn, "r", encoding="utf8") as f:
            for line in f:
                pbar.update(len(line.encode("utf8")))
                yield line

def load_dic_from_tsv(filename, invert=False):
    ja_en = defaultdict(list)
    en_ja = defaultdict(list)
    for line in tqdm_utf_file_reader(filename):
        line = line.strip()
        line = preproccess_line(line)
        splitted = line.split("\t")
        ja, en, idx = splitted

        if len(en) <= 2:
            continue # filtering noise
        en = " "+en+" "

        ja_en[ja].append(en)
        en_ja[en].append(ja)

    if invert:
        return en_ja, ja_en
    else:
        return ja_en, en_ja
    

def build_dic_search(dic: dict):
    A = ahocorasick.Automaton()
    print("adding words to automaton")
    for k, v in tqdm.tqdm(dic.items()):
        A.add_word(k, (k,v))
    print("finalize automaton")
    A.make_automaton()
    return A

def create_search_trie_from_dic_file(tsv_filename, invert=False):
    ja_en_dic, en_ja_dic = load_dic_from_tsv(tsv_filename, invert=invert)

    ja_en_search = build_dic_search(ja_en_dic)
    en_ja_search = build_dic_search(en_ja_dic)
    return ja_en_search, en_ja_search

def create_and_save_search_trie_from_dic_file(tsv_filename, dest_filename):
    search_pair = create_search_trie_from_dic_file(tsv_filename)
    pickle.dump(search_pair, open(dest_filename, "bw"))


def load_search_trie(filename, invert=False):
    ja_en_search, en_ja_search = pickle.load(open(filename, "rb"))
    if invert:
        return ja_en_search, en_ja_search 
    else:
        return en_ja_search, ja_en_search


def match_list(line, lst):
    match = None
    for en_w in lst:
        if en_w in line:
            match = en_w
            break
    return match

def count_match_in_list(line, lst):
    res = 0
    for en_w in lst:
        res += line.count(en_w)
    return res


def find_match_in_list(line, lst):
    A = ahocorasick.Automaton()
    for w in lst:
        A.add_word(w,w)
    A.make_automaton()
    m = {}
    for end, w in A.iter(line):
        if end not in m:
             m[end] = w
             for i in range(1,len(w)):
                 if end-i in m:
                     del m[end-i]
    return m
           

def get_non_overlap_matches(A, line):
    m = {}
    for end, (ja_w, en_w_list) in A.iter(line):
        if len(en_w_list) > 30:
            continue  #remove some dictionnary noise
        if ja_w.strip() in ["ら", "。。", "が", "よう", "な", "べき",
                             "か", "し", "ように", "である",
                             "もの", "ない", "を", "した", "いう", "この", "も", "こと", "できる",
                             "ある", "さ", "れた",
                            "the", "and", "this", "for", "of", "a"]:
            continue
        if end not in m:
             m[end] = (ja_w, set(en_w_list))
             for i in range(1,len(ja_w)):
                 if end-i in m:
                     del m[end-i]
    counts = defaultdict(int)
    translations = {}
    for w, lst in m.values():
        counts[w] += 1
        translations[w] = lst
    return counts, translations

def get_match_list(line_ja, line_en, ja_en_search):
    #matched_list = []
    #missed_list = []
    matched_count, translations = get_non_overlap_matches(ja_en_search, line_ja) #defaultdict(int)
    other_count = {}
    search_line = line_en
    for ja_w, en_w_list in translations.items():
        m = find_match_in_list(search_line, en_w_list) 
        count = 0 
        for num_match, (end, w_en) in enumerate(m.items()):
            if num_match + 1 > matched_count[ja_w]:
                continue
            count += 1
            search_line = search_line[:end-len(w_en)] + " "*len(w_en) + search_line[end:]

            
        other_count[ja_w] = count #find_match_in_list(line_en, en_w_list)

    # for end_pos, (ja_w, en_w_list) in ja_en_search.iter(line_ja):
    #     matched_count[ja_w]+=1
    #     if ja_w not in other_count:
    #         other_count[ja_w] = count_match_in_list(line_en, en_w_list)

        # match = match_list(line_en, en_w_list)
        # if match is None:
        #     missed_list.append(ja_w)
        # else:
        #     matched_list.append((ja_w, match))
    return matched_count, other_count, translations





#PSEUDO_MINUS_INF = -15
def compute_dic_matching_score(nb_dic_words_in_src, nb_src_in_tgt, nb_dic_words_in_tgt, nb_tgt_in_src):
    dic_recall = nb_src_in_tgt / nb_dic_words_in_src if nb_dic_words_in_src != 0 else 1
    dic_precision = nb_tgt_in_src / nb_dic_words_in_tgt if nb_dic_words_in_tgt != 0 else 1

    recall_score = nb_src_in_tgt
    precision_score = nb_tgt_in_src - nb_dic_words_in_tgt #max(np.log(dic_precision), PSEUDO_MINUS_INF) if dic_precision > 0 else PSEUDO_MINUS_INF


    return recall_score + precision_score

def get_dic_score(line_ja, line_en, ja_en_search, en_ja_search):

    line_ja = preproccess_line(line_ja, do_unsegment=True)
    line_en = preproccess_line(line_en, do_unsegment=True)

    line_ja = "".join(line_ja.split(" "))
    line_en = " " + line_en.replace("!", " ").replace(".", " ").replace("?", " ").replace(",", " ").replace(";", " ").replace(":", " ") + " "

    matched_count_ja_en, other_count_ja_en, translations_ja = get_match_list(line_ja, line_en, ja_en_search)
    src_tgt_mismatch = 0

    src_missing = 0
    src_ok = 0
    src_list = []
    for ja in matched_count_ja_en:
        src_ok += min(matched_count_ja_en[ja], other_count_ja_en[ja])
        src_missing += max(matched_count_ja_en[ja] - other_count_ja_en[ja], 0)

        src_list.append((ja, translations_ja[ja],matched_count_ja_en[ja], other_count_ja_en[ja],
                min(matched_count_ja_en[ja], other_count_ja_en[ja]),
                max(matched_count_ja_en[ja] - other_count_ja_en[ja], 0)))

        #src_tgt_mismatch += abs(matched_count_ja_en[ja] - other_count_ja_en[ja])
        # if src_tgt_mismatch == 0:
        #     print(ja)
        #print(ja, matched_count_ja_en[ja], other_count_ja_en[ja], translations_ja[ja])

    #nb_dic_words_in_src = len(matched_list_ja_en) + len(missed_list_ja_en)
    #nb_src_in_tgt = len(matched_list_ja_en)

    matched_count_en_ja, other_count_en_ja, translations_en = get_match_list(line_en, line_ja, en_ja_search)
    #tgt_src_mismatch = 0
    tgt_missing = 0
    tgt_ok = 0
    tgt_list = []
    for en in matched_count_en_ja:
        tgt_ok += min(matched_count_en_ja[en], other_count_en_ja[en])
        tgt_missing += max(matched_count_en_ja[en] - other_count_en_ja[en], 0)
        tgt_list.append((en, translations_en[en], min(matched_count_en_ja[en], other_count_en_ja[en]),
                             max(matched_count_en_ja[en] - other_count_en_ja[en], 0)))
        #tgt_src_mismatch += abs(matched_count_en_ja[en] - other_count_en_ja[en])
        # if tgt_src_mismatch == 0:
        #     print(en)
        #print(en, matched_count_en_ja[en], other_count_en_ja[en], translations_en[en])

    #print(src_tgt_mismatch, tgt_src_mismatch)
    #return -src_tgt_mismatch -tgt_src_mismatch
    
    #print(line_ja)
    #print(line_en)
    #print(src_list, tgt_list)
    #print(f"{src_ok + tgt_ok - src_missing - tgt_missing} = {src_ok} + {tgt_ok} - {src_missing} - {tgt_missing}")
    return src_ok + tgt_ok - src_missing - tgt_missing

    # nb_dic_words_in_tgt = len(matched_list_en_ja) + len(missed_list_en_ja)
    # nb_tgt_in_src = len(matched_list_en_ja)


    # print(len(matched_list_ja_en), len(missed_list_ja_en), 
    #         len(matched_list_en_ja), len(missed_list_en_ja),
    #         compute_dic_matching_score(nb_dic_words_in_src, nb_src_in_tgt, nb_dic_words_in_tgt, nb_tgt_in_src))

    # return compute_dic_matching_score(nb_dic_words_in_src, nb_src_in_tgt, nb_dic_words_in_tgt, nb_tgt_in_src)


def make_constraint(ja_en_search, en_ja_search, tgt_indexer):
    def make_dic_score_computer(line_ja, src_idx):

        def score_computer(tgt_idx):
            line_en = tgt_indexer.deconvert(tgt_idx)
            return get_dic_score(line_ja, line_en, ja_en_search, en_ja_search)

        return score_computer
    return make_dic_score_computer

    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=None,
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                        prog=None)
                                        
    parser.add_argument("tsv_filename", help=None) 
    parser.add_argument("dest_filename", help=None)                         
                                        
    args = parser.parse_args()

    create_and_save_search_trie_from_dic_file(args.tsv_filename, args.dest_filename)
