"""replace_tgt_unk.py: Simple utility to replace target unknown words in nmt output"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import codecs, itertools, json, unicodedata
import logging
logging.basicConfig()
log = logging.getLogger("rnns:replace_tgt")
log.setLevel(logging.INFO)



def replace_unk_from_string(translation_str, src_str, dic_fn, remove_unk, normalize_unicode_unk, 
    attempt_to_relocate_unk_source):

    dic = None
    if dic_fn is not None:
        dic = json.load(open(dic_fn))
    
    splitted_t = translation_str.strip().split(" ")
    splitted_s = src_str.strip().split(" ")
    new_t = []
    for p_w, w in enumerate(splitted_t):
        if w.startswith("#T_UNK_"):
            src_pos = int(w[7:-1])
            if src_pos >= len(splitted_s):
                log.warn("link to source out of bound (%i/%i line %i)" %(src_pos, len(splitted_s), num_line + 1))
                src_pos = len(splitted_s) -1
            src_w = splitted_s[src_pos]
            
            if attempt_to_relocate_unk_source:
                if dic is not None and src_w in dic:
                    prec_w = splitted_t[p_w -1] if p_w != 0 else None
#                     post_w = splitted_t[p_w  +1] if (p_w + 1) < len(splitted_t) else None
                    if prec_w is not None and dic[src_w] == prec_w and (src_pos + 1) <  len(splitted_s):
                        log.info("retargeting unk  (%i/%i line %i)" %(src_pos, len(splitted_s), num_line + 1))
                        src_w = splitted_s[src_pos + 1]
                    
#                 log.info("replacing %s %i"%(src_w, len(dic)))
            if dic is not None and src_w in dic:
                new_t.append(dic[src_w])
            else:
#                     log.info("not found %s"%(src_w,))
                if not remove_unk:
                    if normalize_unicode_unk:
                        src_w = unicodedata.normalize("NFKD", src_w)
                    new_t.append(src_w)
        else:
            new_t.append(w)
    return " ".join(new_t) + "\n"

def replace_unk(translations, src_file, dest, dic_fn, remove_unk, normalize_unicode_unk,
                attempt_to_relocate_unk_source):

    ft = codecs.open(translations, encoding = "utf8")
    fs = codecs.open(src_file, encoding = "utf8")
    
    fd = codecs.open(dest, "w", encoding = "utf8")
    
    dic = None
    if dic_fn is not None:
        dic = json.load(open(dic_fn))
    
    for num_line, (line_t, line_s) in enumerate(itertools.izip(ft, fs)):
        splitted_t = line_t.strip().split(" ")
        splitted_s = line_s.strip().split(" ")
        new_t = []
        for p_w, w in enumerate(splitted_t):
            if w.startswith("#T_UNK_"):
                src_pos = int(w[7:-1])
                if src_pos >= len(splitted_s):
                    log.warn("link to source out of bound (%i/%i line %i)" %(src_pos, len(splitted_s), num_line + 1))
                    src_pos = len(splitted_s) -1
                src_w = splitted_s[src_pos]
                
                if attempt_to_relocate_unk_source:
                    if dic is not None and src_w in dic:
                        prec_w = splitted_t[p_w -1] if p_w != 0 else None
    #                     post_w = splitted_t[p_w  +1] if (p_w + 1) < len(splitted_t) else None
                        if prec_w is not None and dic[src_w] == prec_w and (src_pos + 1) <  len(splitted_s):
                            log.info("retargeting unk  (%i/%i line %i)" %(src_pos, len(splitted_s), num_line + 1))
                            src_w = splitted_s[src_pos + 1]
                        
#                 log.info("replacing %s %i"%(src_w, len(dic)))
                if dic is not None and src_w in dic:
                    new_t.append(dic[src_w])
                else:
#                     log.info("not found %s"%(src_w,))
                    if not remove_unk:
                        if normalize_unicode_unk:
                            src_w = unicodedata.normalize("NFKD", src_w)
                        new_t.append(src_w)
            else:
                new_t.append(w)
        fd.write(" ".join(new_t) + "\n")
        
def commandline():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("translations")
    parser.add_argument("src_file")
    parser.add_argument("dest")
    parser.add_argument("--dic")
    parser.add_argument("--remove_unk", default = False, action = "store_true")
    parser.add_argument("--normalize_unicode_unk", default = False, action = "store_true")
    parser.add_argument("--attempt_to_relocate_unk_source", default = False, action = "store_true")
    args = parser.parse_args()
    
    replace_unk(args.translations, args.src_file, args.dest, args.dic, args.remove_unk, 
                args.normalize_unicode_unk,
                args.attempt_to_relocate_unk_source)
        
if __name__ == '__main__':
    commandline()
