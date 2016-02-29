import codecs, itertools, json

"""replace_tgt_unk.py: Simple utility to replace target unknown words in nmt output"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

def commandline():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("translations")
    parser.add_argument("src_file")
    parser.add_argument("dest")
    parser.add_argument("--dic")
    parser.add_argument("--remove_unk", default = False, action = "store_true")
    args = parser.parse_args()
    
    ft = codecs.open(args.translations, encoding = "utf8")
    fs = codecs.open(args.src_file, encoding = "utf8")
    
    fd = codecs.open(args.dest, "w", encoding = "utf8")
    
    dic = None
    if args.dic is not None:
        dic = json.load(open(args.dic))
    
    for line_t, line_s in itertools.izip(ft, fs):
        splitted_t = line_t.strip().split(" ")
        splitted_s = line_s.strip().split(" ")
        new_t = []
        for w in splitted_t:
            if w.startswith("#T_UNK_"):
                src_pos = int(w[7:-1])
                src_w = splitted_s[src_pos]
                if dic is not None and src_w in dic:
                    new_t.append(dic[src_w])
                else:
                    if not args.remove_unk:
                        new_t.append(src_w)
            else:
                new_t.append(w)
        fd.write(" ".join(new_t) + "\n")
        
if __name__ == '__main__':
    commandline()