import logging
import codecs
import exceptions
from collections import defaultdict

"""lex_dict.py: extract a dictionary from e2f and f2e lexical probability files"""
__author__ = "Chenhui Chu"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "knccch@gmail.com"
__status__ = "Development"

logging.basicConfig()
log = logging.getLogger("aparse")
log.setLevel(logging.INFO)

def load_lex(lex_file, inverse = False):
    lex = codecs.open(lex_file, encoding = "utf8")

    dic = defaultdict(lambda: defaultdict(lambda: 0))
    for line in lex:
        # print line
        fr, en, prob = line.split(r' ')
        if inverse is True:
            dic[en][fr] = prob
        else:
            dic[fr][en] = prob
    return dic

def commandline():
    import argparse, operator, json
    parser = argparse.ArgumentParser()
    parser.add_argument("lex_e2f")
    parser.add_argument("lex_f2e")
    parser.add_argument("dest_dic")
    
    args = parser.parse_args()
    dic_e2f = load_lex(args.lex_e2f, False)
    dic_f2e = load_lex(args.lex_f2e, True)

    prob = {}
    for fr in dic_e2f:
        prob[fr] = 0

    dic = {}
    for fr in dic_e2f:
        en_trans = dic_e2f[fr]
        for en in en_trans:
            prob_f_b = float(dic_e2f[fr][en]) * float(dic_f2e[fr][en])
            if(prob_f_b > float(prob[fr])):
                prob[fr] = prob_f_b
                dic[fr] = en
    #         print "%s, %s, %s, %s, %s" % (fr, en, dic_e2f[fr][en], dic_f2e[fr][en], prob_f_b)
    
    # print "======"
    # for fr in dic:
    #     print fr, dic[fr], prob[fr]
    log.info("saving")
    json.dump(dic, open(args.dest_dic, "w"))
   
         
if __name__ == '__main__':
    commandline()
