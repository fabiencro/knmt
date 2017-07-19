import json
import codecs
import gzip
import logging
from nmt_chainer.dataprocessing import processors

logging.basicConfig()
log = logging.getLogger("rnns:utils:extract")
log.setLevel(logging.INFO)

def define_parser(parser):
    parser.add_argument("datapath")
    parser.add_argument("dest_fn")
    
    
def do_extract(args):
    datapath, destination_fn = args.datapath, args.dest_fn
    voc_fn = datapath + ".voc"
    data_fn = datapath + ".data.json.gz"
    log.info("extracting data from %s using processor in %s", data_fn, voc_fn)
    
    data = json.load(gzip.open(data_fn, "rb"))
    
    bi_pp = processors.load_pp_pair_from_file(voc_fn)
    tgt_processor = bi_pp.tgt_processor()
    for key in data:
        src_fn = destination_fn + ".%s.src.txt"%key
        tgt_fn = destination_fn + ".%s.tgt.txt"%key
        tgt_swallow_fn = destination_fn + ".%s.tgt.swallow.txt"%key
        log.info("extracting key %s into %s and %s and %s", key, src_fn, tgt_fn, tgt_swallow_fn)
        
        src_f = codecs.open(src_fn, "w", encoding = "utf8")
        tgt_f = codecs.open(tgt_fn, "w", encoding = "utf8")
        tgt_swallow_f = codecs.open(tgt_swallow_fn, "w", encoding = "utf8")
        
        for src, tgt in data[key]:
            src_dec, tgt_dec = bi_pp.deconvert(src, tgt)
            src_f.write(src_dec + "\n")
            tgt_f.write(tgt_dec + "\n")
            
            tgt_swallow = tgt_processor.deconvert_swallow(tgt)
            tgt_swallow_string = " ".join(("[@%i]"%w if isinstance(w, int) else w) for w in tgt_swallow)
            tgt_swallow_f.write(tgt_swallow_string + "\n")
