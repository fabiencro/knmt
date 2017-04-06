import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable, Chain
import chainer.serializers as serializers

import numpy as np






class CharDec(Chain):
    def __init__(self, V, Ec, H):
        super(CharDec, self).__init__(
            lin_out = L.Linear(H, V + 1),
            c_emb_dec = L.EmbedID(V, Ec),
            nstep_dec = L.NStepLSTM(1, Ec, H, dropout = 0.5)
        )
#         self.start_id = V
        self.H = H
        self.eos_id = V #self.xp.array([V], dtype = self.xp.int32)
        
    def append_eos_id(self, a):
        return self.xp.concatenate((a, self.xp.array([self.eos_id], dtype = self.xp.int32)), axis = 0)
        
    def compute_loss(self, hx, dataset_as_array_list, 
                     need_to_append_eos = True,
                    use_gumbel = False,
                    temperature = None,
                     train = True,
                    verbose = False):
        if need_to_append_eos:
            dataset_as_var_with_eos = [Variable(self.append_eos_id(a)) for a in dataset_as_array_list]
        else:
            dataset_as_var_with_eos = [Variable(a) for a in dataset_as_array_list]
            
        if need_to_append_eos:
            dataset_as_var_without_eos = [Variable(a) for a in dataset_as_array_list]
        else:
            dataset_as_var_without_eos = [Variable(a[:-1]) for a in dataset_as_array_list]
            
        nb_ex = len(dataset_as_array_list)
        dataset_as_emb_dec = [self.c_emb_dec(v) for v in dataset_as_var_without_eos]
        hx_dec, cx_dec, xs_dec = self.nstep_dec(hx, None, dataset_as_emb_dec, train = train)
        hx_initial = F.split_axis(hx.reshape(nb_ex, -1), nb_ex, axis = 0, force_tuple = True)
        logits_list = [self.lin_out(F.concat((hxi, h), axis = 0)) for hxi, h in zip(hx_initial, xs_dec)]
    
        if verbose:
            print "logits:"
            for logits in logits_list:
                print logits.data
            print
            
        if use_gumbel:
            logits_list = [(logits + self.xp.random.gumbel(size = logits.data.shape)) for logits in logits_list]
        
        if temperature is not None:
            logits_list = [logits/temperature for logits in logits_list]
        
        losses = [F.softmax_cross_entropy(logits, tgt) for logits,tgt in zip(logits_list, dataset_as_var_with_eos)]
        loss = sum(losses)/nb_ex
        return loss
    
    def decode(self, hx, length = 10, verbose = False, train = False):
        hx_dec = hx
        cx_dec = None
#         prev_word = xp.array([self.start_id], dtype = xp.float32)
        nb_inpt = hx.data.shape[1]
        result = [[] for _ in xrange(nb_inpt)]
        finished = [False] * nb_inpt
        for i in xrange(length):
            logits = self.lin_out(hx_dec.reshape(-1, self.H))
            if verbose:
                print "logits", i
                print logits.data
            prev_word = self.xp.argmax(logits.data, axis = 1).astype(self.xp.int32)
            for num_inpt in xrange(nb_inpt):
                if prev_word[num_inpt] == self.eos_id:
                    finished[num_inpt] = True
                if not finished[num_inpt]:
                    result[num_inpt].append(prev_word[num_inpt])
                if finished[num_inpt]:
                    prev_word[num_inpt] = 0
                    
            if verbose:
                print "prev_word", prev_word
#             print prev_word
            prev_word_emb = F.split_axis(self.c_emb_dec(prev_word), len(prev_word), axis = 0, force_tuple = True)
            hx_dec, cx_dec, xs_dec = self.nstep_dec(hx_dec, cx_dec, prev_word_emb, train = train)
        return result
    
class CharEnc(Chain):
    def __init__(self, V, Ec, H, nlayers = 1):
        super(CharEnc, self).__init__(
            c_emb = L.EmbedID(V, Ec),
            nstep_enc = L.NStepLSTM(nlayers, Ec, H, dropout = 0.5)
        )   
        self.nlayers = nlayers
        
    def compute_h(self, dataset_as_array, train = True, use_workaround = True):
        dataset_as_var = [Variable(a) for a in dataset_as_array]
        dataset_as_emb = [self.c_emb(v) for v in dataset_as_var]
        hx, cx, xs = self.nstep_enc(None, None, dataset_as_emb, train = train)
        _, last_layer = F.split_axis(hx, (self.nlayers-1,), axis = 0, force_tuple = True)
        if use_workaround:
            zeroes = [0 * F.sum(xx) for xx in xs]
            for z in zeroes:
                last_layer += F.broadcast_to(z.reshape(1,1,1), last_layer.shape)
        return last_layer
    
class CharEncDec(Chain):
    def __init__(self, V, Ec, H, nlayers_src = 1):
        super(CharEncDec, self).__init__(
            enc = CharEnc(V, Ec, H, nlayers=nlayers_src),
            dec = CharDec(V, Ec, H)
        )
    def compute_loss(self, dataset_as_array, use_gumbel = False, temperature = None, train = True, verbose = False, use_workaround = True):
        return self.dec.compute_loss(self.enc.compute_h(dataset_as_array, train = train, use_workaround = use_workaround), dataset_as_array,
                                    use_gumbel = use_gumbel, temperature = temperature, 
                                    train = train, verbose = verbose)
        
    def report_nan(self):
        for name, param in self.namedparams():
            if self.xp.any(self.xp.isnan(param.data)):
                print "nan in", name
            if param.grad is not None:
                if self.xp.any(self.xp.isnan(param.grad)):
                    print "nan in grad of", name
        
def make_data(filename, max_nb_ex = None):
    import codecs
    import itertools
    import collections
    
    words = collections.defaultdict(int)
    f = codecs.open(filename, encoding = "utf8")
    for line in itertools.islice(f, max_nb_ex):
        for w in line.strip().split(" "):
            words[w] += 1
            
    words = [w for w,cnt in words.iteritems() if cnt >= 2]
            
    print "collected", len(words), "words"
    
    charset = set()
    for w in words:
        for c in w:
            charset.add(c)
            
    print "size charset:", len(charset)
    
    charlist = sorted(charset)
    
    chardict = {}
    for num, c in enumerate(charlist):
        chardict[c] = num
        
    dataset = []
    for w in sorted(words):
        encoded = [chardict[c] for c in w]
        dataset.append(np.array(encoded, dtype = np.int32))
        
    return dataset, charlist, chardict
        

    
        
def test():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("nb_iters", type = int)
    parser.add_argument("--gpu", type = int)
    args = parser.parse_args()    

def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("filename")
    parser.add_argument("dest")
#     parser.add_argument("nb_iters", type = int)
    parser.add_argument("--gpu", type = int)
    parser.add_argument("--max_nb_ex", type = int)
    parser.add_argument("--mb_size", type = int, default = 256)
    parser.add_argument("--Ec", type = int, default = 32)
    parser.add_argument("--H", type = int, default = 64)
    parser.add_argument("--print_loss_every", type = int, default = 10)
    parser.add_argument("--max_nb_iters", type = int)
    parser.add_argument("--debug", default = False, action = "store_true")
    parser.add_argument("--src_layers", type = int, default = 1)
    parser.add_argument("--mode", default="train")
    args = parser.parse_args()
    
    if args.debug:
        chainer.set_debug(True)
    
    dataset, charlist, chardict = make_data(args.filename, max_nb_ex = args.max_nb_ex)
    
    json.dump(charlist, open(args.dest + "char_encdec.indexer", "w"))
    json.dump(args.__dict__, open(args.dest + "char_encdec.config", "w"))
    
    V = len(charlist)
    Ec = args.Ec
    H = args.H
    mb_size = args.mb_size
    
    ced = CharEncDec(V, Ec, H, nlayers_src=args.src_layers)
    
    if args.gpu is not None:
        chainer.cuda.Device(args.gpu).use()
        import cupy
        ced = ced.to_gpu(args.gpu)
        
        
    def minibatcher():
        import random
        dataset_shuffled = list(dataset)
        random.shuffle(dataset_shuffled)
        cursor = 0
        while 1:
            if cursor >= len(dataset_shuffled):
                cursor = 0
            batch = dataset_shuffled[cursor:cursor + mb_size]
            if args.gpu is not None:
                batch = [cupy.array(a, dtype = cupy.int32) for a in batch]
            yield batch
            cursor += mb_size
    
    mb_provider = minibatcher()
    
    optim = chainer.optimizers.Adam()
    optim.setup(ced)
    
    def train_once(print_loss = False, use_gumbel = False, temperature = 1, sample = False):
        dsa = mb_provider.next()
        if sample:
            try:
                hx = ced.enc.compute_h(dsa, train = False)
                decoded = ced.dec.decode(hx, length = 40, train = False)
                for num_inpt in range(len(dsa[:16])):
                    inpt_side = "".join([charlist[int(idx)] for idx in dsa[num_inpt]])
                    output_side = "".join([charlist[int(idx)] for idx in decoded[num_inpt]])
                    print inpt_side.encode("utf8"), "  ->   ", output_side.encode("utf8")
            except UnicodeEncodeError:
                pass
#         print "avant loss compute"
        optim.zero_grads()
        ced.report_nan()
        loss = ced.compute_loss(dsa, use_gumbel = use_gumbel, temperature = temperature, use_workaround = True)
        loss.backward()
        if print_loss:
            print loss.data
#         print "avant update"
#         ced.report_nan()  
        optim.update()
#         print "after update"
        ced.report_nan()  
        
    num_iter = 0
    while 1:
        if args.max_nb_iters is not None and num_iter >= args.max_nb_iters:
            break
        train_once(num_iter%args.print_loss_every== 0, use_gumbel = False, temperature = None, sample = num_iter%200 == 0)
        if num_iter%200 == 0:
            print "saving model at iteration", num_iter
            serializers.save_npz(args.dest + "char_encdec.model", ced)
        num_iter += 1

if __name__ == '__main__':
    main()
    
