import time
import os.path
import hashlib
from nmt_chainer.utilities.argument_parsing_tools import  OrderedNamespace

##########################################
# A function to compute the hash of a file
# Taken from http://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file
#


def hash_bytestr_iter(bytesiter, hasher, ashexstr=False):
    for block in bytesiter:
        hasher.update(block)
    return (hasher.hexdigest() if ashexstr else hasher.digest())

def file_as_blockiter(afile, blocksize=65536):
    with afile:
        block = afile.read(blocksize)
        while len(block) > 0:
            yield block
            block = afile.read(blocksize)

def compute_hash_of_file(filename):
    return hash_bytestr_iter(file_as_blockiter(open(filename, 'rb')), hashlib.sha256(), ashexstr = True)

def create_filename_infos(model_filename):
    model_infos = OrderedNamespace()
    model_infos["path"] = model_filename
    model_infos["last_modif"] = time.ctime(os.path.getmtime(model_filename))
    model_infos["hash"] = compute_hash_of_file(model_filename)
    return model_infos


