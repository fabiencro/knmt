from whoosh.index import create_in, open_dir, Index
from whoosh.fields import Schema, TEXT
import os


def create_index(index_path, x, y):
    schema = Schema(X=TEXT(stored=True), Y=TEXT(stored=True))
    if not os.path.exists(index_path):
        os.mkdir(index_path)
        ix = create_in(index_path, schema)
        writer = ix.writer()
        for (a, b) in zip(x, y):
            writer.add_document(X=a.strip(), Y=b.strip())
        writer.commit()
    else:
        ix = open_dir(index_path)
    return ix
