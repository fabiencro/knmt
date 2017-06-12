from whoosh.qparser import QueryParser, OrGroup
from retriever import BaseEngine


class WhooshEngine(BaseEngine):
    def __init__(self, limit, index):
        self.__limit = limit
        self.__index = index

    def search(self, query):
        with self.__index.searcher() as searcher:
            query = QueryParser("X", self.__index.schema, group=OrGroup).parse(query)
            results = searcher.search(query, limit=self.__limit)
            return [(r["X"], r["Y"]) for r in results]
