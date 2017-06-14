#!/usr/bin/env python

from whoosh.qparser import QueryParser, OrGroup
from retriever import BaseEngine


"""whoosh_engine.py: Implementation of search engine for Retriever"""
__author__ = "Ryota Nakao"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "nakario@gmail.com"
__status__ = "Development"


class WhooshEngine(BaseEngine):
    def __init__(self, limit, index):
        self.__limit = limit
        self.__index = index

    def search(self, query):
        with self.__index.searcher() as searcher:
            query = QueryParser("X", self.__index.schema, group=OrGroup).parse(query)
            results = searcher.search(query, limit=self.__limit)
            return [(r["X"], r["Y"]) for r in results]
