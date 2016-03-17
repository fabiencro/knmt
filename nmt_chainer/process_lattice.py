#!/usr/bin/env python
"""process_lattice.py: Use a RNNSearch Model"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import json
import numpy as np
from chainer import cuda, serializers, Variable

import models
from make_data import Indexer, build_dataset_one_side
# from utils import make_batch_src, make_batch_src_tgt, minibatch_provider, compute_bleu_with_unk_as_wrong, de_batch
from evaluation import (greedy_batch_translate, 
#                         convert_idx_to_string, 
                        batch_align, 
                        beam_search_translate, 
#                         convert_idx_to_string_with_attn
                        )

import visualisation

import logging
import codecs
# import h5py

logging.basicConfig()
log = logging.getLogger("rnns:lattice")
log.setLevel(logging.INFO)

from collections import defaultdict
import functools
import re, codecs, sys
import operator
import itertools, copy
# from __builtin__ import None
# import hypothesis
# from __builtin__ import list, False
# reload(hypothesis)
#from hypothesis import Hypothesis, AugmentedPattern, AdditionalsList, Additional

#logger = logging.getLogger()#"Decoder")

escape_dict = {"0":"", "\\":"\\", "n":"\n", "_":" ", "p":"|", "c":":"}
def de_escape(string):
    if "\\" not in string:
        return string
    res = []
    cursor = 0
    while(cursor < len(string)):
        c = string[cursor]
        if c!="\\":
            res.append(c)
        else:
            cursor += 1
            c = string[cursor]
            res.append( escape_dict[c])
        cursor += 1
    return "".join(res)

class Edge(object):
    def __init__(self):
        self.type = None
        self.v_start = None
        self.v_end = None
        
        self.word = None
        self.sublattice_id = None
#         
#         self.p_b_relationship = None
#         
#         self.impossible = False
#         self.features = {}

    def __hash__(self):
        return hash(self.type, self.v_start, self.v_end, self.word, self.sublattice_id)
    
    def __eq__(self, other):
        assert isinstance(other, Edge)
        return (self.type == other.type and 
                self.v_start == other.v_start and 
                self.v_end == other.v_end and 
                self.word == other.word and
                self.sublattice_id == other.sublattice_id)
        
    def __repr__(self):
        if self.type == "W":
            desc = self.word
        elif self.type == "B":
            desc = "%i"%self.sublattice_id
        else:
            desc = ""
        return "<%i -> %i [%s %s]>"%(self.v_start, self.v_end, self.type, desc)
    def copy(self):
        return copy.deepcopy(self)
    
def parse_edge_line(line, feature_names):
    edge = Edge()
    
    fields = line.split("|")
    start_v, end_v, type_edge = fields[0].split()
    edge.v_start = int(start_v)
    edge.v_end = int(end_v)
    edge.type = type_edge
    
    assert type_edge in "eEWB"
    
    if type_edge == "e":
        rest = fields[1:]
    elif type_edge == "W":
        edge.word = de_escape(fields[1])
        rest = fields[2:]
    elif type_edge == "B":
        type_bond, parent_relation, sublattice_id = fields[1].split("_")
        edge.sublattice_id = int(sublattice_id)
        rest = fields[2:]
    elif type_edge == "E":
        rest = fields[1:]
#         
#     if len(rest) == 3:
#         assert rest[2] == "i"
#         edge.impossible = True
#     else:
#         assert len(rest) == 2
#         
#     edge.p_b_relationship = rest[0]
#     
#     features = {}
#     for feature_as_string in rest[1].split():
#         feature_as_string_splitted = feature_as_string.split(":")
#         features[ feature_names[int(feature_as_string_splitted[0])] ] = float(feature_as_string_splitted[1])
#         
#     edge.features = features
    
    return edge

def parse_lattice_file(f):
    edges_per_lattice = None
    num_lattice = None
    for num_line, line in enumerate(f):
        line = line.strip()
        if num_line == 0:
            start_str, nb_lattices = line.split()
            assert start_str == "START_LATTICES"
            nb_lattices = int(nb_lattices)
            edges_per_lattice = [None] * nb_lattices
            continue
        elif num_line == 1:
            feature_names = line.split()
            continue
        elif line.startswith("END"):
            num_lattice = None
            continue
        elif line.startswith("BEGIN"):
            begin_str, num_lattice = line.split()
            assert begin_str == "BEGIN"
            num_lattice = int(num_lattice)
            edges_per_lattice[num_lattice] = []
        else:
            edge = parse_edge_line(line, feature_names)
            assert num_lattice is not None
            edges_per_lattice[num_lattice].append(edge)
            
    return edges_per_lattice
                
class Lattice(object):
    kInitial = 0
    kFinal = 1
    EOS = "<EOS>"
    EPSILON = "#$eps#"
    def __init__(self, edges):
        self.outgoing = defaultdict(list)
        for edge in edges:
            assert edge.v_start != edge.v_end
            self.outgoing[edge.v_start].append(edge)
    def __str__(self):
        res = []
        for v_start, edges in self.outgoing.iteritems():
            if v_start == Lattice.kFinal:
                assert len(edges) == 0
                continue
            res.append("v_start: %i"%v_start)
            for edge in edges:
                res.append(repr(edge))
        return "\n".join(res)
        
# import copy
# 
# class Path(object):
#     def __init__(self, path, level = 0):
#         assert isinstance(path, tuple)
#         assert len(path) == 0 or isinstance(path[0], tuple)
#         self.path = path
#         self.level = level
#         
#     def extremity(self):
#         return self.path[-1]
#     def prefix(self):
#         return self.path[:-1]
#     def copy(self):
#         res = Path(tuple(self.path), level = self.level)
#         return res
#     def add_prefix(self, prefix):
#         assert isinstance(prefix, tuple)
#         self.path = (prefix,) + self.path
#     def __repr__(self):
#         assert isinstance(self.path, tuple)
#         assert len(self.path) == 0 or isinstance(self.path[0], tuple)
#         if self.level == 0:
#             return "Path:%r"%(self.path,)
#         else:
#             return "Path(-%i):%r"%(self.level, self.path,)
#     
#     def __add__(self, other):
#         assert isinstance(other, Path)
#         assert len(other.path) > 0
#         assert len(self.path) > 0
#         if other.level > 0:
#             path_leveled = self.path[:-other.level]
#         else:
#             path_leveled = self.path
#         assert path_leveled[-1][0] == other.path[0][0]
#         path_joined = path_leveled[:-1] + other.path
#         return Path(path_joined)
#         
#     def reduce(self):
#         if len(self.path) > 0  and self.path[-1][1] == Lattice.kFinal:
#             self.path = self.path[:-1]
#             self.reduce()
#             
#     def is_empty(self):
#         return len(self.path) == 0
#     
#     def pop_last(self):
#         self.path = self.path[:-1]
#         
#         
# class PathList(object):
#     def count(self):
#         raise NotImplemented
#     def extremities(self):
#         raise NotImplemented
#     
# class PathListSimple(PathList):
#     def __init__(self):
#         self.lst = []
#         self.need_to_compute_length = True
#         self.need_to_compute_extremities = True
#         self.need_to_compute_extremities_with_prefix = True
#         self.length = None
#         self.extremities_ = None
#         self.extremities_with_prefix_ = None
#         
#     def append(self, other):
#         assert isinstance(other, PathList) or isinstance(other, Path)
#         self.lst.append(other)
#         self.need_to_compute_length = True
#         self.need_to_compute_extremities = True
#         self.need_to_compute_extremities_with_prefix = True
#         
#     def count(self):
#         if self.need_to_compute_length:
#             res = 0
#             for elem in self.lst:
#                 if isinstance(elem, PathList):
#                     res += elem.count()
#                 else:
#                     res += 1
#             self.length = res
#             self.need_to_compute_length = False
#         return self.length
#     
#     def contains_elem(self, path_list):
#         assert isinstance(path_list, PathList) or isinstance(path_list, Path)
#         return path_list in self.lst
#     
#     def extremities(self):
#         if self.need_to_compute_extremities:
#             res = set()
#             for elem in self.lst:
#                 if isinstance(elem, PathList):
#                     res |= elem.extremities()
#                 else:
#                     res.add(elem.extremity())
#             self.extremities_ = res
#             self.need_to_compute_extremities = False
#         return self.extremities_
#     
#     def extract_one(self):
#         assert len(self.lst) != 0
#         elem = self.lst[0]
#         if isinstance(elem, PathList):
#             return elem.extract_one()
#         else:
#             assert isinstance(elem, Path)
#             return elem
#     
#     def extremities_with_prefix(self):
#         if self.need_to_compute_extremities_with_prefix:
#             res = defaultdict(list)
#             for elem in self.lst:
#                 if isinstance(elem, PathList):
#                     subres = elem.extremities_with_prefix()
#                     for pos in subres:
#                         res[pos].append(subres[pos])
#                 else:
#                     res[elem.extremity()].append(elem.prefix())
#             self.extremities_with_prefix_ = res
#             self.need_to_compute_extremities_with_prefix = False
#         return self.extremities_with_prefix_
# 
# class FactoredPath(PathList):
#     def __init__(self, prefix, path_list):
#         assert isinstance(path_list, PathList)
#         self.prefix = prefix
#         self.path_list = path_list
#         self.extremities_with_prefix_ = None
#     def count(self):
#         return self.path_list.count()
#     def extremities(self):
#         return self.path_list.extremities()
#     def extremities_with_prefix(self):
#         if self.extremities_with_prefix_ is None:
#             res = {}
#             subres = self.path_list.extremities_with_prefix()
#             for pos, pref in subres.iteritems():
#                 res[pos] = [self.prefix, pref]
#             self.extremities_with_prefix_ = res
#         return self.extremities_with_prefix_
#     def extract_one(self):
#         suffix = self.path_list.extract_one()
#         res = suffix.copy()
#         res.add_prefix(self.prefix)
#         return res
# 
# class FactoredLevelPath(PathList):
#     def __init__(self, level, path_list):
#         assert isinstance(path_list, PathList)
#         self.level = level
#         self.path_list = path_list
#         self.extremities_with_prefix_ = None
#     def count(self):
#         return self.path_list.count()
#     def extremities(self):
#         return self.path_list.extremities()
#     def extremities_with_prefix(self):
#         if self.extremities_with_prefix_ is None:
#             res = {}
#             subres = self.path_list.extremities_with_prefix()
#             for pos, pref in subres.iteritems():
#                 res[pos] = [self.prefix, pref]
#             self.extremities_with_prefix_ = res
#         return self.extremities_with_prefix_
#     def extract_one(self):
#         suffix = self.path_list.extract_one()
#         res = suffix.copy()
#         res.level += self.level
#         return res
#       
# 
# def merge_sub(current_lattice, res, sub_result, v_end, memoizer, lattice_map):
#     for w, next_path_set in sub_result.iteritems():
#         assert w != Lattice.EPSILON
# #         if w == Lattice.EPSILON:
# #             continue
#         res[w].append(FactoredPath((current_lattice, v_end), next_path_set))
#         
#     if Lattice.EPSILON in sub_result:
#         treat_epsilon(res, current_lattice, v_end, memoizer, lattice_map)
# 
# def treat_epsilon(res, current_lattice, v_end, memoizer, lattice_map):
#     if v_end == Lattice.kFinal:
#         res[Lattice.EPSILON] = PathListSimple()
#     else:
#         next_pos = (current_lattice, v_end)
#         next_res = next_words_simple_pos2(next_pos, memoizer, lattice_map)
#         merge_in(res, next_res)
                    
# def merge_in(res1, res2):
# #     res1[w].append(res2)
#     for w, next_path_set in res2.iteritems():
#         if res1[w].contains_elem(next_path_set):
# #             print "double epsilon entry for", w
#             pass
#         else:
#             res1[w].append(next_path_set)
        
#         
# class IdGenerator(object):
#     def __init__(self):
#         self.id_ = 0
#     def get_new_id(self):
#         res = self.id_
#         self.id_ += 1
#         return res
        
class PosElem(object):
    def __init__(self, p, child_node = None):
        self.p = p
        self.child_node = child_node
    def is_leaf(self):
        return self.child_node is None
    def is_final(self):
        return self.p == Lattice.kFinal and self.is_leaf()
    def is_empty(self):
        return (not self.is_leaf()) and self.child_node.is_empty_node()
    def make_childless(self):
        res = PosElem(self.p)
        return res
    def __hash__(self):
        if self.child_node is not None:
            raise NotImplemented
        else:
            return hash(self.p)
    def __eq__(self, other):
        return isinstance(other, PosElem) and self.p == other.p and self.child_node == other.child_node
    def __str__(self):
        res = "P%i"%self.p
        if self.child_node is not None:
            res += "<" + str(self.child_node) + ">"
        return res
    
        
class Node(object):
    def __init__(self, lattice_id):
        self.lattice_id = lattice_id
#         self.id_generator = IdGenerator()
        self.inner_lst = defaultdict(list)
        self.leaf_lst = set() #defaultdict(list)
#         for elem in pos_lst:
#             id_ = self.id_generator.get_new_id()
#             self.pos_lst[id_] = elem
# 
    def count_paths(self, global_count_memoizer):
        if id(self) in global_count_memoizer:
            return global_count_memoizer[id(self)]
        count = 0
        for elem in self.pos_iter():
            if elem.is_leaf():
                count += 1
            else:
                count += elem.child_node.count_paths(global_count_memoizer)
        global_count_memoizer[id(self)] = count
        return count

    def count_unique_leaves(self, counter = None, local_memoizer = None):
        if local_memoizer is None:
            local_memoizer = set()
        if id(self) in local_memoizer:
            return
        else:
            local_memoizer.add(id(self))
            
        if counter is None:
            counter = [0]
            
        for elem in self.pos_iter():
            if elem.is_leaf():
                counter[0] += 1
            else:
                elem.child_node.count_unique_leaves(counter, local_memoizer)
        return counter[0]        
        
    def count_distincts_subnodes(self, local_memoizer = None):
        if local_memoizer is None:
            local_memoizer = set()
        if id(self) in local_memoizer:
            return
        else:
            local_memoizer.add(id(self))
        for elem in list(self.pos_iter()):
            if elem.child_node is not None:
                elem.child_node.count_distincts_subnodes(local_memoizer)
        return len(local_memoizer)
        
        
    def __str__(self):
        res = "L:%i"%self.lattice_id
        pos_list = []
        for pos in self.pos_iter():
            pos_list.append(str(pos))
        res = res + "[" + ", ".join(pos_list) + "]"
        return res
    
    def is_empty_node(self):
        return len(list(self.pos_iter())) == 0

    def add_elem(self, elem):
#         id_ = self.id_generator.get_new_id()
#         self.pos_lst[id_] = elem
        if elem.is_leaf():
            self.leaf_lst.add(elem)
#             for existing in self.leaf_lst[elem.p]:
#                 if existing is None:
#                     continue
#                 assert existing.is_leaf()
#                 if existing.p == elem.p:
#                     return
#             self.leaf_lst[elem.p].append(elem)
        else:
            self.inner_lst[elem.p].append(elem)
                
#     def __hash__(self):
#         return hash((self.lattice_id, tuple(self.pos_lst)))
#     
#     def __eq__(self):
#         return isinstance(other, Node) and self.lattice_id == other.lattice_id and self.pos_lst == other.pos_lst
#     
#     def rehash_pos_lst(self):
#         new_set = set()
#         for 
        
    def pos_iter(self):
        return (x for x in itertools.chain(
                                        iter(self.leaf_lst), 
                                        itertools.chain(*self.inner_lst.itervalues())) if x is not None)
    
#     def replace(self, elem, new_elems):
#         assert elem.is_leaf()
#         assert isinstance(new_elems, Node)
#         self.remove(elem)
#         for new_e in new_elems.pos_iter():
#             self.add_elem(new_e)
          
    def replace_all_at_once(self, replace_list):
        new_leaf_lst = set()
        for elem in self.leaf_lst:
#             if elem in replace_list:
            for sub_elem in replace_list[elem].pos_iter():
                if sub_elem.is_leaf():
                    new_leaf_lst.add(sub_elem)
                else:
                    self.inner_lst[sub_elem.p].append(sub_elem)
#             else:
#                 new_leaf_lst.append(elem)
        self.leaf_lst = new_leaf_lst
#         for elem, new_elems in replace_list:
#             assert elem.is_leaf()
#             assert isinstance(new_elems, Node)
#             self.remove(elem)
#             for new_e in new_elems.pos_iter():
#                 self.add_elem(new_e)
            
#         found = False
#         for i in xrange(len(self.pos_lst[elem.p])):
#             existing = self.pos_lst[elem.p][i]
#             if existing is None:
#                 continue
#             if existing.is_leaf() and existing.p == elem.p:
#                 assert not found
#                 found = True
#                 self.pos_lst[elem.p][i] = new_elem
#         assert found
             
    def assert_is_reduced_and_consistent(self, local_memoizer = None):
        if local_memoizer is None:
            local_memoizer = set()
        if id(self) in local_memoizer:
            return
        else:
            local_memoizer.add(id(self))
        seen_p = set()
        for elem in self.pos_iter():
            assert not elem.is_final()
            assert not elem.is_empty()
            if elem.is_leaf():
                assert elem.p not in seen_p
                seen_p.add(elem.p)
            else:
                elem.child_node.assert_is_reduced_and_consistent(local_memoizer)
        
                
    def remove(self, elem):
        assert elem.is_leaf() or elem.is_empty()
        if elem.is_leaf():
            assert elem in self.leaf_lst
            self.leaf_lst.remove(elem)
#             pos_lst = self.leaf_lst[elem.p]
        else:
            found = False
            pos_lst = self.inner_lst[elem.p]
            for i in xrange(len(pos_lst)):
                existing = pos_lst[i]
                if existing is None:
                    continue
                if ( (elem.is_leaf() and (existing.is_leaf() and existing.p == elem.p)) or
                     (elem.is_empty() and (existing.is_empty() and existing.p == elem.p))):
                    assert not found
                    found = True
                    pos_lst[i] = None
                    if elem.is_empty():
                        break
            assert found        
        
    def get_next_w(self, lattice_map, global_memoizer, global_count_memoizer, local_memoizer = None):
        if local_memoizer is None:
            local_memoizer = {}
        if id(self) in local_memoizer:
            return local_memoizer[id(self)]
        else:
            res = defaultdict(lambda:defaultdict(int))
            for pos_elem in self.pos_iter():
                if pos_elem.is_leaf():
                    position = (self.lattice_id, pos_elem.p) 
                    next_result = next_words_simple_pos3(position, global_memoizer, lattice_map)
                    for w, sub_node in next_result.iteritems():
                        res[w][id(self)] += sub_node.count_unique_leaves() #res.get(w, 0) + sub_node.count_unique_leaves() #count_paths(global_count_memoizer)
                else:
                    sub_res = pos_elem.child_node.get_next_w(lattice_map, global_memoizer, global_count_memoizer, local_memoizer)
                    for w in sub_res:
#                         res[w] = res.get(w, 0) + sub_res[w]
                        res[w].update(sub_res[w])
#                     res |= pos_elem.child_node.get_next_w(lattice_map, global_memoizer, local_memoizer)
            local_memoizer[id(self)] = res
            return res
              
    def make_global_replace_list(self, w, lattice_map, global_memoizer, local_memoizer = None):
        if local_memoizer is None:
            local_memoizer = {}
        if id(self) in local_memoizer:
            return local_memoizer[id(self)]
        local_memoizer[id(self)] = None
        
        replace_list = {}
        for pos_elem in list(self.pos_iter()):
            if pos_elem.is_leaf():
                position = (self.lattice_id, pos_elem.p)  
                next_result = next_words_simple_pos3(position, global_memoizer, lattice_map).get(w, None)
#                 print "next_result", self, pos_elem, w, str(next_result)
                if next_result is not None:
#                     next_result = copy.deepcopy(next_result)
#                     self.replace(pos_elem, next_result)
                    assert pos_elem not in replace_list
                    replace_list[pos_elem] = next_result #.append((pos_elem, next_result))
                else:
                    self.remove(pos_elem)
            else:
                pos_elem.child_node.make_global_replace_list(w, lattice_map, global_memoizer, local_memoizer)
                
        local_memoizer[id(self)] = replace_list
        return local_memoizer
    
    def update_with_global_replace_list(self, global_replace_list, local_memoizer = None):
        if local_memoizer is None:
            local_memoizer = set()
        if id(self) in local_memoizer:
            return
        else:
            local_memoizer.add(id(self))
        
        for pos_elem in list(self.pos_iter()):
            if not pos_elem.is_leaf():
                pos_elem.child_node.update_with_global_replace_list(global_replace_list, local_memoizer)
        self.replace_all_at_once(global_replace_list[id(self)])
                             
    def update_better(self, w, lattice_map, global_memoizer):
        global_replace_list = self.make_global_replace_list(w, lattice_map, global_memoizer)
        global_replace_list = copy.deepcopy(global_replace_list)  
        self.update_with_global_replace_list(global_replace_list)
                  
    def update(self, w, lattice_map, global_memoizer, local_memoizer = None):
        if local_memoizer is None:
            local_memoizer = set()
        if id(self) in local_memoizer:
            return
        else:
            local_memoizer.add(id(self))
            
        replace_list = {}
        for pos_elem in list(self.pos_iter()):
            if pos_elem.is_leaf():
                position = (self.lattice_id, pos_elem.p)  
                next_result = next_words_simple_pos3(position, global_memoizer, lattice_map).get(w, None)
#                 print "next_result", self, pos_elem, w, str(next_result)
                if next_result is not None:
#                     next_result = copy.deepcopy(next_result)
#                     self.replace(pos_elem, next_result)
                    assert pos_elem not in replace_list
                    replace_list[pos_elem] = next_result #.append((pos_elem, next_result))
                else:
                    self.remove(pos_elem)
            else:
                pos_elem.child_node.update(w, lattice_map, global_memoizer, local_memoizer)
        replace_list = copy.deepcopy(replace_list)
        self.replace_all_at_once(replace_list)
#         for pos_elem, next_result in replace_list:
#             self.replace(pos_elem, next_result)
           
    def _transfer_final_pos(self, local_memoizer = None):
        if local_memoizer is None:
            local_memoizer = set()
        if id(self) in local_memoizer:
            return
        else:
            local_memoizer.add(id(self))
        for elem in list(self.pos_iter()):
            if elem.child_node is not None:
                elem.child_node._transfer_final_pos(local_memoizer)
                has_final_child = False
                for sub_elem in elem.child_node.pos_iter():
                    if sub_elem.is_final():
                        has_final_child = True
                if has_final_child:
                    self.add_elem(elem.make_childless())

    def _remove_final_pos(self, local_memoizer = None):
        if local_memoizer is None:
            local_memoizer = set()
        if id(self) in local_memoizer:
            return
        else:
            local_memoizer.add(id(self))
#         print "remove_final for", id(self)
        for elem in list(self.pos_iter()):
            if elem.is_final() or elem.is_empty():
#                 print "remove_final", elem
                self.remove(elem)
            elif not elem.is_leaf():
                elem.child_node._remove_final_pos(local_memoizer)
                if elem.is_empty():
#                     print "remove_final 2", elem
                    self.remove(elem)
                
    def reduce(self):
        self._transfer_final_pos()
        self._remove_final_pos()
                
            
def next_words_simple_pos3(position, memoizer, lattice_map):
    if position not in memoizer:
        current_lattice, current_vertex = position
#                 res = defaultdict(PathListSimple)
        res = defaultdict(lambda: Node(current_lattice))
        assert current_vertex != Lattice.kFinal
        for edge in lattice_map[current_lattice].outgoing[current_vertex]:
    #         new_pos = Path(self.stack[:-1] + ((current_lattice_id, edge.v_end),), self.lattice_map)
            assert edge.type != "e"
            if edge.type == "W":
#                 new_path = Path(((current_lattice, edge.v_end),))
                pos_elem = PosElem(edge.v_end)
                res[edge.word].add_elem(pos_elem)
#             elif edge.type == "e":
#                 treat_epsilon(res, current_lattice, edge.v_end, memoizer, lattice_map)
            elif edge.type == "B":
                pos_sublattice = (edge.sublattice_id, Lattice.kInitial)
                sub_result = next_words_simple_pos3(pos_sublattice, memoizer, lattice_map) #new_pos.next_words_simple(memoizer)
#                 merge_sub(current_lattice, res, sub_result, edge.v_end, memoizer, lattice_map)
                for w, next_node in sub_result.iteritems():
                    pos_elem = PosElem(edge.v_end, next_node)
                    res[w].add_elem(pos_elem)
            elif edge.type == "E":
                assert edge.v_end == Lattice.kFinal
#                 new_path = Path(((current_lattice, edge.v_end),))
                pos_elem = PosElem(edge.v_end)
                res[Lattice.EOS].add_elem(pos_elem) #append(new_path)
            else:
                assert False
        assert len(res) > 0, position
        memoizer[position] = res
#         print "updated position", position, len(res), len(memoizer)
    #     print "updated memoizer", len(memoizer), lattice_num, vertex
    return memoizer[position]
       
# def next_words_simple_pos2(position, memoizer, lattice_map):
#     if position not in memoizer:
#         res = defaultdict(PathListSimple)
#         current_lattice, current_vertex = position
#         assert current_vertex != Lattice.kFinal
#         for edge in lattice_map[current_lattice].outgoing[current_vertex]:
#     #         new_pos = Path(self.stack[:-1] + ((current_lattice_id, edge.v_end),), self.lattice_map)
#             assert edge.type != "e"
#             if edge.type == "W":
#                 new_path = Path(((current_lattice, edge.v_end),))
#                 res[edge.word].append(new_path)
# #             elif edge.type == "e":
# #                 treat_epsilon(res, current_lattice, edge.v_end, memoizer, lattice_map)
#             elif edge.type == "B":
#                 pos_sublattice = (edge.sublattice_id, Lattice.kInitial)
#                 sub_result = next_words_simple_pos2(pos_sublattice, memoizer, lattice_map) #new_pos.next_words_simple(memoizer)
#                 merge_sub(current_lattice, res, sub_result, edge.v_end, memoizer, lattice_map)
#             elif edge.type == "E":
#                 assert edge.v_end == Lattice.kFinal
#                 new_path = Path(((current_lattice, edge.v_end),))
#                 res[Lattice.EOS].append(new_path)
#             else:
#                 assert False
#         assert len(res) > 0, position
#         memoizer[position] = res
#         print "updated position", position, len(res), len(memoizer)
#     #     print "updated memoizer", len(memoizer), lattice_num, vertex
#     return memoizer[position]
        
def build_incoming(lattice):
    incoming = defaultdict(list)
    for v_start in lattice.outgoing:
        for edge in lattice.outgoing[v_start]:
            incoming[edge.v_end].append(edge)
    return incoming

def build_topo_order(outgoing, start = Lattice.kInitial):
    topo_sorted = []
    visited = set()
    processed = set()
    def dfs(vertex):
        visited.add(vertex)
        for edge in outgoing[vertex]:
            v_end = edge.v_end
            assert (v_end in visited) == (v_end in processed)
            if v_end not in processed:
                dfs(v_end)
        topo_sorted.append(vertex)
        processed.add(vertex)
    dfs(start)
    return topo_sorted

def remove_epsilon(lattice, epsilonpotent_lattices = None, empty_lattices = None):
    
    if epsilonpotent_lattices is None:
        epsilonpotent_lattices = set()
    if empty_lattices is None:
        empty_lattices = set()
    
    topo_order = build_topo_order(lattice.outgoing)
    incoming = build_incoming(lattice)
    is_epsilonpotent = False
    
    def edge_is_epsilonpotent(edge):
        if edge.type == "e":
            return True
        if edge.type == "B":
            return edge.sublattice_id in epsilonpotent_lattices
        return False
    
    for current_v in topo_order:
        v_end_with_epsilon = set()
        for edge in lattice.outgoing[current_v]:
            assert edge.v_start == current_v
            if edge_is_epsilonpotent(edge):
                v_end_with_epsilon.add(edge.v_end)
        
        if len(v_end_with_epsilon) > 0:
            new_outgoing = []
            for edge in lattice.outgoing[current_v]:
                assert edge.v_start == current_v
                if edge.type != "e":
                    if edge.type != "B" or edge.sublattice_id not in empty_lattices:
                        new_outgoing.append(edge)
            lattice.outgoing[current_v] = new_outgoing
                    
        for v_end in v_end_with_epsilon:
            if current_v != Lattice.kInitial:
                for edge in incoming[current_v]:
                    assert edge.v_end == current_v
                    duplicated_edge = edge.copy()
                    duplicated_edge.v_end = v_end
                    lattice.outgoing[edge.v_start].append(duplicated_edge)
            else:
                if v_end == Lattice.kFinal:
                    is_epsilonpotent = True
                for edge in lattice.outgoing[v_end]:
                    duplicated_edge = edge.copy()
                    duplicated_edge.v_start = Lattice.kInitial
                    lattice.outgoing[Lattice.kInitial].append(duplicated_edge)
                    
        if len(lattice.outgoing[current_v]) == 0 and current_v != Lattice.kFinal:
            previous_v = set()
            for edge in incoming[current_v]:
                previous_v.add(edge.v_start)
            for pv in previous_v:
                new_outgoing = []
                for edge in lattice.outgoing[pv]:
                    if edge.v_end != current_v:
                        new_outgoing.append(edge)
                lattice.outgoing[pv] = new_outgoing
    
    if len(lattice.outgoing[Lattice.kInitial]) > 0:
        topo_order_check = build_topo_order(lattice.outgoing)
        assert topo_order_check[0] == Lattice.kFinal
        assert topo_order_check[-1] == Lattice.kInitial
    return is_epsilonpotent
                    
def remove_unreachable(lattice):
    reachable = set(build_topo_order(lattice.outgoing))
    for v in lattice.outgoing.keys():
        if v not in reachable:
            del lattice.outgoing[v]

def remove_all_epsilons(lattice_map):
    epsilonpotent_lattices = set()
    empty_lattices = set()
    for num_lattice, lattice in enumerate(lattice_map):
        is_epsilonpotent = remove_epsilon(lattice, epsilonpotent_lattices, empty_lattices)
        if is_epsilonpotent:
            epsilonpotent_lattices.add(num_lattice)
        if len(lattice.outgoing[Lattice.kInitial]) == 0:
            empty_lattices.add(num_lattice)
    top_lattice_id = len(lattice_map) - 1
    return top_lattice_id in epsilonpotent_lattices
  
def command_line2():
    import argparse
    parser = argparse.ArgumentParser(description= "Use a RNNSearch model", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("lattice_fn")
    parser.add_argument("source_sentence_fn")
    parser.add_argument("training_config", help = "prefix of the trained model")
    parser.add_argument("trained_model", help = "prefix of the trained model")
    
    parser.add_argument("--gpu", type = int, help = "specify gpu number to use, if any")
    parser.add_argument("--skip_in_src", type = int, default = 0)
    args = parser.parse_args()
    
    config_training_fn = args.training_config #args.model_prefix + ".train.config"
    
    log.info("loading model config from %s" % config_training_fn)
    config_training = json.load(open(config_training_fn))

    voc_fn = config_training["voc"]
    log.info("loading voc from %s"% voc_fn)
    src_voc, tgt_voc = json.load(open(voc_fn))
    
    src_indexer = Indexer.make_from_serializable(src_voc)
    tgt_indexer = Indexer.make_from_serializable(tgt_voc)
    tgt_voc = None
    src_voc = None
    
    
#     Vi = len(src_voc) + 1 # + UNK
#     Vo = len(tgt_voc) + 1 # + UNK
    
    Vi = len(src_indexer) # + UNK
    Vo = len(tgt_indexer) # + UNK
    
    print config_training
    
    Ei = config_training["command_line"]["Ei"]
    Hi = config_training["command_line"]["Hi"]
    Eo = config_training["command_line"]["Eo"]
    Ho = config_training["command_line"]["Ho"]
    Ha = config_training["command_line"]["Ha"]
    Hl = config_training["command_line"]["Hl"]
    
    eos_idx = Vo
    encdec = models.EncoderDecoder(Vi, Ei, Hi, Vo + 1, Eo, Ho, Ha, Hl)
    
    log.info("loading model from %s" % args.trained_model)
    serializers.load_npz(args.trained_model, encdec)
    
    if args.gpu is not None:
        encdec = encdec.to_gpu(args.gpu)
       
    src_sent_f = codecs.open(args.source_sentence_fn, encoding = "utf8")
    for _ in xrange(args.skip_in_src):
        src_sent_f.readline()
    src_sentence = src_sent_f.readline().strip().split(" ")
    log.info("translating sentence %s"%(" ".join(src_sentence)))
    src_seq = src_indexer.convert(src_sentence)
    log.info("src seq: %r"%src_seq)
    
    log.info( "loading lattice %s"%args.lattice_fn)
    lattice_f = codecs.open(args.lattice_fn, "r", encoding = "utf8")
    all_edges = parse_lattice_file(lattice_f)
    log.info("loaded")
    
    lattice_map = [None] * len(all_edges)
    for num_lattice, edge_list in enumerate(all_edges):
        lattice_map[num_lattice] = Lattice(edge_list)
        top_lattice_id = num_lattice
        
    log.info("built lattices")
    
    log.info("removing epsilons")
    log.info("nb edges before %i"%sum(
                        len(edge_list)  for lattice in lattice_map for edge_list in lattice.outgoing.itervalues()))
    remove_all_epsilons(lattice_map)
    log.info("nb edges before %i"%sum(
                        len(edge_list)  for lattice in lattice_map for edge_list in lattice.outgoing.itervalues()))
    
    
    if args.gpu is not None:
        seq_as_batch = [Variable(cuda.to_gpu(np.array([x], dtype = np.int32), args.gpu), volatile = "on") for x in src_seq]
    else:
        seq_as_batch = [Variable(np.array([x], dtype = np.int32), volatile = "on") for x in src_seq]
    predictor = encdec.get_predictor(seq_as_batch, [])
    
    global_memoizer = {}
    global_count_memoizer = {}
    initial_node = Node(top_lattice_id)
    initial_node.add_elem(PosElem(Lattice.kInitial))
    current_path = initial_node
    selected_seq = []
    while 1:
        print "#node current_path", current_path.count_distincts_subnodes()
        current_path.assert_is_reduced_and_consistent()
        next_words_set = current_path.get_next_w(lattice_map, global_memoizer, global_count_memoizer)
        for w in next_words_set:
            next_words_set[w] = sum(next_words_set[w].itervalues())
        has_eos = Lattice.EOS in next_words_set
        next_words_list = sorted(list(w for w in next_words_set if w != Lattice.EOS))
        print "next_words_set", next_words_set
        voc_choice = tgt_indexer.convert(next_words_list)
        if has_eos:
            voc_choice.append(eos_idx)
        chosen = predictor(voc_choice)
        
        if chosen != eos_idx and tgt_indexer.is_unk_idx(chosen):
            print "warning: unk chosen"
            unk_list = []
            for ix, t_idx in enumerate(voc_choice):
                if tgt_indexer.is_unk_idx(t_idx):
                    unk_list.append((next_words_set[next_words_list[ix]], next_words_list[ix]))
            unk_list.sort(reverse = True)
            print "UNK:", unk_list
            selected_w = unk_list[0][1]
        else:
            idx_chosen = voc_choice.index(chosen) #TODO: better handling when several tgt candidates map to UNK
            
            selected_w = (next_words_list + [Lattice.EOS])[idx_chosen]
        
#         for num_word, word in enumerate(next_words_list):
#             print num_word, word
#         print "selected_seq", selected_seq 
#         i = int(raw_input("choice\n"))
#         selected_w = next_words_list[i]
#         
        
        selected_seq.append(selected_w)
        print "selected_seq", selected_seq 
        
        current_path.update_better(selected_w, lattice_map, global_memoizer)
        current_path.reduce()
        if current_path.is_empty_node():
            print "DONE"
            break
    print "final seq:", selected_seq
    
    
def build_word_tree(lattice_map, top_lattice_id):  
    global_memoizer = {}
    global_count_memoizer = {}
    initial_node = Node(top_lattice_id)
    initial_node.add_elem(PosElem(Lattice.kInitial))
    
    
    def build_word_tree_rec(current_path):
        print "bwt", repr(current_path), str(current_path), current_path.count_distincts_subnodes()
        current_path.assert_is_reduced_and_consistent()
        next_words_set = current_path.get_next_w(lattice_map, global_count_memoizer, global_memoizer)
        print next_words_set
        res = []
        for w in next_words_set:
            current_path_copy = copy.deepcopy(current_path)
#             print str(current_path_copy), current_path_copy.count_distincts_subnodes()
            current_path_copy.update_better(w, lattice_map, global_memoizer)
#             print " -> ", w, " ->", str(current_path_copy),
            current_path_copy.reduce()
#             print " -> reduced ->", str(current_path_copy)
            if not current_path_copy.is_empty_node():
                sub_word_tree = build_word_tree_rec(current_path_copy)
                res.append({0:w, 1:sub_word_tree})
            else:
                res.append(w)
        return res
       
    res = build_word_tree_rec(initial_node)
    return res
  
def commandline():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("lattice_fn")
    
    args = parser.parse_args()
    
    lattice_f = codecs.open(args.lattice_fn, "r", encoding = "utf8")
    
    print "loading", args.lattice_fn
    all_edges = parse_lattice_file(lattice_f)
    
    print "loaded"
    
    lattice_map = [None] * len(all_edges)
    for num_lattice, edge_list in enumerate(all_edges):
        lattice_map[num_lattice] = Lattice(edge_list)
        top_lattice_id = num_lattice
        
    print "built lattices"
    
    print "removing epsilons"
    print "nb edges before", sum(len(edge_list)  for lattice in lattice_map for edge_list in lattice.outgoing.itervalues())
    remove_all_epsilons(lattice_map)
    print "nb edges after", sum(len(edge_list)  for lattice in lattice_map for edge_list in lattice.outgoing.itervalues())
    
#     pos0 = Path.make_initial(top_lattice_id, lattice_map)
#     
#     print pos0
#     
#     if 0:
#         memoize_simple = {}
#         wordset = next_words_simple(top_lattice_id, Lattice.kInitial, memoize_simple)
#         print wordset
#         sys.exit(0)
        
    if 1:
        global_memoizer = {}
        initial_node = Node(top_lattice_id)
        initial_node.add_elem(PosElem(Lattice.kInitial))
        current_path = initial_node
        selected_seq = []
        while 1:
            current_path.assert_is_reduced_and_consistent()
            next_words_set = current_path.get_next_w(lattice_map, global_memoizer)
            next_words_list = sorted(list(next_words_set))
            for num_word, word in enumerate(next_words_list):
                print num_word, word
            print "selected_seq", selected_seq 
            i = int(raw_input("choice\n"))
            selected_w = next_words_list[i]
            selected_seq.append(selected_w)
            current_path.update(selected_w, lattice_map, global_memoizer)
            current_path.reduce()
            if current_path.is_empty_node():
                print "DONE"
                break
        print "final seq:", selected_seq
#     if 0:
# #         import operator
#         memoize_simple = {}
#         position = (top_lattice_id, Lattice.kInitial)
#         selected_seq = []
#         current_path = None
#         while 1:
#             print position
#             next_dict = next_words_simple_pos2(position, memoize_simple, lattice_map)
#             
#             
# #             if Lattice.EPSILON in next_dict:
# #                 next_dict = copy.deepcopy(next_dict)
# #                 dict_with_epsilon = next_dict
# #                 while Lattice.EPSILON in dict_with_epsilon:
# #                 
# #                 next_dict = copy.deepcopy(next_dict)
# #                 del next_dict[Lattice.EPSILON]
# #                 upper_pos, levels = current_path.return_popped_reduced_extremity()
# #                 upper_dict = next_words_simple_pos2(upper_pos, memoize_simple, lattice_map)
# #                 for upper_w, upper_next_pos in upper_dict.iteritems():
# #                     if upper_w in next_dict:
# #                         next_dict[upper_w].append(FactoredLevelPath(1, upper_next_pos))
# #                     else:
# #                         next_dict[upper_w] = FactoredLevelPath(1, upper_next_pos)
#             wmap = []
#             for idx, (w, next_pos) in enumerate(sorted(next_dict.items(), key = lambda x:x[1].count())):
#                 print idx, w, next_pos.count(), next_pos.extremities()
#                 print len(next_pos.extremities_with_prefix())
# #                 print next_pos.extract_one()
#                 print
#                 wmap.append(w)
#             print "selected_seq", selected_seq
#             print "current_path", current_path
#             i = int(raw_input("choice\n"))
# #             if wmap[i] == Lattice.EPSILON:
# #                 assert next_dict[wmap[i]].count() == 0
# #                 assert current_path is not None
# #                 current_path.pop_last()
# #             else:
#             subpath = next_dict[wmap[i]].extract_one()
#             if current_path is None:
#                 current_path = subpath.copy()
#             else:
#                 current_path = current_path + subpath
#             current_path.reduce()
#             if current_path.is_empty():
#                 print "DONE"
#                 break
#             position = current_path.extremity()
# #             if current_path is None:
# #                 subpath = subpath.copy()
# #             else:
#                 
# #             print next_dict[wmap[i]].extremities_with_prefix()[0]
# #             position = next(iter(next_dict[wmap[i]].extremities()))
#             selected_seq.append(wmap[i])
#         sys.exit(0)
#     if 0:   
#         memoize = {}
#         while 1:
#             print pos0
#             
#             nexts = list(pos0.next_words(memoize))
#             for i, next_pos in enumerate(nexts):
#                 print i, next_pos[0]
#             i = int(raw_input("choice"))
#             pos0 = nexts[i][1]
         
def test1():
#     latt_desc = """START_LATTICES 1
# NULL_content
# BEGIN 0
# 0 2 W|DERS
# 3 1 W|power
# 2 3 e
# 2 3 W|for
# END""".split("\n")


    latt_desc = """START_LATTICES 6
NULL_content
BEGIN 0
0 2 e
2 1 W|DERS
2 1 e
END
BEGIN 1
0 2 B|A_B_0
0 3 B|A_B_0
3 4 W|power
2 3 e
2 3 W|for
4 1 e
END
BEGIN 2
0 2 W|the
2 1 B|A_B_1
0 1 B|A_B_0
BEGIN 3
0 2 W|a
2 1 B|A_B_1
BEGIN 4
0 2 W|one
2 1 B|A_B_3
0 3 W|two
3 1 B|A_B_2
BEGIN 5
0 2 W|the
2 1 B|A_B_4
END""".split("\n")

    all_edges = parse_lattice_file(latt_desc)
    
    print "loaded"
    
    lattice_map = [None] * len(all_edges)
    for num_lattice, edge_list in enumerate(all_edges):
        lattice_map[num_lattice] = Lattice(edge_list)
        top_lattice_id = num_lattice
        
    print "built lattices"
    
    print "removing epsilons"
    print "nb edges before", sum(len(edge_list)  for lattice in lattice_map for edge_list in lattice.outgoing.itervalues())
    is_epsilonpotent = remove_all_epsilons(lattice_map)
    print "is_epsilonpotent", is_epsilonpotent
    print "nb edges after", sum(len(edge_list)  for lattice in lattice_map for edge_list in lattice.outgoing.itervalues())
    for num_lattice, lattice in enumerate(lattice_map):
        remove_unreachable(lattice)
    print "nb edges after removing unreachable", sum(len(edge_list)  for lattice in lattice_map for edge_list in lattice.outgoing.itervalues())

    for num_lattice, lattice in enumerate(lattice_map):
        print "L:", num_lattice
        print str(lattice)
        
    word_tree = build_word_tree(lattice_map, top_lattice_id)
    import pprint
    pp = pprint.PrettyPrinter(indent = 2)
    print json.dumps(word_tree, indent = 4)
         
if __name__ == '__main__':
#     test1()
    command_line2()
#     import cProfile
#     cProfile.run("commandline()")
#     commandline()
    