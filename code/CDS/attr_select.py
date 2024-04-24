import json
import os
import shutil
import re
import sys
import time
import pandas as pd
from multiprocessing import Process, Lock
from pebble import ProcessPool
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from collections import defaultdict

model_name = '/path/to/model/bge-large-en-v1.5/'
model = SentenceTransformer(model_name, device='cuda')
embedding_dim = 1024

lock = Lock()
terms_df = None
triple_df = None
entity_set = set()
title = None
qid_list = []
edge_set = defaultdict(list)

def do_faiss_lookup(index, query_text, top_k):
    embedding_q = np.reshape(model.encode(query_text), [1, embedding_dim])
    embedding_q = embedding_q.astype('float32') # let it be float32
    faiss.normalize_L2(embedding_q)
    matched_em, matched_indexes = index.search(embedding_q, top_k)
    return matched_em, matched_indexes

def read_file(file_path):
    nodes = []
    edges = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith('node:'):
            nodes.extend([int(node) for node in line.split(':')[1].split(',') if node.strip()])
        elif line.startswith('edge:'):
            edge_str = line.split(':')[1]
            edge_pairs = re.findall(r'\[(\d+),(\d+)\]', edge_str)
            edges.extend([(int(u), int(v)) for u, v in edge_pairs])

    return nodes, edges

def read_tid_nid(file_path):
    tID_2_nID = {}
    nID_2_tID = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        x, y = map(int, line.strip().split())
        tID_2_nID[x] = y
        nID_2_tID[y] = x

    return tID_2_nID, nID_2_tID

def deal(file_name, base_path):
    global qid_list, terms_df, triple_df, entity_set
    for qid in qid_list:
        tmp_str = str(qid) + '_0.55'
        folder_path = f'~/data/graph_acordar/{file_name}/{tmp_str}/UW/CBA+result/0'
        folder_path = os.path.expanduser(folder_path)
        new_base = os.path.expanduser(f'~/data/graph_acordar/{file_name}/{tmp_str}')
        tID_nID_file_path = os.path.join(new_base, 'tID_2_nID.txt')

        tID_2_nID, nID_2_tID = {}, {}
        tID_2_nID, nID_2_tID = read_tid_nid(tID_nID_file_path)

        edges_index_path = os.path.join(base_path, 'edges_bge-en-v1.5.index')
        index = faiss.read_index(edges_index_path)
        
        content = None #query
        with open(os.path.join(new_base, 'query.txt'), 'r') as file:
            content = file.read()


        matched_em, matched_indexes = do_faiss_lookup(index, str(content), top_k=100)
        index_to_em = {index: em for index, em in zip(matched_indexes[0], matched_em[0])}


        for dia in range(2, 7):
            if dia != 4:
                continue
            node_list, edge_list = [], []
            res_path = os.path.join(folder_path, 'diameter_' + str(dia) + '_result.txt')
            if os.path.exists(res_path):
                node_list, edge_list = read_file(res_path)
            else:
                print(f"File '{res_path}' not found.")
                return
            
            if len(node_list) == 1:
                core_node = node_list[0]
                with open(os.path.join(new_base, 'graph.txt'), 'r') as file:
                    lines = file.readlines()
                    for line in lines[1:]:
                        x, y = map(int, line.split())
                        if x == core_node:
                            node_list.append(y)
                            edge_list.extend([(int(x), int(y))])
                        elif y == core_node:
                            node_list.append(x)
                            edge_list.extend([(int(x), int(y))])

            ans = []
            tmp_edge = []
            edge_used = set()
            edge_sc_list = []
            edge_dict = {k: [] for k in range(1, 11)}

            for node in node_list:
                for edge in edge_set[nID_2_tID[node]]:
                    if edge[2] in entity_set:
                        nid_s = tID_2_nID[edge[0]]
                        nid_o = tID_2_nID[edge[2]]
                        if (nid_s, nid_o) in edge_list or (nid_o, nid_s) in edge_list:
                            ans.append((edge[0], edge[1], edge[2]))
                    else:
                        if edge[1] in index_to_em:
                            if edge[1] not in edge_used:
                                edge_used.add(edge[1])
                                edge_sc_list.append((edge[1], index_to_em[edge[1]]))
                            tmp_edge.append((edge[0], edge[1], edge[2]))
            sorted_list = sorted(edge_sc_list, key=lambda x: x[1], reverse=True)

            top_id = [x for x, y in sorted_list[:10]] 
            for edge in tmp_edge:
                if edge[1] in top_id:
                    edge_dict[top_id.index(edge[1]) + 1].append(edge)
            topk_edge = []

            for k in range(1, 11): # param here
                topk_edge.extend(edge_dict[k])
                if k == 5:
                    with open(os.path.join(folder_path, 'query_graph_' + str(dia) + '_' + str(k) + '.tsv'), 'w') as file:
                        for triple in ans:
                            file.write(f"{triple[0]} {triple[1]} {triple[2]}\n")
                        for triple in topk_edge:
                            file.write(f"{triple[0]} {triple[1]} {triple[2]}\n")
    return

def run(file_name, base_path)
    start_time = time.time()
    global qid_list, terms_df, triple_df, entity_set, edge_set
    terms_df = None
    triple_df = None
    entity_set = set()
    title = None
    qid_list = []
    edge_set = defaultdict(list)

    with open(os.path.join(base_path, 'query.txt'), 'r') as file:
        for line in file:
            qid = line.split('\t')[0]
            qid_list.append(qid)
            
    term_file = os.path.join(base_path, 'new_term.tsv')
    triple_file = os.path.join(base_path, 'triple.tsv')

    if os.path.exists(term_file):
        terms_df = pd.read_csv(term_file, sep='\t', header=None, names=['term_id', 'label', 'kind'], dtype={'term_id': np.int64, 'label': str, 'kind': np.int16}, na_values=[], keep_default_na=False, quoting=3, engine='python')
        entity_set.update(terms_df.loc[terms_df['kind'] == 0, 'term_id'].values)

    if os.path.exists(triple_file):
        triple_df = pd.read_csv(triple_file, sep='\t', header=None, names=['subject', 'predicate', 'object'], dtype={'subject': np.int64, 'predicate': np.int64, 'object': np.int64}, na_values=[], keep_default_na=False, quoting=3, engine='python')

    mask_subj_in_entity_set = triple_df['subject'].isin(entity_set)
    mask_obj_in_entity_set = triple_df['object'].isin(entity_set)
    mask_subj_less_than_obj = triple_df['subject'] < triple_df['object']

    for mask, group_key in [(mask_subj_in_entity_set & mask_obj_in_entity_set & mask_subj_less_than_obj, 'subject'),
                        (mask_subj_in_entity_set & mask_obj_in_entity_set & ~mask_subj_less_than_obj, 'object'),
                        (mask_subj_in_entity_set & ~mask_obj_in_entity_set, 'subject'),
                        (~mask_subj_in_entity_set & mask_obj_in_entity_set, 'object')]:
        new_edges = triple_df.loc[mask].groupby(group_key)[['subject', 'predicate', 'object']].apply(lambda x: x.values.tolist()).to_dict()
        for key, value in new_edges.items():
            if key not in edge_set:
                edge_set[key] = []
            edge_set[key].extend(value)
    end1_time = time.time()

    deal(file_name, base_path)
    end2_time = time.time()
    with lock:
        print(file_name + ' ' + str(end1_time - start_time) + ' ' + str(end2_time - end1_time))
        sys.stdout.flush()
    return

def task_done(future):
    try:
        result = future.result()
    except TimeoutError as error:
        print("Function took longer than %d seconds" % error.args[1])
    except Exception as error:
        print("Function raised %s" % error)
        print(error.traceback)

if __name__ == "__main__":
    graph_path = '/path/to/graph'
    cba_path = '/path/to/graph/CBA.txt'

    with open(cba_path, 'r') as f:
        cba_files = set(line.strip() for line in f)

    with ProcessPool(max_workers=10) as pool:
        for filename in os.listdir(graph_path):
            base_path = os.path.join(graph_path, filename)
            if filename in cba_files:
                try:
                    future = pool.schedule(run, (filename, base_path, ))
                    future.add_done_callback(task_done)
                except Exception as error:
                    exit(0)