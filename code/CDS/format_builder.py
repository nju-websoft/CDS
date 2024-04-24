import json
import os
import re
import csv
import sys
import jsonlines
import numpy as np
import pandas as pd
import shutil
import fcntl
import traceback
import xmltodict
from collections import defaultdict
from rdflib.plugins.shared.jsonld import util
from rdflib import Graph
from pebble import ProcessPool
import faiss
from pyserini.search.lucene import LuceneSearcher

graph = []
qid_list = []
triple_list = []    #edge_list
term_list = []  #label_list
cnt = 0     #term_count

entity_n3_dict = {}
edge_n3_dict = {}
term_2_kind = {}
term_2_label = {}
id_2_n3 = {}
entity_label_dict = {}
edge_label_dict = {}
str_id2label_dict = {}

edge_dict, entity_dict, literal_dict = {}, {}, {}
def build_dict(root, edge_set):
    global term_2_kind, term_2_label
    result = {}
    for kvp in edge_set[root]:
        key, value = kvp
        print(key, term_2_label[key], value, term_2_kind[value])
        if term_2_kind[value] == 0:
            result[term_2_label[key]] = build_dict(value, edge_set)
        else:
            result[term_2_label[key]] = term_2_label[value]
    return result

def add_term(label, kind): 
    global term_list, cnt
    global edge_dict, entity_dict, literal_dict

    if label is None or label == "":
        label = " "
    if len(label) > 200:
        label = label[:200]
    label = str(label).replace("\n", " ").replace("\t", " ").replace("\r", " ")

    if kind == 0: #entity
        if label == " ":
            cnt = cnt + 1
            node_id = cnt
            entity_dict[(" ", node_id)] = cnt
            term_list.append([cnt, label, kind])
        else:
            if label not in entity_dict:
                cnt = cnt + 1
                entity_dict[label] = cnt
                term_list.append([cnt, label, kind])
            node_id = entity_dict[label]
    elif kind == 1: #literal
        cnt = cnt + 1
        node_id = cnt
        term_list.append([node_id, label, kind])
    elif kind == 2: #edge
        if label not in edge_dict:
            cnt = cnt + 1
            edge_dict[label] = cnt
            term_list.append([cnt, label, kind])
        node_id = edge_dict[label]
    return node_id

def deal(file_name, dia):
    global qid_list, term_2_label
    for qid in qid_list:
        tmp_str = str(qid) + '_0.55'
        folder_path = f'/path/to/graph/{file_name}/{tmp_str}/UW/CBA+result/0'
        if file_name.endswith('.csv'):
            for k in range(2, 11):
                if k > 2 and k < 10:
                    continue
                header = []
                headers = []
                row = []
                data = []
                ans = []
                with open(os.path.join(folder_path, 'query_graph_' + str(dia) + '_' + str(k) + '.tsv'), 'r') as file:
                    for line in file:
                        triple = line.strip().split()
                        ans.append((int(triple[0]), int(triple[1]), int(triple[2])))
                        header.append(triple[1])
                        row.append(triple[0])
                header = list(set(header))
                row = list(set(row))
                row_dict = {key : {} for key in row}
                for triple in ans:
                    row_dict[str(triple[0])][triple[1]] = triple[2]
                for col in header:
                    headers.append(term_2_label[int(col)])
                for line in row:
                    tmp = []
                    for col in header:
                        tmp.append(term_2_label[row_dict[line][int(col)]])
                    data.append(tmp)
                with open(os.path.join(folder_path, 'query_result_' + str(dia) + '_' + str(k) + '.csv'), 'w') as file:
                    writer = csv.writer(file)
                    writer.writerow(headers)
                    writer.writerows(data)
                            
        elif file_name.endswith('.json'):
            for k in range(2, 11):
                if k != 5:
                    continue
                ans = []
                l_dict_set = set()
                r_dict_set = set()
                edge_set = defaultdict(list)
                with open(os.path.join(folder_path, 'query_graph_' + str(dia) + '_' + str(k) + '.tsv'), 'r') as file:
                    for line in file:
                        triple = line.strip().split()
                        ans.append((int(triple[0]), int(triple[1]), int(triple[2])))
                        edge_set[int(triple[0])].append((int(triple[1]), int(triple[2])))
                        if term_2_kind[int(triple[0])] == 0:
                            l_dict_set.add(int(triple[0]))
                        if term_2_kind[int(triple[2])] == 0:
                            r_dict_set.add(int(triple[2]))
                dset = l_dict_set - r_dict_set
                assert(len(dset) == 1)
                ans = build_dict(dset.pop(), edge_set)
                with open(os.path.join(folder_path, 'query_result_' + str(dia) + '_' + str(k) + '.json'), 'w') as file:
                    json.dump(ans, file, indent = 4)

        elif file_name.endswith('.rdf+xml'):
            for k in range(2, 11):
                ans = []
                with open(os.path.join(folder_path, 'query_graph_' + str(dia) + '_' + str(k) + '.tsv'), 'r') as file:
                    for line in file:
                        triple = line.strip().split()
                        ans.append((int(triple[0]), int(triple[1]), int(triple[2])))
                with open(os.path.join(folder_path, 'query_result_' + str(dia) + '_' + str(k) + '.nt'), 'w') as file:
                    for edge in ans:
                        sstr, pstr, ostr = '', '', ''
                        sstr = id_2_n3[edge[0]]
                        pstr = id_2_n3[edge[1]]
                        if term_2_kind[edge[2]] == 1:
                            ostr = str_id2label_dict[edge[2]]
                        else:
                            ostr = id_2_n3[edge[2]]
                        file.write(f"{sstr} {pstr} {ostr} .\n")

def initialize():
    global triple_list, term_list, cnt
    global edge_dict, entity_dict, literal_dict
    global entity_n3_dict, edge_n3_dict, term_2_kind, term_2_label, entity_label_dict, edge_label_dict, id_2_n3

    triple_list = []  
    term_list = []
    cnt = 0
    edge_dict.clear()
    entity_dict.clear()
    literal_dict.clear()
    
    entity_n3_dict.clear()
    edge_n3_dict.clear()
    term_2_kind.clear()
    term_2_label.clear()

    entity_label_dict.clear()
    edge_label_dict.clear()
    id_2_n3.clear()

def get_label(label):
    if label is None or label == "":
        label = " "
    if len(label) > 200:
        label = label[:200]
    label = str(label).replace("\n", " ").replace("\t", " ").replace("\r", " ")
    return label

def prepare(file_name, base_path):
    global entity_label_dict, edge_label_dict, term_2_kind, term_2_label
    initialize()

    terms_df = None
    triple_df = None
    term_file = os.path.join(base_path, 'term.tsv')
    if os.path.exists(term_file):
        terms_df = pd.read_csv(term_file, sep='\t', header=None, names=['term_id', 'label', 'kind'], dtype={'term_id': np.int64, 'label': str, 'kind': np.int16}, na_values=[], keep_default_na=False, quoting=3, engine='python')
    for index, row in terms_df.iterrows():
        term_2_kind[row['term_id']] = row['kind']
        term_2_label[row['term_id']] = row['label']
        if row['kind'] == 0:
            entity_label_dict[row['label']] = (row['term_id'])
        elif row['kind'] == 2:
            edge_label_dict[row['label']] = (row['term_id'])
        else:
            str_id2label_dict[row['term_id']] = get_label(row['label'])
    if file_name.endswith('.xml+rdf'):
        file_path = os.path.join('/path/to/data', file_name)
        g = Graph()
        g.parse(file_path, format='xml')
        for s, p, o in g:
            s_label = util.split_iri(s)[1]
            p_label = util.split_iri(p)[1]
            o_label = util.split_iri(o)[1]

            #print(s_label, p_label, o_label)
            s_label = get_label(s_label)
            p_label = get_label(p_label)
            s_id = entity_label_dict[s_label]
            id_2_n3[s_id] = s.n3()
            p_id = edge_label_dict[p_label]
            id_2_n3[p_id] = p.n3()
            if o_label is None:
                pass
            else:
                o_label = get_label(o_label)
                o_id = entity_label_dict[o_label]
                id_2_n3[o_id] = o.n3()

    return True

def run(file_name, base_path):
    global qid_list
    qid_list = []
    with open(os.path.join(base_path, 'query.txt'), 'r') as file:
        for line in file:
            qid = line.split('\t')[0]
            qid_list.append(qid)
    prepare(file_name, base_path)
    deal(file_name, 4)

if __name__ == "__main__":
        graph_path = '/path/to/graph'
        for filename in os.listdir(graph_path):
            base_path = os.path.expanduser(f'/path/to/graph/{filename}')
            try:
                run(filename, base_path)
                print(filename)
                sys.stdout.flush()
            except Exception as error:
                exit(0)