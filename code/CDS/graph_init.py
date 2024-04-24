import json
import os
import re
import sys

import numpy as np
import pandas as pd
from pebble import ProcessPool

base_path = '/path/to/graph'
terms_df = None
triple_df = None
selected_dataset = []

def split_underline(label):
    pattern = r'^[a-zA-Z]+(?:_[a-zA-Z]+)*$'
    if re.match(pattern, label):
        return [part for part in label.split('_') if part]
    else:
        return None

def is_camel_case(s):
    return bool(re.match(r'^[a-zA-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)*$', s))

def split_camel_case(word):
    return re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', word)

def deal_label(label):
    underlineTokens = split_underline(label)
    if underlineTokens is not None:
        label = ' '.join(underlineTokens)
    if is_camel_case(label):
        label = ' '.join(split_camel_case(label))
    return label

def read(filename, filepath):
    global terms_df, triple_df
    terms_df = None
    triple_df = None

    term_file = os.path.join(filepath, 'term.tsv')
    triple_file = os.path.join(filepath, 'triple.tsv')
    if os.path.exists(term_file):
        terms_df = pd.read_csv(term_file, sep='\t', header=None, names=['term_id', 'label', 'kind'], dtype={'term_id': np.int64, 'label': str, 'kind': np.int16}, na_values=[], keep_default_na=False, quoting=3, engine='python')
        
    if os.path.exists(triple_file):
        triple_df = pd.read_csv(triple_file, sep='\t', header=None, names=['subject', 'predicate', 'object'], dtype={'subject': np.int64, 'predicate': np.int64, 'object': np.int64}, na_values=[], keep_default_na=False, quoting=3, engine='python')

def generate(filename, filepath):
    new_path = os.path.join(filepath, 'CBA')
    if not os.path.exists(new_path):
        os.mkdir(new_path)

    global terms_df, triple_df

    is_entity = terms_df['kind'] == 0
    entity_set = set(terms_df.loc[is_entity, 'term_id'])
    Tid_2_Nid = {term_id: i for i, term_id in enumerate(terms_df.loc[is_entity, 'term_id'])}
    entity_label_map = {i: deal_label(label) for i, label in enumerate(terms_df.loc[is_entity, 'label'])}
    node_num = len(Tid_2_Nid)

    valid_rows = triple_df['subject'].isin(entity_set) & triple_df['object'].isin(entity_set)
    graph = list(zip(triple_df.loc[valid_rows, 'subject'].map(Tid_2_Nid),
                     triple_df.loc[valid_rows, 'object'].map(Tid_2_Nid)))

    Node_file = os.path.join(new_path, 'nodeName.txt')
    with open(Node_file, 'w') as fw:
        for i in range(node_num):
            label = entity_label_map[i]
            if not label.strip():
                label = 'NULL' # caution here
            fw.write(f"{i} {label}\n")

    Graph_file = os.path.join(new_path, 'graph.txt')
    with open(Graph_file, 'w') as fw:
        fw.write(f"{node_num}\n")
        for Edge in graph:
            x, y = Edge
            fw.write(f"{x} {y}\n")

    Map_file = os.path.join(new_path, 'tID_2_nID.txt')
    with open(Map_file, 'w') as fw:
        for key, value in Tid_2_Nid.items():
            fw.write(f"{key} {value}\n")

    return True

def run(folder_name, folder_path):
    global cnt, terms_df, triple_df

    if os.path.isdir(folder_path):
        read(folder_name, folder_path)
        if terms_df is not None and triple_df is not None:
            try:
                flag = generate(folder_name, folder_path)
                assert(flag != None)
                print(folder_name)
            except:
                print(f'error: {folder_name}')
            sys.stdout.flush()
            
def task_done(future):
    try:
        result = future.result()
    except TimeoutError as error:
        print("Function took longer than %d seconds" % error.args[1])
    except Exception as error:
        print("Function raised %s" % error)
        print(error.traceback)

def prepare():
    global selected_dataset
    rpath = '/path/to/graph/graph_init_info.txt'
    with open(rpath, "r") as dataset:
        selected_dataset = [line.strip() for line in dataset]

if __name__ == "__main__":
    prepare()
    with ProcessPool(max_workers=15) as pool:
        for filename in os.listdir(base_path):
            if filename in selected_dataset:
                future = pool.schedule(run, (filename, os.path.join(base_path, filename), ))
                future.add_done_callback(task_done)