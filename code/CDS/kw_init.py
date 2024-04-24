import json
import os
import shutil
import re
import sys

import pandas as pd

from pebble import ProcessPool
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model_name = '/path/to/model/bge-large-en-v1.5/'
model = SentenceTransformer(model_name, device='cuda')
embedding_dim = 1024

base_path = '/path/to/graph'
words_list = []
tid_2_nid = {}

def do_faiss_lookup(index, query_text, top_k):
    embedding_q = np.reshape(model.encode(query_text), [1, embedding_dim])
    embedding_q = embedding_q.astype('float32')
    faiss.normalize_L2(embedding_q)
    matched_em, matched_indexes = index.search(embedding_q, top_k)
    return matched_em, matched_indexes

def run(folder_name, folder_path):
    global words_list, tid_2_nid
    words_list = []
    tid_2_nid = {}
    kw_map_list = {}

    query_list = []
    stop_words = []
    if not os.path.exists(os.path.join(folder_path, 'query.txt')):
        return
    with open(os.path.join(folder_path, 'query.txt'), 'r') as file:
        for line in file:
            items = line.strip().split('\t')
            str1 = items[0]
            str2 = items[1]
            list1 = items[2].split(' ')
            query_list.append((str1, str2, list1))
    with open('/code/nltk_stopword.txt', 'r') as f:
        stop_words = set(line.strip() for line in f)

    for triple in query_list:
        qid, did, words_list = triple
        source_folder = os.path.join(folder_path, "CBA")
        target_folder = os.path.join(folder_path, str(qid) + '_0.55')
        os.makedirs(target_folder, exist_ok=True)
        for filename in os.listdir(source_folder):
            source_file = os.path.join(source_folder, filename)
            target_file = os.path.join(target_folder, filename)
            shutil.copy2(source_file, target_file)

        words_list = [word for word in words_list if word not in stop_words]

        with open(os.path.join(target_folder, 'kwName.txt'), 'w') as file: # write kwName.txt
            for index, value in enumerate(words_list):
                file.write(f"{index} {value}\n")

        with open(os.path.join(target_folder, 'tID_2_nID.txt'), 'r') as file:
            for line in file:
                x, y = line.split()
                tid_2_nid[x] = y

        with open(os.path.join(target_folder, 'query.txt'), 'w') as file: # write query.txt
            qry_str = ' '.join(words_list)
            file.write(f"{qry_str}")

        entities_index_path = os.path.join(folder_path, 'entities_bge-en-v1.5.index')   
        index = faiss.read_index(entities_index_path)
        for cnt, query in enumerate(words_list): # search
            matched_em, matched_indexes = do_faiss_lookup(index, "Represent this sentence for searching relevant passages:" + str(query), top_k=5)  # entity
            filtered_indexes = matched_indexes[matched_em >= 0.55].tolist()
            if len(filtered_indexes) == 0:
                filtered_indexes = [matched_indexes.flatten()[0]]

            kw_map_list[cnt] = filtered_indexes

        with open(os.path.join(target_folder, 'kwMap.txt'), 'w') as file: # write kwName.txt
            for index in range(len(words_list)):
                file.write(f"{index}")
                for tid in kw_map_list[index]:
                    file.write(f"   {tid_2_nid[str(tid)]}")
                file.write(f"\n")
    print(folder_name)
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
    for filename in os.listdir(base_path):
        try:
            run(filename, os.path.join(base_path, filename))
        except Exception as error:
            exit(0)