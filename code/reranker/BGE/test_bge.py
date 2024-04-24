from FlagEmbedding import FlagModel
import os
import json
import csv
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

data_path = "/path_to_data"
model_path = '/path_to_checkpoint'
rerank_path = '/path_to_baseline'

def cal_score(model, query, passages):
    res = []
    q_embeddings = model.encode_queries([query["question"]], batch_size=512)
    p_embeddings = model.encode([p["text"] for p in passages], batch_size=512)
    scores = q_embeddings @ p_embeddings.T
    # print(scores)
    for i in range(len(scores[0])):
        res.append((query['q_id'], passages[i], scores[0][i]))
    res.sort(key=lambda x:x[2], reverse=True)
    return res

with open(f'{model_path}/ntcir/BM25_metadata_top10_reranking.tsv', 'w+') as fout:
    print(f'ntcir metadata')
    with open(f'{rerank_path}/ntcir/metadata/BM25_top10_metadata_split_test.json', 'r') as fin:
        test_json = json.load(fin)
    model = FlagModel(f'{model_path}/ntcir/metadata', query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ") 
    for qp in tqdm(test_json):
        for res in cal_score(model, {"q_id": qp["q_id"], "question": qp["question"]}, qp["ctxs"]):
            fout.write(f'{res[0]}\t{int(res[1]["c_id"])}\t{res[2]}\n')

for diameter in [4]:
        for edge in [5]:
            for epoch in [3, 5, 10]:
                for lr in ['1e-6', '1e-5', '3e-5', '5e-5']:
                    for split in ['test', 'dev']:
                        extra = '_dev' if split == 'dev' else ''

                        for fold in [15]:
                            print(f'ntcir{fold} snippet_{diameter}_{edge}_{epoch}_{lr}')
                            with open(f'{model_path}/ntcir{fold}/BM25_data_cds_{diameter}_{edge}_top10_reranking{extra}_lr_{lr}_bs_2_epoch_{epoch}.tsv', 'w+') as fout:
                                with open(f'{rerank_path}/ntcir{fold}/data/BM25_top10_cds_{diameter}_{edge}_split_{split}.json', 'r') as fin:
                                    test_json = json.load(fin)
                                    model = FlagModel(f'{model_path}/ntcir{fold}/cds_{diameter}_{edge}/lr_{lr}_bs_2_epoch_{epoch}', query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ") 
                                for qp in tqdm(test_json):
                                    dataset_id_set = set()
                                    for res in cal_score(model, {"q_id": qp["q_id"], "question": "Represent this sentence for searching relevant passages: " + qp["question"]}, qp["ctxs"]):
                                        dataset_id = res[1]["c_id"].split('___')[0]
                                        if dataset_id in dataset_id_set:
                                            continue
                                        dataset_id_set.add(dataset_id)
                                        fout.write(f'{res[0]}\t{dataset_id}\t{res[2]}\n')
        