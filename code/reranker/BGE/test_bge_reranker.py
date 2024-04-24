from FlagEmbedding import FlagReranker
import os
import json
import csv
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

data_path = "/path_to_data"
model_path = '/path_to_checkpoint'
rerank_path = '/path_to_baseline'

def cal_score(reranker, query, passages):
    res = []
    scores = reranker.compute_score([[query["question"], p['text']] for p in passages], batch_size=512)
    # print(scores)
    for i in range(len(scores)):
        res.append((query['q_id'], passages[i], scores[i]))
    res.sort(key=lambda x:x[2], reverse=True)
    return res

# metadata
with open(f'{model_path}/ntcir/BM25_metadata_top10_reranker_reranking.tsv', 'w+') as fout:
    print(f'ntcir metadata')
    with open(f'{rerank_path}/ntcir/metadata/BM25_top10_metadata_split_test.json', 'r') as fin:
        test_json = json.load(fin)
    reranker = FlagReranker(f'{model_path}/ntcir/metadata_reranker/lr_1e-5_bs_2', use_fp16=True) #use fp16 can speed up computingfor qp in tqdm(test_json):
    for qp in tqdm(test_json):
        for res in cal_score(reranker, {"q_id": qp["q_id"], "question": qp["question"]}, qp["ctxs"]):
            fout.write(f'{res[0]}\t{int(res[1]["c_id"])}\t{res[2]}\n')

# data

for diameter in [3,4]:
        for edge in [5]:
            for epoch in [3, 5, 10]:
                for lr in ['1e-6', '1e-5', '3e-5', '5e-5']:
                    for split in ['test', 'dev']:
                        extra = '_dev' if split == 'dev' else ''

                        for fold in [15]:
                            print(f'ntcir{fold} snippet_{diameter}_{edge}_{epoch}_{lr}')
                            with open(f'{model_path}/ntcir{fold}/BM25_data_cds_{diameter}_{edge}_top10_reranker_reranking{extra}_lr_{lr}_bs_2_epoch_{epoch}.tsv', 'w+') as fout:
                                with open(f'{rerank_path}/ntcir{fold}/data/BM25_top10_cds_{diameter}_{edge}_split_{split}.json', 'r') as fin:
                                    test_json = json.load(fin)
                                    reranker = FlagReranker(f'{model_path}/ntcir{fold}/cds_{diameter}_{edge}_reranker/lr_{lr}_bs_2_epoch_{epoch}', use_fp16=True) #use fp16 can speed up computingfor qp in tqdm(test_json):
                                for qp in tqdm(test_json):
                                    dataset_id_set = set()
                                    for res in cal_score(reranker, {"q_id": qp["q_id"], "question": qp["question"]}, qp["ctxs"]):
                                        dataset_id = res[1]["c_id"].split('___')[0]
                                        if dataset_id in dataset_id_set:
                                            continue
                                        dataset_id_set.add(dataset_id)
                                        fout.write(f'{res[0]}\t{dataset_id}\t{res[2]}\n')           

        