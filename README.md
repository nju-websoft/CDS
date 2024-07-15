# CDS

This is the source code and data of the paper "Enhancing Dataset Search with Compact Data Snippets" SIGIR 2024.

## Directory Structure

Directory `./code` contains all the source code based on Python 3.10 and JDK 11.

+ Directory `./code/CBA` contains code of the CBA algorithm which is aim to efficiently extract a subgraph as the skeleton of a data snippet.
  For the instructions of this part of the code, please refer to [CBA](https://github.com/nju-websoft/CBA).

+ Directory `./code/CDS` contains code for the other parts of our work.
  The code consists of the following sections:
  + `graph_init.py` reads from data graph which connecting entities associated with attributes and  makes preparations for the execution of the CBA algorithm.
  + `HBLL.sh` builds HBLL for the CBA algorithm.
  + `kw_init.py`  takes each word in the query as a separate group, and selects the entity nodes that meet the correlation requirements as nodes in this group according to the pre-constructed index.
  + `CBA_runner.sh` runs CBA algorithm.
  + `attr_select.py` selects attributes most relevant to the query based on the attribute index.
  + `format_builder.py` reverts unified data graphs to their original formats.

+ Directory `./code/reranker` contains code for dataset reranking including [monoBERT](https://huggingface.co/castorini/monobert-large-msmarco), [BGE](https://huggingface.co/BAAI/bge-large-en-v1.5), and [BGE-reranker](https://huggingface.co/BAAI/bge-reranker-large).

+ File `./nltk_stopword.txt` is a collection of common stop words provided by NLTK to filter queries.

## Get Started

In our experiment, each data consists of `term.tsv`  and `triple.tsv`, and you need to pre-build the BGE indexes `entities_bge-en-v1.5.index` and `edges_bge-en-v1.5.index` based on them.

The parameter **diameter bound** is set to 4 and **attribute number bound** is set to 5 in this code.

Use following command to prepare for the execution of CBA algorithm:

```
python graph_init.py
bash ./HBLL.sh
```

Then `graph.txt`,`nodeName.txt`,` tID_2_nID.txt` and`UWHBLL.txt` will be generated.

Use following command to generate a separate folder for each Query-Data pair with previously generated data along with `kwMap.txt`, `kwName.txt` and `query.txt`:

```
python kw_init.py
```

The command for running CBA alorithm is similar:

```
bash ./CBA_runner.sh
```

Use following command to filter the attributes with the support of the attribute index:

```
python attr_select.py
```

Then `query_graph_4_5.tsv` will be generated, which is the unified graph format data snippet in our paper.

If you want to convert the snippet above to its original format, run the following code：

```
python format_builder.py
```

Then data snippet will be generated in the original format named  `query_result_4_5.tsv` .

The complete structure of each data is as follows:

```
├─data
│  │  edges_bge-en-v1.5.index
│  │  entities_bge-en-v1.5.index
│  │  query.txt
│  │  term.tsv
│  │  triple.tsv
│  │
│  ├─CBA
│  │      graph.txt
│  │      nodeName.txt
│  │      tID_2_nID.txt
│  │      UWHBLL.txt
│  │
│  └─QID_0.55
│      │  graph.txt
│      │  kwMap.txt
│      │  kwName.txt
│      │  nodeName.txt
│      │  query.txt
│      │  tID_2_nID.txt
│      │  UWHBLL.txt
│      │
│      └─UW
│          └─CBA+result
│              └─0
│                      diameter_4_result.txt
│                      query_graph_4_5.tsv
```

## Evaluation

All the results for reranking experiments are at `./results`. The result files are named as `{test_collection}_{reranking_model}_{snippet_extraction_method}_{topk}_{normalization_method}_{fusion_method}.txt` and in TREC format. For example, `ntcir_bge_cds_top10_min-max_mixed.txt` means the reranking results of bge with CDS as snippet extraction method, min-max normalization as normalization method, and mixed fusion method.

```
DS1-E-1001 Q0 7629c1d5-5da8-45b5-bc8b-58483f97921a 1 1.1486173567566846 mixed
DS1-E-1001 Q0 f8cfaa69-3f89-4ebe-96e2-d15a30173f43 2 1.0281533671470748 mixed
DS1-E-1001 Q0 adaf0ce0-1064-4f55-9397-df553bd1ef75 3 0.925949423187341 mixed
```

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Citation

```
@inproceedings{CDS,
  author       = {Qiaosheng Chen and
                  Jiageng Chen and
                  Xiao Zhou and
                  Gong Cheng},
  title        = {Enhancing Dataset Search with Compact Data Snippets},
  booktitle    = {Proceedings of the 47th International {ACM} {SIGIR} Conference on
                  Research and Development in Information Retrieval, {SIGIR} 2024, Washington
                  DC, USA, July 14-18, 2024},
  publisher    = {{ACM}},
  year         = {2024},
  url          = {https://doi.org/10.1145/3626772.3657837},
  doi          = {10.1145/3626772.3657837}
}
```
