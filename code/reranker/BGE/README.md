# BGE and BGE-reranker

We implemented a reranking model for **BGE** and **BGE-reranker** based on code from [official github](https://github.com/FlagOpen/FlagEmbedding). We use [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) and [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) as initial checkpoints, respectively.

## Finetuning

We follow the official instructions to mine hard negatives first and then finetune both models according to the recommended method.

```
bash ./mineHN_{model}.sh
bash ./finetune_{model}.sh
```

## Test

We follow the official instructions calculate the relevance score of each `<query,dataset>` pair.

```
python test_bge.py
python test_bge_reranker.py
```