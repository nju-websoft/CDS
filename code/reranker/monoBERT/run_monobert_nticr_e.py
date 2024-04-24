# coding=utf-8
import argparse
import ast
import json
import os
import random
from collections import OrderedDict

import torch
from torch import nn
from tqdm import tqdm, trange
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification,
                          get_linear_schedule_with_warmup)

from monobert_eval import simple_eval
from utils import read_dataset, save_dataset, save_model, set_seed

device = torch.device("cuda:0")


class MonoBERT(nn.Module):
    def __init__(self, model_path):
        super(MonoBERT, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        dim = self.model.config.hidden_size
        self.scoring = nn.Linear(dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        output, = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        logits = output[:, 1]
        logits = self.sigmoid(logits)
        if labels is not None:
            loss = -torch.sum(torch.log(logits) * labels) - torch.sum(torch.log(1 - logits) * (1 - labels))
            return loss
        else:
            return logits


def get_input_feature_train(features, max_source_length, max_pos, max_neg):
    input_texts, labels = [], []
    for sample in features:
        positive_ctxs = sample['positive_ctxs']
        hard_negative_ctxs = sample['hard_negative_ctxs']

        if len(positive_ctxs) > max_pos:
            positive_ctxs = positive_ctxs[:max_pos]
        if len(hard_negative_ctxs) > max_neg:
            hard_negative_ctxs = hard_negative_ctxs[:max_neg]

        question = sample['question']
        for positive_ctx in positive_ctxs:
            input_texts.append([question, positive_ctx['text']])
            labels.append(1)
        for hard_negative_ctx in hard_negative_ctxs:
            input_texts.append([question, hard_negative_ctx['text']])
            labels.append(0)

    encoding = tokenizer(input_texts,
                         padding='longest',
                         max_length=max_source_length,
                         truncation='longest_first',
                         return_tensors="pt")
    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)
    token_type_ids = encoding.token_type_ids.to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    return input_ids, token_type_ids, attention_mask, labels


def get_input_feature_inference(sample, max_source_length):
    input_texts = []
    ctxs = sample['ctxs']
    question = sample['question']
    c_ids = []
    for ctx in ctxs:
        c_ids.append(ctx['c_id'])
        input_texts.append([question, ctx['text']])
    encoding = tokenizer(input_texts, padding='longest', max_length=max_source_length, truncation='longest_first')
    input_ids = encoding.input_ids
    attention_mask = encoding.attention_mask
    token_type_ids = encoding.token_type_ids
    return input_ids, token_type_ids, attention_mask, c_ids


def get_input_feature_evaluate(sample, max_source_length):
    input_texts, labels = [], []
    positive_ctxs = sample['positive_ctxs']
    hard_negative_ctxs = sample['hard_negative_ctxs']
    question = sample['question']
    q_rels = {}
    c_id = 0
    for positive_ctx in positive_ctxs:
        q_rels[str(c_id)] = 1
        c_id += 1
        input_texts.append([question, positive_ctx['text']])
        labels.append(1)
    for hard_negative_ctx in hard_negative_ctxs:
        q_rels[str(c_id)] = 0
        c_id += 1
        input_texts.append([question, hard_negative_ctx['text']])
        labels.append(0)

    encoding = tokenizer(input_texts, padding='longest', max_length=max_source_length, truncation='longest_first')
    input_ids = encoding.input_ids
    attention_mask = encoding.attention_mask
    token_type_ids = encoding.token_type_ids
    labels = torch.tensor(labels, dtype=torch.long)
    return input_ids, token_type_ids, attention_mask, labels, q_rels


@torch.no_grad()
def inference(model, test_examples, q_rels, eval_batch_size, max_len):
    model.eval()
    res_dict = {}
    for sample in tqdm(test_examples):
        q_id = sample['q_id']
        if len(sample['ctxs']) == 0:
            res_dict[q_id] = []
            continue
        input_ids, token_type_ids, attention_mask, c_ids = get_input_feature_inference(sample, max_len)
        pred_scores = []
        step_count = len(input_ids) // eval_batch_size
        if step_count * eval_batch_size < len(input_ids):
            step_count += 1
        for step in range(step_count):
            beg_index = step * eval_batch_size
            end_index = min((step + 1) * eval_batch_size, len(input_ids))
            input_ids_bt = input_ids[beg_index:end_index]
            token_type_ids_bt = token_type_ids[beg_index:end_index]
            attention_mask_bt = attention_mask[beg_index:end_index]

            input_ids_bt = torch.tensor(input_ids_bt, dtype=torch.long).to(device)
            token_type_ids_bt = torch.tensor(token_type_ids_bt, dtype=torch.long).to(device)
            attention_mask_bt = torch.tensor(attention_mask_bt, dtype=torch.long).to(device)

            logits = model(input_ids_bt, token_type_ids_bt, attention_mask_bt)
            logits = logits.tolist()
            pred_scores += logits
        pred_scores_id = [[c_id, pred_score] for c_id, pred_score in zip(c_ids, pred_scores)]
        pred_scores_id = sorted(pred_scores_id, key=lambda x: x[1], reverse=True)
        res_dict[q_id] = pred_scores_id
    # q_rels_cut = {}
    # for q_id in res_dict.keys():
    #     q_rels_cut[q_id] = q_rels[q_id]
    # result_score = simple_eval(q_rels_cut, res_dict, k=10)
    return res_dict


@torch.no_grad()
def evaluate(model, test_examples, eval_batch_size, max_len):
    model.eval()
    q_rels = {}
    res_dict = {}
    q_id = 0
    for sample in tqdm(test_examples):
        input_ids, token_type_ids, attention_mask, labels, q_rels_item = get_input_feature_evaluate(sample, max_len)
        q_rels[str(q_id)] = q_rels_item
        pred_scores = []
        step_count = len(input_ids) // eval_batch_size
        if step_count * eval_batch_size < len(input_ids):
            step_count += 1
        for step in range(step_count):
            beg_index = step * eval_batch_size
            end_index = min((step + 1) * eval_batch_size, len(input_ids))
            input_ids_bt = input_ids[beg_index:end_index]
            token_type_ids_bt = token_type_ids[beg_index:end_index]
            attention_mask_bt = attention_mask[beg_index:end_index]

            input_ids_bt = torch.tensor(input_ids_bt, dtype=torch.long).to(device)
            token_type_ids_bt = torch.tensor(token_type_ids_bt, dtype=torch.long).to(device)
            attention_mask_bt = torch.tensor(attention_mask_bt, dtype=torch.long).to(device)

            logits = model(input_ids_bt, token_type_ids_bt, attention_mask_bt)
            logits = logits.tolist()
            pred_scores += logits

        pred_scores_id = [[c_id, pred_score] for c_id, pred_score in enumerate(pred_scores)]
        pred_scores_id = sorted(pred_scores_id, key=lambda x: x[1], reverse=True)
        res_dict[str(q_id)] = pred_scores_id
        q_id += 1
    result_score = simple_eval(q_rels, res_dict, k=10)
    return result_score, res_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--task", default='NTCIR-E', type=str)
    parser.add_argument("--model_name", default='/home/qschen/Models/monobert-large-msmarco', type=str)
    parser.add_argument("--save_model", default=False, type=ast.literal_eval)
    parser.add_argument("--only_eval", default=False, type=ast.literal_eval)
    parser.add_argument("--only_dev", default=False, type=ast.literal_eval)
    parser.add_argument("--debug", default=False, type=ast.literal_eval)
    parser.add_argument("--train_path", default='./data/ntcir15/metadata/metadata_split_train.json', type=str)
    parser.add_argument("--dev_path", default='./data/ntcir15/metadata/ntcir15_BM25_top10_dev.json', type=str)
    parser.add_argument("--test_path", default='./data/ntcir/metadata/ntcir15_BM25_top10.json', type=str)
    parser.add_argument("--test_qrels_path", default='./data/qrels/ntcir_merge_qrels.json', type=str)
    parser.add_argument("--results_save_path", default='./results/ntcir15/', type=str)
    parser.add_argument("--max_train_pos", default=3, type=int)
    parser.add_argument("--max_train_neg", default=3, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total batch size for eval.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=16,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--output_dir",
                        default='./outputs/ntcir15/',
                        type=str,
                        help="The output dreader2ctory whretriever the model checkpoints will be written.")
    parser.add_argument("--init_checkpoint",
                        default=False,
                        type=ast.literal_eval,
                        help="Initial checkpoint (usually from a pre-trained BERT model)")
    parser.add_argument("--init_checkpoint_path",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model)")
    parser.add_argument("--max_len", default=512, type=int)
    parser.add_argument("--lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--epoch_num", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
    parser.add_argument('--topk', type=int, default=10)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    only_eval = args.only_eval
    only_dev = args.only_dev
    debug = args.debug
    test_name = args.test_path.split('/')[-1].replace('.json', '')
    encoder_name = args.model_name.split('/')[-1].split('-')[0]
    config_name = f'monoBERT_{args.task}_{encoder_name}'
    parameter_name = f'lr_{args.lr}_bs_{args.train_batch_size}_epoch_{args.epoch_num}'
    output_model_path = f'{args.output_dir}/{config_name}/{parameter_name}/'
    path_save_result = f'{args.results_save_path}/{config_name}/{parameter_name}/'

    os.makedirs(path_save_result, exist_ok=True)
    set_seed(args.seed)

    train_examples, dev_examples = [], []
    train_examples += read_dataset(args.train_path)
    dev_examples += read_dataset(args.dev_path)

    test_examples = read_dataset(args.test_path)
    test_qrels = read_dataset(args.test_qrels_path)
    if debug:
        train_examples = train_examples[:20]
        dev_examples = dev_examples[:20]
        test_examples = test_examples[:20]

    train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = MonoBERT(args.model_name).to(device)

    print('# parameters:', sum(param.numel() for param in model.parameters()))
    print(
        json.dumps(
            {
                "lr": args.lr,
                "model": args.model_name,
                "seed": args.seed,
                "bs": args.train_batch_size,
                'task': args.task,
                'gradient_accumulation_steps': args.gradient_accumulation_steps,
                "epoch": args.epoch_num,
                "train_size": len(train_examples),
                "dev_size": len(dev_examples),
                "test_size": len(test_examples),
                'max_len': args.max_len,
                'output_model_path': output_model_path,
                'path_save_result': path_save_result,
                'init_checkpoint': args.init_checkpoint,
                'max_train_pos': args.max_train_pos,
                'max_train_neg': args.max_train_neg,
                'topk': args.topk
            },
            indent=2))
    if args.init_checkpoint:
        if args.init_checkpoint_path is None:
            init_checkpoint = f'{output_model_path}/pytorch_model.bin'
        else:
            init_checkpoint = args.init_checkpoint_path
        checkpoint = torch.load(init_checkpoint, map_location='cpu')
        model_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict()
        for k in list(model_dict.keys()):
            name = k
            if k.startswith('module.bert.bert.'):
                name = k.replace("module.bert.", "")
            new_state_dict[name] = model_dict[k]
            del model_dict[k]
        model.load_state_dict(new_state_dict, False)
        print('init from:', init_checkpoint)
    if only_eval:
        results_test = inference(model, test_examples, test_qrels, args.eval_batch_size,
                                                    args.max_len)
        # print('test scores:', json.dumps(result_score_test['mean'], indent=2))
        save_dataset(path_save_result + '/test_top{args.topk}.json', results_test)
        exit(0)
    elif only_dev:
        results_test = inference(model, dev_examples, test_qrels, args.eval_batch_size,
                                                    args.max_len)
        # print('test scores:', json.dumps(result_score_test['mean'], indent=2))
        save_dataset(path_save_result + f'/dev_top{args.topk}.json', results_test)
        exit(0)
    else:
        results_test = inference(model, test_examples, test_qrels, args.eval_batch_size,
                                                    args.max_len)
        # print('init test scores:', json.dumps(result_score_test['mean'], indent=2))

    warm_up_ratio = 0.05
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    t_total = args.epoch_num * (len(train_examples) // train_batch_size // args.gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=int(warm_up_ratio * (t_total)),
                                                num_training_steps=t_total)
    step_count, step_all, early_stop = 0, 0, 0

    best_test_ndcg = 0
    best_dev_ndcg = 0
    best_dev_result, best_test_result = None, None
    for epoch in range(args.epoch_num):
        tr_loss, nb_tr_steps = 0, 0.1
        early_stop += 1
        order = list(range(len(train_examples)))
        random.seed(args.seed + epoch)
        random.shuffle(order)
        model.train()
        step_count = len(train_examples) // train_batch_size
        if step_count * train_batch_size < len(train_examples):
            step_count += 1
        step_trange = trange(step_count)
        for step in step_trange:
            step_all += 1
            beg_index = step * train_batch_size
            end_index = min((step + 1) * train_batch_size, len(train_examples))
            order_index = order[beg_index:end_index]
            batch_example = [train_examples[index] for index in order_index]
            input_ids, token_type_ids, attention_mask, labels = get_input_feature_train(batch_example,
                                                                                        args.max_len,
                                                                                        max_pos=args.max_train_pos,
                                                                                        max_neg=args.max_train_neg)
            loss = model(input_ids, token_type_ids, attention_mask, labels)
            loss = loss.mean()
            tr_loss += loss.item()
            nb_tr_steps += 1
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            loss_show = ' Epoch:' + str(epoch) + " loss:" + str(round(tr_loss / nb_tr_steps,
                                                                      4)) + f" lr:{'%.2E' % scheduler.get_last_lr()[0]}"
            step_trange.set_postfix_str(loss_show)
        save_model(output_model_path, model, best_dev_ndcg, optimizer)
        # result_score_dev, results_dev = evaluate(model, dev_examples, args.eval_batch_size, args.max_len)
        # ndcg_cut_5_mean = result_score_dev['mean']['ndcg_cut_5_mean']
        # print('dev scores:', json.dumps(result_score_dev['mean'], indent=2))
        # save_dataset(path_save_result + '/dev.json', results_dev)
        # if ndcg_cut_5_mean > best_dev_ndcg:
        #     print('new best')
        #     best_dev_result = result_score_dev
        #     best_dev_ndcg = ndcg_cut_5_mean
        #     save_model(output_model_path, model, best_dev_ndcg, optimizer)
            # result_score_test, results_test = inference(model, test_examples, test_qrels, args.eval_batch_size,
            #                                             args.max_len)
            # best_test_result = result_score_test['mean']
            # print('test scores:', json.dumps(result_score_test['mean'], indent=2))
            # save_dataset(path_save_result + '/test.json', results_test)

    print('best test scores:', json.dumps(best_test_result, indent=2))