'''
Code to run LM-based reranker over abstracts retrieved per query
Command: python text_reranker.py --retrieval_results <RETRIEVAL_PICKLE_FILE> --entities <ENTITY_PICKLE_FILE> --out_dir <OUT_DIR> --model_name_or_path <MODEL> --checkpoint <MODEL_CHECKPOINT>
'''

import argparse
import os
import pickle
import csv
import random
import statistics
from itertools import chain
import time

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
from data_loader import TextDataset


def test(model, dev_data, label_vocab, epoch=0):
    model.eval()
    softmax = nn.Softmax(dim=1)
    label_list = [x[0] for x in list(sorted(label_vocab.items(), key=lambda x: x[1]))]
    dev_loss = 0.0
    model_predictions = {}
    gold_labels = []
    for batch in dev_data:
        gpu_batch = {x:y.cuda() for x,y in batch.items() if x not in ['query_id', 'article_id']}
        outputs = model(**gpu_batch)
        loss = outputs[0]
        cur_preds = outputs[1].detach().cpu()
        cur_probs = softmax(cur_preds)
        relevance = cur_probs.numpy()[:,1].tolist()
        if batch['query_id'][0] not in model_predictions:
            model_predictions[batch['query_id'][0]] = []
        model_predictions[batch['query_id'][0]] += list(zip(batch['article_id'], relevance))
        # cur_gold_relevance = batch['labels'].detach().cpu().numpy().tolist()
        # for i, qid in enumerate(batch['query_id']):
        #    aid = batch['article_id'][i]
        #    if qid not in model_predictions:
        #        model_predictions[qid] = []
        #    model_predictions[qid].append((aid, relevance[i]))
    return model_predictions


def compute_precision_at_k(preds, k=10):
    precision = {}
    for file in preds:
        cur_ordered_preds = list(reversed(sorted(preds[file], key=lambda x : x[1])))
        cur_prec = 0.0
        for chosen_id in cur_ordered_preds[:k]:
            cur_prec += 1 if chosen_id[-1] == 1 else 0
        cur_prec /= float(k)
        precision[file] = cur_prec
    return precision


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--data', type=str, action='store', required=True, help='Path to directory containing train/dev/test data')
    parser.add_argument('--entities', type=str, action='store', required=True, help='Path to file containing extracted entities for queries')
    parser.add_argument('--retrieval_results', type=str, action='store', required=True, help='Path to file containing retrieved query results')
    parser.add_argument('--abstracts', type=str, action='store', default='../pubmed-retrieval/pubmed_texts_and_dates.pkl',\
            help='Path to file containing abstract texts')
    parser.add_argument('--out_dir', type=str, action='store', required=True, help='Provide path to directory to store outputs')
    parser.add_argument('--model_name_or_path', type=str, action='store', required=True, help='Path to pretrained LM to be used')
    parser.add_argument('--query_type', type=str, action='store', default='all', help='Indicate which entity types to include in query [mesh/all]')
    parser.add_argument('--query_format', type=str, action='store', default='entity', help='Indicate how query should be framed (text/entity)')
    # parser.add_argument('--do_train', action='store_true', default=False, help='Specify if training should be performed')
    # parser.add_argument('--do_test', action='store_true', default=False, help='Specify if evaluation on test data should be performed')
    parser.add_argument('--rebalance', action='store_true', default=False, help='Specify whether training data for reranker should be rebalanced')
    parser.add_argument('--batch_size', type=int, action='store', default=20, help='Specify batch size')
    parser.add_argument('--lr', type=float, action='store', default=2e-5, help='Specify learning rate')
    parser.add_argument('--epochs', type=int, action='store', default=20, help='Specify number of epochs')
    parser.add_argument('--seed', type=int, action='store', default=42, help='Specify random seed')
    parser.add_argument('--checkpoint', type=str, action='store', help='Path to checkpoint to load model weights from')
    parser.add_argument('--k', type=int, action='store', help='Number of top results to rerank')
    parser.add_argument('--outcome', type=str, action='store', required=True, help='Target outcome to predict')
    parser.add_argument('--years', type=str, action='store', help='Provide years from which TREC data should be used')
    # parser.add_argument('--split', type=int, action='store', help='Split of IDs to rerank for LOS/Mortality outcomes')
    parser.add_argument('--text_file', type=str, action='store', required=True, help='Path to file containing raw texts from EHRs')
    args = parser.parse_args()

    set_seed(args.seed)

    dataset = TextDataset(args.retrieval_results, args.abstracts, args.entities)
    dataset.select_articles(args.k)
    dataset.create_dataset(args.outcome)
    print('Loaded train, dev and test splits...')
    dataset.add_ehr_text(args.text_file)
    dataset.add_entities()
    print('Added entities to dataset...')

    label_vocab = {'Relevant': 1, 'Irrelevant': 0}

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(list(label_vocab.keys())),
        label2id=label_vocab,
        id2label={i: l for l, i in label_vocab.items()},
        # finetuning_task='mnli',
        cache_dir='../../cache',
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir='../../cache',
        use_fast=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir='../../cache',
    )

    special_tokens_dict = {'additional_special_tokens': ['[ENTSEP]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    def preprocess_function(examples):
        # Tokenize the texts
        args = (([x[2] for x in examples], [x[3] for x in examples]))
        # We can try out longformer models at some point??
        result = tokenizer(*args, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        result["labels"] = torch.cuda.LongTensor([(x[-1]) for x in examples])
        result["query_id"] = [(x[0]) for x in examples]
        result["article_id"] = [(x[1]) for x in examples]
        return result

    outcome_questions = {'mortality': 'What is the hospital mortality?', \
            'pmv': 'What is the probability of prolonged mechanical ventilation?', \
            'los': 'What is the probable length of stay?'}

    def batch_and_tokenize_data(examples, batch_size, split):
        example_list = []
        for file in examples:
            example = examples[file]
            query = example['outcome'] + ' [ENTSEP] ' + ' [ENTSEP] '.join(example['entities']['mesh'])
            if args.query_type == 'all':   # Flag to include non-mesh linked entities too
                query += ' ' + ' [ENTSEP] '.join(example['entities']['non-mesh'])
            if args.query_format == 'text':
                query = outcome_questions[example['outcome']] + ' ' + example['text']
            for id in example['articles']:
                article = example['articles'][id]
                example_list.append([file, id, query, article['article_text'], article['judgement']])
        if split == 'train':
            random.shuffle(example_list)
            if args.rebalance:
                example_list = rebalance(example_list)
        batches = []
        for i in range(0, len(example_list), batch_size):
            start = i
            end = min(start+batch_size, len(example_list))
            batch = preprocess_function(example_list[start:end])
            batches.append(batch)
            if len(batches) % 1000 == 0:
                print('Created {} batches'.format(len(batches)))
        return batches

    model = model.cuda()
    if 'checkpoint' in args.checkpoint:
        full_checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(full_checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(args.checkpoint))
    fnum = 0
    # NOTE: If dataset is large, the split variables can be used to divide data into folds and run multiple reranking processes in parallel
    # split_to_keep = pickle.load(open('../pubmed-retrieval/all_ids_split{}.pkl'.format(args.split), 'rb')) if args.outcome != 'pmv' else None
    for file in list(dataset.data.keys()):
        # if split_to_keep is not None and file not in split_to_keep:
        #    continue
        start = time.time()
        cur_query = {file: dataset.data[file]}
        print('Ranking documents for query {} ({})'.format(file, fnum))
        fnum += 1
        batches = batch_and_tokenize_data(cur_query, args.batch_size, 'test')
        ranked_docs = test(model, batches, label_vocab)
        if file not in ranked_docs:
            print('Warning: No reranking performed for {}'.format(file))
            continue
        end = time.time()
        print('Time taken: {}'.format(end-start))
        pickle.dump(ranked_docs, open(os.path.join('{}_reranked_final/'.format(args.outcome), file+'.pkl'), 'wb'))
