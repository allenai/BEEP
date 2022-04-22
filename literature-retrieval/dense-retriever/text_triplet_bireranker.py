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

from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    set_seed,
)
from data_loader_bireranker import TextDataset


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
    parser.add_argument('--text_file', type=str, action='store', required=True, help='Path to file containing raw texts from EHRs')
    args = parser.parse_args()

    set_seed(args.seed)

    dataset = TextDataset(args.retrieval_results, args.abstracts, args.entities)
    # dataset.select_articles(args.k)
    dataset.add_ehr_text(args.text_file, args.outcome)
    dataset.create_dataset(args.outcome)
    print('Loaded train, dev and test splits...')
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

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir='../../cache',
    )

    special_tokens_dict = {'additional_special_tokens': ['[ENTSEP]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    model = model.cuda()
    if 'checkpoint' in args.checkpoint:
        full_checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(full_checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(args.checkpoint))

    def encode_texts(texts):
        embeds = {}
        i = 0
        for file in texts:
            print('Encoding article {}'.format(i))
            i += 1
            text = texts[file]
            text = tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
            text = {x:y.cuda() for x,y in text.items()}
            embed = model(**text)
            embeds[file] = embed['last_hidden_state'][0,0,:].detach().cpu().numpy()
        return embeds

    # TODO: Let full literature be used after testing
    # keys = random.sample(list(dataset.texts.keys()), 100)
    # dataset.texts = {k:v for k,v in dataset.texts.items() if k in keys}
    encoded_pubmed_articles = encode_texts(dataset.texts)
    print('Encoded all outcome-specific articles')

    article_items = list(encoded_pubmed_articles.items())
    article_ids, article_matrix = [x[0] for x in article_items], [x[1] for x in article_items]
    article_matrix = np.vstack(article_matrix)

    outcome_questions = {'mortality': 'What is the hospital mortality?', \
            'pmv': 'What is the probability of prolonged mechanical ventilation?', \
            'los': 'What is the probable length of stay?'}

    fnum = 0
    # result_csv = open(os.path.join(args.out_dir, '{}_reranked_results.csv'.format(args.outcome)), 'w')
    # writer = csv.writer(result_csv)
    # writer.writerow(['ID','Note'] + ['ArticleID', 'Article']*10)
    for file in list(dataset.data.keys()):
        cur_query = {file: dataset.data[file]}
        print('Ranking documents for query {} ({})'.format(file, fnum))
        fnum += 1
        cur_query_text = outcome_questions[args.outcome] + ' ' + dataset.query_texts[file]
        cur_query_text = tokenizer(cur_query_text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
        cur_query_text = {x:y.cuda() for x,y in cur_query_text.items()}
        cur_query_embed = model(**cur_query_text)
        cur_query_embed = cur_query_embed['last_hidden_state'][0,0,:].detach().cpu().numpy().transpose()
        cur_query_embed = cur_query_embed.reshape(1,-1)
        similarities = euclidean_distances(cur_query_embed, article_matrix).tolist()[0]
        ranked_docs = list(zip(article_ids, similarities))
        ranked_docs = list(sorted(ranked_docs, key=lambda x:x[1]))
        pickle.dump(ranked_docs[:1000], open(os.path.join('{}_reranked_triplet_bienc/'.format(args.outcome), file+'.pkl'), 'wb'))
        # writer.writerow([file, dataset.data[file]['text']] + \
        #        list(chain.from_iterable((x[0] ,dataset.data[file]['articles'][x[0]]['article_text']) for x in ranked_docs[:10])))
    # result_csv.close()
