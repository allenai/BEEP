'''
Code to run LM-based reranker over abstracts retrieved per query
Command: python run_reranker.py --data <DATA_DIR> --entities <ENTITY_PICKLE_FILE> --out_dir <OUT_DIR> --model_name_or_path <MODEL> --do_train --do_test
'''

import argparse
import os
import random
import statistics
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
from data_loader import TRECDataset


def train(model, train_data, dev_data, out_dir, label_vocab, epochs, lr, fold):
    label_list = [x[0] for x in list(sorted(label_vocab.items(), key=lambda x: x[1]))]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    step = 0
    prev_dev_loss = 10000
    for epoch in range(epochs):
        random.shuffle(train_data)
        model.train()
        epoch_loss = 0.0
        for batch in train_data:
            optimizer.zero_grad()
            gpu_batch = {x:y.cuda() for x,y in batch.items() if x not in ['query_id', 'article_id']}  # Tensors need to be moved to GPU
            outputs = model(**gpu_batch)   # Batch is a dictionary from HF that needs to be unpacked
            loss = outputs[0]
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            step += 1
            if step%1000 == 0:
                print('Completed {} training steps'.format(step))
        epoch_loss /= len(train_data)
        print('Training loss after epoch {}: {}'.format(epoch, epoch_loss))
        # Checkpoint model after each epoch anyway, in addition to storing best loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, os.path.join(out_dir, 'checkpoints/checkpoint_{}_fold_{}.pt'.format(epoch, fold)))
        dev_loss, _ = test(model, dev_data, label_vocab, epoch=epoch, return_loss=True)
        if dev_loss < prev_dev_loss:
            prev_dev_loss = dev_loss
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model_fold_{}.pt'.format(fold)))
        scheduler.step(dev_loss)


def test(model, dev_data, label_vocab, epoch=0, return_loss=False):
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
        dev_loss += loss.item()
        cur_preds = outputs[1].detach().cpu()
        cur_probs = softmax(cur_preds)
        relevance = cur_probs.numpy()[:,1].tolist()
        cur_gold_relevance = batch['labels'].detach().cpu().numpy().tolist()
        for i, qid in enumerate(batch['query_id']):
            aid = batch['article_id'][i]
            if qid not in model_predictions:
                model_predictions[qid] = []
            model_predictions[qid].append((aid, relevance[i], cur_gold_relevance[i]))
    dev_loss /= len(dev_data)
    # pickle.dump(model_predictions, open('trec_2016_filtered_lowlr.pkl', 'wb'))
    print('Validation loss after epoch {}: {}'.format(epoch, dev_loss))
    scores = compute_precision_at_k(model_predictions, k=10)
    score_list = list(scores.values())
    mean_preck = sum(score_list) / len(score_list)
    median_preck = statistics.median(score_list)
    print('------------------Precision@K Scores for Epoch {}-------------------'.format(epoch))
    print('Mean Precision@K: {}'.format(mean_preck))
    print('Median Precision@K: {}'.format(median_preck))
    for file in scores:
        print('{}\t{}'.format(file, scores[file]))
    # TODO: Maybe add NDCG or other metric implementations too
    if return_loss:
        return dev_loss, scores
    return scores


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
    parser.add_argument('--data', type=str, action='store', required=True, help='Path to directory containing train/dev/test data')
    parser.add_argument('--entities', type=str, action='store', required=True, help='Path to file containing extracted entities for queries')
    parser.add_argument('--out_dir', type=str, action='store', required=True, help='Provide path to directory to store outputs')
    parser.add_argument('--model_name_or_path', type=str, action='store', required=True, help='Path to pretrained LM to be used')
    parser.add_argument('--query_type', type=str, action='store', default='all', help='Indicate which entity types to include in query [mesh/all]')
    parser.add_argument('--query_format', type=str, action='store', default='entity', help='Indicate how query should be framed (text/entity)')
    parser.add_argument('--do_train', action='store_true', default=False, help='Specify if training should be performed')
    parser.add_argument('--do_test', action='store_true', default=False, help='Specify if evaluation on test data should be performed')
    parser.add_argument('--rebalance', action='store_true', default=False, help='Specify whether training data for reranker should be rebalanced')
    parser.add_argument('--batch_size', type=int, action='store', default=8, help='Specify batch size')
    parser.add_argument('--lr', type=float, action='store', default=2e-5, help='Specify learning rate')
    parser.add_argument('--epochs', type=int, action='store', default=20, help='Specify number of epochs')
    parser.add_argument('--seed', type=int, action='store', default=42, help='Specify random seed')
    parser.add_argument('--checkpoint', type=str, action='store', help='Path to checkpoint to load model weights from')
    parser.add_argument('--years', type=str, action='store', help='Provide years from which TREC data should be used')
    parser.add_argument('--folds', type=int, action='store', default=5, help='Provide number of folds to use in cross-validation')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    checkpoint_dir = os.path.join(args.out_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    set_seed(args.seed)

    years = None
    if args.years is not None:
        years = [int(x) for x in args.years.split(',')] if ',' in args.years else [int(args.years)]
    dataset = TRECDataset(args.data, years)
    print('Loaded train, dev and test splits...')
    dataset.add_entities(args.entities)
    print('Added entities to dataset...')

    label_vocab = {'Relevant': 1, 'Irrelevant': 0}

    # Initialized within each fold
    tokenizer = None
    model = None
    config = None

    def preprocess_function(examples):
        # Tokenize the texts
        args = (([x[2] for x in examples], [x[3] for x in examples]))
        # We can try out longformer models at some point??
        result = tokenizer(*args, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        result["labels"] = torch.cuda.LongTensor([(x[-1]) for x in examples])
        result["query_id"] = [(x[0]) for x in examples]
        result["article_id"] = [(x[1]) for x in examples]
        return result

    def rebalance(examples):
        per_query_relevance = {}
        for sample in examples:
            if sample[0] not in per_query_relevance:
                per_query_relevance[sample[0]] = {0: 0, 1: 0}
            per_query_relevance[sample[0]][sample[-1]] += 1
        extra_samples = []
        for query in per_query_relevance:
            gap = per_query_relevance[query][0] - per_query_relevance[query][1]
            rel_samples = [x for x in examples if x[0] == query and x[-1] == 1]
            for i in range(gap):
                chosen_sample = random.choice(rel_samples)
                article_tokens = chosen_sample[-2].split()
                while len(article_tokens) == 0:
                    chosen_sample = random.choice(rel_samples)
                    article_tokens = chosen_sample[-2].split()
                # Drop out one word
                del article_tokens[random.choice(range(len(article_tokens)))]
                chosen_sample[-2] = ' '.join(article_tokens)
                extra_samples.append(chosen_sample)
        final_samples = examples + extra_samples
        random.shuffle(final_samples)
        return final_samples

    def batch_and_tokenize_data(examples, batch_size, split):
        example_list = []
        for file in examples:
            example = examples[file]
            query = example['outcome'] + ' [ENTSEP] ' + ' [ENTSEP] '.join(example['entities']['mesh'])
            if args.query_type == 'all':   # Flag to include non-mesh linked entities too
                query += ' ' + ' [ENTSEP] '.join(example['entities']['non-mesh'])
            if args.query_format == 'text':
                query = 'What is the {}? '.format(example['outcome']) + example['text']
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

    all_data = {**dataset.train_data, **dataset.dev_data, **dataset.test_data}
    print('Creating {} folds...'.format(args.folds))
    fold_size = 1.0/args.folds
    test_size = len(all_data) * fold_size
    train_size = len(all_data) * ((args.folds-1) * fold_size)
    queries = list(all_data.keys())
    random.shuffle(queries)
    scores = {}
    for i in range(args.folds):
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
        model = model.cuda()

        test_start =int(i*test_size)
        test_end = int((i+1)*test_size)
        test_queries = queries[test_start:test_end]
        train_queries = [x for x in queries if x not in test_queries]
        random.shuffle(train_queries)
        dev_cutoff = int(0.5*test_size)
        dev_queries = train_queries[:dev_cutoff]
        train_queries = train_queries[dev_cutoff:]
        train_data = {k:v for k,v in all_data.items() if k in train_queries}
        dev_data = {k:v for k,v in all_data.items() if k in dev_queries}
        test_data = {k:v for k,v in all_data.items() if k in test_queries}
        print('Started batch creation for fold {}'.format(i))
        train_batches = batch_and_tokenize_data(train_data, args.batch_size, 'train')
        print('Created {} train batches'.format(len(train_batches)))
        dev_batches = batch_and_tokenize_data(dev_data, args.batch_size, 'dev')
        print('Created {} dev batches'.format(len(dev_batches)))
        test_batches = batch_and_tokenize_data(test_data, args.batch_size, 'test')
        print('Created {} test batches'.format(len(test_batches)))

        print('Running training/testing loop for fold {}'.format(i))
        if args.do_train:
            train(model, train_batches, dev_batches, args.out_dir, label_vocab, args.epochs, args.lr, fold=i)
        if args.do_test:
            if args.checkpoint is not None:
                if 'checkpoint' in args.checkpoint:
                    full_checkpoint = torch.load(args.checkpoint)
                    model.load_state_dict(full_checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(torch.load(args.checkpoint))
            else:
                model.load_state_dict(torch.load(os.path.join(args.out_dir, 'best_model_fold_{}.pt').format(i)))
            fold_scores = test(model, test_batches, label_vocab)
            scores = {**scores, **fold_scores}

    print('Final Scores On All Queries:')
    print('Mean Precision@K: {}'.format(statistics.mean(list(scores.values()))))
    print('Median Precision@K: {}'.format(statistics.median(list(scores.values()))))
    for file in scores:
        print('{}: {}'.format(file, scores[file]))
