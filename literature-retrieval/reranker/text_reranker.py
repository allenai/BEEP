'''
Code to run LM-based reranker over abstracts retrieved per query
Command: python text_reranker.py --retrieval_results <RETRIEVAL_PICKLE_FILE> --entities <ENTITY_PICKLE_FILE> --out_dir <OUT_DIR> --model_name_or_path <MODEL> --checkpoint <MODEL_CHECKPOINT>
'''
print("started text_reranker...")
import torch
print("is cuda available?", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("cuda not available. exiting")
    exit(0)
import argparse
import os
import pickle
import time
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
from optimum.bettertransformer import BetterTransformer
import torch.nn as nn
from scipy import stats

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
print("importing TextDataset")
from data_loader import TextDataset


def predict(model, dev_data):
    with torch.no_grad():
        model.eval()
        softmax = nn.Softmax(dim=1)
        model_predictions = {}
        i = 1
        sum_of_iters = 0
        start_for = time.time()
        cur_preds_array  = []
        relevances_array = []
        batches_inds = []
        batches_articles_ids = []
        for batch in dev_data:
            start_iter = time.time()
            start = time.time()
            gpu_batch = {x:y.cuda() for x,y in batch.items() if x not in ['query_id', 'article_id']}
            end = time.time()
            gpuing = round(end-start, 4)
            start = time.time()
            outputs = model(**gpu_batch)
            end = time.time()
            applying_model = round(end-start, 4)
            start = time.time()
            torch.cuda.synchronize()
            end = time.time()
            sync = round(end-start, 4)
            start = time.time()
            cur_preds = outputs[1]
            end = time.time()
            taking_output = round(end-start, 4)
            start = time.time()
            cur_preds = cur_preds.detach()
            #cur_preds_array.append(cur_preds)
            end = time.time()
            detaching = round(end-start, 4)
            
            #batches_inds.append(batch['query_id'][0])
            #batches_articles_ids.append(batch["article_id"])
            
            start = time.time()
            cur_preds = cur_preds.cpu()
            end = time.time()
            cpuing = round(end-start, 4)
            start = time.time()
            cur_probs = softmax(cur_preds)
            end = time.time()
            softmaxing = round(end-start, 4)
            start = time.time()
            relevance = cur_probs.numpy()[:,1].tolist()
            #relevances_array.append(cur_probs.numpy())
            end = time.time()
            relev = round(end-start, 4)
            if batch['query_id'][0] not in model_predictions:
                model_predictions[batch['query_id'][0]] = []
            start = time.time()
            model_predictions[batch['query_id'][0]] += list(zip(batch['article_id'], relevance))
            end = time.time()
            zipping = round(end-start, 4)
            i += 1
            end_iter = time.time()
            iter_time = round(end_iter-start_iter, 4)
            sum_of_iters += iter_time
            #print(f"iter took {iter_time}: gpuing: {gpuing}, applying model: {applying_model}, synchronizing: {sync}, taking_output: {taking_output}, detaching: {detaching}, cpuing: {cpuing}, softmaxing: {softmaxing}, relevance: {relev}, zipping: {zipping}", flush=True)
        
        end_for = time.time()
        print(f"for {i} batches took {round(end_for-start_for, 4)}, with sum_of_iters: {round(sum_of_iters, 4)}", flush=True)

        """
        for cur_preds, batch_ind, batch_articles_ids in zip(cur_preds_array, batches_inds, batches_articles_ids):
            start_iter = time.time()
            cur_preds = cur_preds.cpu()
            cur_probs = softmax(cur_preds)
            relevance = cur_probs.numpy()[:, 1].tolist()
            if batch_ind not in model_predictions:
                model_predictions[batch_ind] = []
            model_predictions[batch_ind] += list(zip(batch_articles_ids, relevance))
            end_iter = time.time()
            iter_time = round(end_iter-start_iter, 4)
            print(f"iter after took {iter_time}", flush=True)
        """

        return model_predictions


def rerank(doc_ranks, outcome, ids2keep_file, ranking_type, top_k, abstracts, out_dir,
           query_format, query_type, ehr_entities, ehr_texts,
           model_name_or_path, checkpoint, batch_size, seed, mod=None):
    print("started rerank...", flush=True)

    if outcome == "pmv" and ids2keep_file is None:
        raise(ValueError("for outcome 'pmv' ids2keep should be provided."))

    if query_format == "text":
        assert ehr_texts is not None, "if query_format is text, path to texts should be provided"
    elif query_format == "entity":
        assert ehr_entities is not None, "if query_format is entity, path to ehr's entities should be provided"
    else:
        raise(ValueError(f"query format '{query_format}' is not recognized"))

    if seed is not None:
        set_seed(seed)

    dataset = TextDataset(doc_ranks, abstracts, ids2keep_file)
    dataset.select_articles(top_k, ranking_type)
    dataset.create_dataset(outcome)
    if query_format == "text":
        print("adding ehr texts...", flush=True)
        dataset.add_ehr_text(ehr_texts)
        print('Added texts to dataset...', flush=True)
    elif query_format == "entity":
        print("adding ehr entities", flush=True)
        dataset.add_ehr_entities(ehr_entities)
        print('Added entities to dataset...', flush=True)
    else:
        raise(ValueError(f"query format '{query_format}' is not recognized"))

    label_vocab = {'Relevant': 1, 'Irrelevant': 0}

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=len(list(label_vocab.keys())),
        label2id=label_vocab,
        id2label={i: l for l, i in label_vocab.items()},
        cache_dir='../../cache',
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir='../../cache',
        use_fast=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
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

    # TODO why do need questions at all?
    outcome_questions = {'mortality': 'What is the hospital mortality?', # the probaility of?
                        'pmv': 'What is the probability of prolonged mechanical ventilation?',
                        'los': 'What is the probable length of stay?'}

    def batch_and_tokenize_data(examples, batch_size):
        example_list = []
        for file in examples:
            example = examples[file]
            if query_format == "text":
                query = outcome_questions[example['outcome']] + ' ' + example['text']
            elif query_format == "entity":
                if query_type == "mesh":
                    query = example['outcome'] + ' [ENTSEP] ' + ' [ENTSEP] '.join(example['entities']['mesh'])
                elif query_type == "all":
                    query = example['outcome'] + ' [ENTSEP] ' + ' [ENTSEP] '.join(example['entities']['mesh'])
                    query += ' ' + ' [ENTSEP] '.join(example['entities']['non-mesh'])
                else:
                    raise (ValueError(f"query type '{query_type}' is not recognized"))
            else:
                raise(ValueError(f"query format '{query_format}' is not recognized"))

            for id in example['articles']:
                article = example['articles'][id]
                example_list.append([file, id, query, article['article_text'], article['judgement']])

        batches = []
        for i in range(0, len(example_list), batch_size):
            print(f"{i} out of {len(example_list)}", flush=True)
            start = i
            end = min(start+batch_size, len(example_list))
            batch = preprocess_function(example_list[start:end])
            #yield batch
            batches.append(batch)
        print('Created {} batches'.format(len(batches)), flush=True)

        return batches

    model = model.cuda()
    if 'checkpoint' in checkpoint:
        full_checkpoint = torch.load(checkpoint)
        model.load_state_dict(full_checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(checkpoint))
    model = BetterTransformer.transform(model)
    fnum = 0

    reranked_all = {}

    num_files = len(dataset.data.keys())
    for file in sorted(dataset.data.keys()):
        out_file_path = os.path.join(out_dir, file+'.pkl')
        fnum += 1
        if mod is not None and fnum%10 !=  mod:
            continue
        print('Ranking documents for query {} ({}/{})'.format(file, fnum, num_files), flush=True)
        if os.path.isfile(out_file_path):
            time_stamp_last_modified = os.path.getmtime(out_file_path)
            time_last_modified = datetime.fromtimestamp(time_stamp_last_modified)
            print(f"{file} existing from {time_last_modified}", flush=True)
            with open(out_file_path, "rb") as f:
                reranked_docs = pickle.load(f)
        else:
            start = time.time()
            cur_query = {file: dataset.data[file]}
            try:
                batches = batch_and_tokenize_data(cur_query, batch_size)
            except BaseException as e:
                print(e)
                continue
            end = time.time()
            print(f"tokenizing and batching for {top_k} docs took {end-start}", flush=True)
            start_predicting = time.time()
            reranked_docs = predict(model, batches)
            if file not in reranked_docs:
                print('Warning: No reranking performed for {}'.format(file), flush=True)
                continue
            end_predicting = time.time()
            print('predicting for {} docs took: {}'.format(top_k, end_predicting-start_predicting, flush=True))
            reranked_docs = reranked_docs[file]

        ranked = [x[1] for x in sorted(dataset.doc_ranks[file], key= lambda x: x[0])]
        reranked = [x[1] for x in sorted(reranked_docs, key=lambda x: x[0])]
        try:
            pearson_corr = np.corrcoef(ranked, reranked)[0, 1]
            spearman_corr = stats.spearmanr(ranked, reranked).statistic
        except Exception as e:
            print(e)
            continue 
        print(f"pearson corr: {pearson_corr}, spearman corr: {spearman_corr}")

        reranked_docs = sorted(reranked_docs, key=lambda x: x[1], reverse=True)

        reranked_all[file] = reranked_docs
        pickle.dump(reranked_docs, open(out_file_path, 'wb'))
        
    out_file_name = os.path.basename(doc_ranks).replace(".pkl", "_reranked.pkl")
    pickle.dump(reranked_all, open(os.path.join(out_dir, out_file_name), 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc_ranks', type=str, action='store', required=True,
                        help='Path to file {ehr_id: [(doc_id, rank) for m doc_ids ]')
    parser.add_argument('--ranking_type', type=str, choices=["similarity", "distance"], action='store', required=True,
                        help='type of ranking "similarity"/"distance" (relevant in taking top k)')
    parser.add_argument('--abstracts', type=str, action='store', required=True,
                        help='Path to file containing abstract texts')
    parser.add_argument('--outcome', type=str, action='store', required=True, help='Target outcome to predict')
    parser.add_argument('--query_format', type=str, action='store', choices=["text", "entity"], default='text',
                        help='Indicate how query should be framed (text/entity)')
    parser.add_argument('--query_type', type=str, action='store', choices=["mesh", "all"], default='all',
                        help='Indicate which entity types to include in query [mesh/all]')
    parser.add_argument('--ehr_entities', type=str, action='store', default=None,
                        help='Path to file containing extracted entities for queries')
    parser.add_argument('--ehr_texts', type=str, action='store', default=None,
                        help='Path to file containing raw texts from EHRs')
    parser.add_argument('--checkpoint', type=str, action='store', help='Path to checkpoint to load model weights from')
    parser.add_argument('--top_k', type=int, action='store', help='Number of top results to rerank')
    parser.add_argument('--model_name_or_path', type=str, action='store', required=True,
                        help='Path to pretrained LM to be used')
    parser.add_argument('--out_dir', type=str, action='store', required=True, help='Provide path to directory to store outputs')
    parser.add_argument('--ids2keep_file', type=str, action='store', default=None,
                        help='file for ehr ids to keep (e.g. for pmv)')
    parser.add_argument('--batch_size', type=int, action='store', default=20, help='Specify batch size')
    parser.add_argument('--seed', type=int, action='store', default=42, help='Specify random seed')
    parser.add_argument('--mod', type=int, action='store', default=None)


    args = parser.parse_args()
    rerank(**vars(args))
    print("Done.")
