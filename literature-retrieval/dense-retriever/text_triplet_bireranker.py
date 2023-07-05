'''
Code to run LM-based reranker over abstracts retrieved per query
Command: python text_reranker.py --retrieval_results <RETRIEVAL_PICKLE_FILE> --entities <ENTITY_PICKLE_FILE> --out_dir <OUT_DIR> --model_name_or_path <MODEL> --checkpoint <MODEL_CHECKPOINT>
'''

import argparse
import gc
import os
import pickle
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    set_seed,
)

from data_loader_bireranker import TextDataset

def dense_ranker(abstracts, ehr_entities, text_file, outcome, ids2keep,
                 checkpoint, model_name_or_path, censor_after_year, out_dir,
                 query_type, query_format,
                 top_k, seed, cutoff):

    if seed is not None:
        set_seed(seed)

    if outcome == "pmv" and ids2keep is None:
        raise(ValueError("for outcome 'pmv' ids2keep should be provided."))

    dataset = TextDataset(abstracts, ehr_entities)
    dataset.add_ehr_text(text_file, ids2keep)
    dataset.create_dataset(outcome, censor_after_year)
    dataset.add_entities(cutoff)

    print('Added entities to dataset...')

    label_vocab = {'Relevant': 1, 'Irrelevant': 0}

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=len(list(label_vocab.keys())),
        label2id=label_vocab,
        id2label={i: l for l, i in label_vocab.items()},
        # finetuning_task='mnli',
        cache_dir='../../cache',
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir='../../cache',
        use_fast=True,
    )

    model = AutoModel.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir='../../cache',
    )

    special_tokens_dict = {'additional_special_tokens': ['[ENTSEP]']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    model = model.cuda()
    if 'checkpoint' in checkpoint:
        full_checkpoint = torch.load(checkpoint)
        model.load_state_dict(full_checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(checkpoint))

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

    encoded_pubmed_articles = encode_texts(dataset.texts)
    print('Encoded all outcome-specific articles')
    del dataset.texts
    gc.collect()

    article_items = list(encoded_pubmed_articles.items())
    article_ids, article_matrix = [x[0] for x in article_items], [x[1] for x in article_items]
    article_matrix = np.vstack(article_matrix)

    outcome_questions = {'mortality': 'What is the hospital mortality?', # the probability of?
                         'pmv': 'What is the probability of prolonged mechanical ventilation?',
                         'los': 'What is the probable length of stay?'}

    query_question = outcome_questions[outcome]

    fnum = 0
    all_ehrs_ranked = {}
    for file in list(dataset.data.keys()):
        print('Ranking documents for query {} ({})'.format(file, fnum))
        fnum += 1

        if query_format == "text":
            query_data = dataset.query_texts[file]
        elif query_format == "entity":
            entities = dataset.data[file]["entities"]
            mesh_entities = entities["mesh"]
            non_mesh_entities = entities["non-mesh"]
            if query_type == "all":
                query_terms = mesh_entities + non_mesh_entities
            elif query_type == "mesh":
                query_terms = mesh_entities
            else:
                raise(ValueError(f"query type {query_type} is not recognized"))
            query_data = ' [ENTSEP] ' + ' [ENTSEP] '.join(query_terms)
        else:
            raise (ValueError(f"query format {query_format} is not recognized"))

        cur_query_text = query_question + ' ' + query_data  # TODO: why do we need question at all?

        cur_query_text = tokenizer(cur_query_text, padding='max_length', max_length=512, truncation=True,
                                   return_tensors="pt")
        cur_query_text = {x:y.cuda() for x,y in cur_query_text.items()}
        cur_query_embed = model(**cur_query_text)
        cur_query_embed = cur_query_embed['last_hidden_state'][0,0,:].detach().cpu().numpy().transpose()
        cur_query_embed = cur_query_embed.reshape(1,-1)
        distances = euclidean_distances(cur_query_embed, article_matrix).astype(np.float16).tolist()[0]
        ranked_docs = list(zip(article_ids, distances))
        ranked_docs = list(sorted(ranked_docs, key=lambda x:x[1]))
        all_ehrs_ranked[file] = ranked_docs[:top_k]
        out_path = os.path.join(out_dir, file + ".pkl")
        pickle.dump(ranked_docs[:top_k], open(out_path, 'wb'))

    split = ""
    if "train" in text_file:
        split = "train"
    elif "test" in text_file:
        split = "test"
    elif "dev" in text_file:
        split = "dev"
    else:
        print("not sure what split is it, cannot save overall file.")

    print("dumping all...")
    overall_outpath = os.path.join(out_dir, f"dense_ranked_{split}" + ".pkl")
    pickle.dump(all_ehrs_ranked, open(overall_outpath, 'wb'))
    print("Done.")









if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file', type=str, action='store', required=True,
                        help='Path to file containing raw texts from EHRs')
    parser.add_argument('--ehr_entities', type=str, action='store', required=True,
                        help='Path to file containing extracted entities for queries')
    parser.add_argument('--ids2keep', type=str, action='store', default=None,
                        help='Path to ehr ids to keep (if we take only subset)')
    parser.add_argument('--abstracts', type=str, action='store',
                        help='Path to pkl file containing id: abstract texts')
    parser.add_argument('--out_dir', type=str, action='store', required=True,
                        help='Provide path to output directory')
    parser.add_argument('--outcome', type=str, action='store', required=True, help='Target outcome to predict')
    parser.add_argument('--model_name_or_path', type=str, action='store', required=True,
                        help='Path to pretrained LM to be used')
    parser.add_argument('--checkpoint', type=str, action='store', help='Path to checkpoint to load model weights from')
    parser.add_argument('--top_k', type=int, action='store', help='Number of top results to rerank', default=1000)
    parser.add_argument('--censor_after_year', type=int, action='store', default=2016,
                        help='censor literature after this year')
    parser.add_argument('--cutoff', type=float, action='store', default=0.9,
                        help='cut entities with probability less than that value')
    parser.add_argument('--query_format', type=str, action='store', default='text',
                        help='Indicate how query should be framed (text/entity)')
    parser.add_argument('--query_type', type=str, action='store', default='all',
                        help='Indicate which entity types to include in query [mesh/all]')
    parser.add_argument('--seed', type=int, action='store', default=42, help='Specify random seed')

    args = parser.parse_args()
    arguments_dict = vars(args)
    dense_ranker(**arguments_dict)

