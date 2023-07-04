import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math
from collections import Counter
import argparse
from scipy.sparse import coo_matrix, coo_array, vstack as sparse_vstack
import time
import psutil
import gc



def sparse_rank(lit_mentions_file, ehr_mentions_file, outcome, outpath):
    GB = 1024**3

    # Loading MeSH terms for EHRs for patient cohort
    print("reading EHR mentions file...")
    mention_info = pickle.load(open(ehr_mentions_file, 'rb'))
    print(f"{len(mention_info)} EHRs")

    # Loading MeSH terms for documents from outcome-specific literature collection
    print("reading literature mentions file...")
    doc_tags_info = pickle.load(open(lit_mentions_file, 'rb'))
    print(f"{len(doc_tags_info)} abstracts")

    #mentions_info = {key: mention_info[key] for key in list(mention_info.keys())[:1000]}
    #doc_tags_info = {key: doc_tags_info[key] for key in list(doc_tags_info.keys())[:1000]}

    # Note that the cohort for PMV prediction is smaller than other outcomes
    # So we need to filter out patients for whom PMV information is not available
    ids2keep = pickle.load(open('../data/pmv_ids.pkl', 'rb')) if outcome == 'pmv' else None

    # Reformat EHR MeSH term data
    print("preprocessing EHR mentions...")
    ehr_tags = {}
    for file in mention_info:
        if ids2keep is not None and file not in ids2keep:
            continue
        ehr_mesh_terms = []
        for sent in mention_info[file]:
            for mention in mention_info[file][sent]:
                if 'mesh_ids' not in mention:
                    continue
                for pair in mention['mesh_ids']:
                    ehr_mesh_terms.append(pair[0])
        ehr_tags[file] = ehr_mesh_terms

    # Reformat literature MeSH term data
    print("preprocessing literature mentions...")
    doc_tags = {}
    for file in doc_tags_info:
        """
        if ids2keep is not None and file not in ids2keep:
            continue
        """
        doc_mesh_terms = []
        for sent in doc_tags_info[file]:
            for mention in doc_tags_info[file][sent]:
                if 'mesh_ids' not in mention:
                    continue
                for pair in mention['mesh_ids']:
                    doc_mesh_terms.append(pair[0])
        doc_tags[file] = doc_mesh_terms

    doc_tags_unique = set([x for y in doc_tags.values() for x in y])
    ehr_tags_unique = set([x for y in ehr_tags.values() for x in y])

    # Compute vocabulary of MeSH terms for TF-IDF vector building
    mesh_vocab = doc_tags_unique & ehr_tags_unique
    print('MeSH vocabulary size: {}'.format(len(mesh_vocab)))
    mesh_vocab = dict(list(zip(list(mesh_vocab), range(len(mesh_vocab)))))

    # Construct TF-IDF vectors for both outcome-specific literature and EHRs
    doc_freq = Counter()
    ehr_vectors_sparse = {}
    article_vectors_sparse = {}

    # Term frequency computation
    # saving in sparse matrix type,
    # which has same API as numpy array but saves- and computes on- actually the non-zeros only
    # thus saving a lot of memory and compute

    print("computing TF for EHR files...")
    for file in ehr_tags:
        term_list = [x for x in ehr_tags[file] if x in mesh_vocab]
        doc_freq.update(set(term_list))
        unique_values, counts = np.unique(term_list, return_counts=True)
        indices = [mesh_vocab[x] for x in unique_values]
        cur_vec_sparse = coo_matrix((counts, (np.zeros_like(indices), indices)), shape=(1, len(mesh_vocab)))
        ehr_vectors_sparse[file] = cur_vec_sparse

    print("computing TF for literature files...")
    for file in doc_tags:
        term_list = [x for x in doc_tags[file] if x in mesh_vocab]
        doc_freq.update(set(term_list))
        unique_values, counts = np.unique(term_list, return_counts=True)
        indices = [mesh_vocab[x] for x in unique_values]
        cur_vec_sparse = coo_matrix((counts, (np.zeros_like(indices), indices)), shape=(1, len(mesh_vocab)))
        article_vectors_sparse[file] = cur_vec_sparse

    print("computing IDF...")
    num_docs = len(doc_tags) + len(ehr_tags)
    inverse_doc_freq = {k: math.log(num_docs / float(v)) for k, v in doc_freq.items()}
    inverse_doc_freq_vector = [1] * len(mesh_vocab)
    for x in mesh_vocab:
        inverse_doc_freq_vector[mesh_vocab[x]] = inverse_doc_freq[x]
    inverse_doc_freq_vector_sparse = coo_array(inverse_doc_freq_vector)


    # Construct TF-IDF vector matrices for both literature and outcomes
    # This helps speed up cosine similarity computation

    print("constructing TF-IDF vectors for EHR files ...")
    ehr_items_sparse = list(ehr_vectors_sparse.items())
    ehr_ids, ehr_matrix_sparse = [x[0] for x in ehr_items_sparse], [x[1] for x in ehr_items_sparse]
    ehr_matrix_sparse = sparse_vstack(ehr_matrix_sparse)
    ehr_matrix_sparse *= inverse_doc_freq_vector_sparse

    print("constructing TF-IDF vectors for literature files ...")
    article_items_sparse = list(article_vectors_sparse.items())
    article_ids, article_matrix_sparse = [x[0] for x in article_items_sparse], [x[1] for x in article_items_sparse]
    article_matrix_sparse = sparse_vstack(article_matrix_sparse)
    article_matrix_sparse *= inverse_doc_freq_vector_sparse

    # Computing cosine similarities and identifying top ranked documents
    keep_var_names = ["ehr_ids", "ehr_matrix_sparse", "article_ids", "article_matrix_sparse", "outpath",
                      "locals_dict", "local_var_names", "var_name", "keep_var_names"]



    num_unreferenced_but_not_freed = gc.collect()


    ranked_pairs = {}
    available_bytes = psutil.virtual_memory().available
    print(f"available before: {available_bytes}")
    print(f"available before: {available_bytes/GB} GB")
    num_articles = len(article_ids)
    row_size_in_bytes = num_articles * np.dtype("float64").itemsize
    num_rows_fitting_in_memory = available_bytes // (6 * row_size_in_bytes)
    needed_available = int(2 * row_size_in_bytes * num_rows_fitting_in_memory)
    print(f"needed available: {needed_available/GB}")
    if ehr_matrix_sparse.shape[0] < num_rows_fitting_in_memory:
        print("computing similarities...")
        similarities = cosine_similarity(ehr_matrix_sparse, article_matrix_sparse)
        print("ranking...")
        print("argsort...")
        top_indices = np.argsort(similarities)[:, -1:-1001:-1]
        print("taking along axis...")
        top_similarities = np.take_along_axis(similarities, top_indices, axis=-1)
        del similarities
        print("top pairs...")
        top_pairs = np.stack((top_indices, top_similarities), axis=2).tolist()
        print("ranked pairs...")
        for i, file in enumerate(ehr_ids):
            ranked_pairs[file] = [(article_ids[int(x[0])], x[1]) for x in top_pairs[i]]
    else:
        for start in range(0, ehr_matrix_sparse.shape[0], num_rows_fitting_in_memory):
            print(f"waiting for free memory...")
            i=0
            while psutil.virtual_memory().available < needed_available:
                time.sleep(1)
                i += 1
            print(f"waited {i} secs")
            end = min(start + num_rows_fitting_in_memory, ehr_matrix_sparse.shape[0])
            print(f"Computing similarities for EHRs {start}-{end}")
            cur_ehr_matrix = ehr_matrix_sparse[start:end, :]
            cur_similarities = cosine_similarity(cur_ehr_matrix, article_matrix_sparse)
            print("ranking...")
            print("argsort...")
            top_indices = np.argsort(cur_similarities)[:, -1:-1001:-1]
            print("taking along axis...")
            top_similarities = np.take_along_axis(cur_similarities, top_indices, axis=-1)
            top_pairs = np.stack((top_indices, top_similarities), axis=2).tolist()
            cur_ehr_ids = ehr_ids[start:end]
            for i, file in enumerate(cur_ehr_ids):
                ranked_pairs[file] = [(article_ids[int(x[0])], x[1]) for x in top_pairs[i]]
            print(f"before deleting have {psutil.virtual_memory().available/GB}")
            del cur_similarities
            del top_similarities
            gc.collect()
            print(f"after deleting have {psutil.virtual_memory().available / GB}")


    # Store ranked results from sparse retriever
    print("dumping...")
    pickle.dump(ranked_pairs, open(outpath, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lit_mentions_file', type=str, action='store', required=True,
                        help='Provide path to pkl file containing outcome-specific literature linked mentions')
    parser.add_argument('--ehr_mentions_file', type=str, action='store', required=True,
                        help='Provide path to pkl file containing ehr linked mentions')
    parser.add_argument('--outcome', type=str, action='store', required=True,
                        help='name of the outcome')
    parser.add_argument('--outpath', type=str, action='store', required=True,
                        help='path for out file')

    args = parser.parse_args()
    start = time.time()
    sparse_rank(**vars(args))
    end = time.time()
    print(f"took {end-start} secs")


