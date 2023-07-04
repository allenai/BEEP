import pickle
import argparse
import os

def merge(sparse_ranked_path, dense_ranked_path, out_path, top_n):
    sparse_ranked = pickle.load(open(sparse_ranked_path, "rb"))
    dense_ranked = pickle.load(open(dense_ranked_path, "rb"))
    sparse_keys = set(sparse_ranked.keys())
    dense_keys = set(dense_ranked.keys())
    common_keys = sparse_keys & dense_keys
    dense_not_sparse_keys = dense_keys - sparse_keys
    sparse_not_dense_keys = sparse_keys - dense_keys
    if len(dense_not_sparse_keys) > 0:
        print(f"{len(dense_not_sparse_keys)} EHRs have sparse ranking but not dense ranking")
    if len(sparse_not_dense_keys) > 0:
        print(f"{len(sparse_not_dense_keys)} EHRs have dense ranking but not sparse ranking")

    half_n = top_n // 2
    merged = {}
    for k, ehr in enumerate(common_keys):
        ehr_sparse_ranking = sparse_ranked[ehr] # by similarity. the bigger the closer
        top_half_n_sparse = sorted(ehr_sparse_ranking, key=lambda x: x[1])[-half_n:][::-1]
        ehr_dense_ranking = dense_ranked[ehr] # by distance. the smaller the closer
        top_half_n_dense = sorted(ehr_dense_ranking, key=lambda x: x[1])[:half_n]
        top_half_n_sparse_docs = [x[0] for x in top_half_n_sparse]
        top_half_n_dense_docs = [x[0] for x in top_half_n_dense]
        ehr_docs_combined = []
        existing = set()

        for i in range(half_n):
            sparse_ith = top_half_n_sparse_docs[i]
            dense_ith = top_half_n_dense_docs[i]
            rank = 1-i/half_n
            if sparse_ith not in existing:
                ehr_docs_combined.append((sparse_ith, rank))
                existing.add(sparse_ith)
            if dense_ith not in existing:
                ehr_docs_combined.append((dense_ith, rank))
                existing.add(dense_ith)

        merged[ehr] = ehr_docs_combined
        print(f"processed {k+1}/{len(common_keys)} docs", end="\r", flush=True)
    pickle.dump(merged, open(out_path, "wb"))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse_ranked_path", required=True, type=str,
                        help="path to sparse ranked (assuming the ranking are similairties, the bigger the closer")
    parser.add_argument("--dense_ranked_path", required=True, type=str,
                        help="path to dense ranked (assuming the ranking are distances, the less the closer")
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=1000, required=True)
    args = parser.parse_args()
    args = vars(args)
    merge(**args)



