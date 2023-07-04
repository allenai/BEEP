import os.path
import pickle
import argparse
import numpy as np
import os


def take_top_n_and_untie(input_file_path, ranking_type, out_dir, top_n):
    os.makedirs(out_dir, exist_ok=True)

    if ranking_type != "similarity":
        print("note that the similarity score are some decreasing function of the distances,"
              "and may not represent the 'correct' similarity score")

    data = pickle.load(open(input_file_path, "rb"))
    counter = 0
    out_of = len(data)
    if ranking_type == "similarity":
        for ehr in data.keys():
            num_files = min(top_n, len(data[ehr]))
            top_n_docs_with_similarity_score = sorted(data[ehr], key=lambda x: x[1])[-num_files:][::-1]
            out_path = os.path.join(out_dir, f"{ehr}.pkl")
            pickle.dump(top_n_docs_with_similarity_score, open(out_path, "wb"))
            counter += 1
            print(f"processed {counter}/{out_of} files", end="\r", flush=True)


    elif ranking_type == "distance":
        all_dists_min = float("inf")
        all_dists_max = -float("inf")
        all_dists_counts = 0
        all_dists_sum = 0
        all_dists_squared_sum = 0
        for ehr in data.keys():
            distances = np.array([x[1] for x in data[ehr]])
            all_dists_min = min(all_dists_min, min(distances))
            all_dists_max = max(all_dists_max, max(distances))
            all_dists_sum += sum(distances)
            all_dists_squared_sum += sum(distances**2)
            all_dists_counts += len(distances)
        dists_mean = all_dists_sum / all_dists_counts
        dists_var = all_dists_squared_sum / all_dists_counts - dists_mean**2
        dists_std = dists_var ** 0.5

        all_dists_range = all_dists_max - all_dists_min

        for ehr in data.keys():
            num_files = min(top_n, len(data[ehr]))
            top_n_docs_with_distance_score = sorted(data[ehr], key=lambda x: x[1])[:num_files]
            top_distances = np.array([x[1] for x in top_n_docs_with_distance_score])

            # mean std normalization
            #normalized_top_distances = (top_distances-distances_mean)/distances_std # minus sign to make them positive

            # min-max normalization: the least distance will become sim=1. other will become 0<sim<1 (0 for min distance)
            normalized_top_distances = 1 - (top_distances - all_dists_min) / all_dists_range
            assert np.array_equal(sorted(normalized_top_distances, reverse=True), normalized_top_distances)
            top_n_docs_with_similarity_score = [(x[0], y)
                                                for x,y in zip(top_n_docs_with_distance_score, normalized_top_distances)]
            out_path = os.path.join(out_dir, f"{ehr}.pkl")
            pickle.dump(top_n_docs_with_similarity_score, open(out_path, "wb"))
            counter += 1
            print(f"processed {counter}/{out_of} files", end="\r", flush=True)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", required=True, type=str)
    parser.add_argument("--ranking_type", required=True, type=str, choices=["similarity", "distance"])
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=1000, required=True)
    args = parser.parse_args()
    args = vars(args)
    take_top_n_and_untie(**args)


