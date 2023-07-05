import csv
import os
import pickle
csv.field_size_limit(2147483647)
from collections import Counter

class EHRDataset:

    def __init__(self, train_path, dev_path, test_path, do_train=True, do_test=True):
        assert do_train or do_test, "if no train and no test, which data should it loads?"
        self.train_data = self.read_csv(train_path)
        self.dev_data = self.read_csv(dev_path)
        self.test_data = self.read_csv(test_path)
        self.do_train = do_train
        self.do_test = do_test

    def read_csv(self, path):
        reader = csv.reader(open(path))
        data = {}
        next(reader, None)
        for row in reader:
            data[row[0]] = {'ehr': row[1], 'outcome': int(row[2])}
        len_data = len(data)
        ten_percent = int(0.1*len_data) 
        #data = {x[0]: x[1] for x in list(data.items())[:ten_percent]} for debug purposes (less data)
        return data

    def compute_class_weights(self):
        class_counts = Counter()
        data = self.train_data
        for example in data:
            class_counts.update([data[example]['outcome']])
        num_samples = sum(list(class_counts.values()))
        num_classes = len(class_counts)
        balance_coeff = float(num_samples)/num_classes
        self.class_weights = {k:balance_coeff/float(v) for k,v in class_counts.items()}

    def add_relevant_literature(self, lit_dir, topk, lit_file):
        all_texts = pickle.load(open(lit_file, 'rb'))

        rankings_files = os.listdir(lit_dir)
        num_rankings_files = len(rankings_files)
        missing_lit_counter = 0

        ehrs_to_process = set()
        if self.do_train:
            ehrs_to_process = ehrs_to_process | set(self.train_data.keys()) | set(self.dev_data.keys())
        if self.do_test:
            ehrs_to_process = ehrs_to_process | set(self.test_data.keys())

        all_ehrs = set(self.train_data.keys()) | set(self.dev_data.keys()) | set(self.test_data.keys())

        for i, file in enumerate(rankings_files):
            id = file.split('.pkl')[0]
            if id not in all_ehrs:
                print(f"id {id} not in train/dev/test datasets")
            if id not in ehrs_to_process:
                continue

            docs = pickle.load(open(os.path.join(lit_dir, file), 'rb'))
            if isinstance(docs, dict) and len(docs.keys()) == 1:
                docs = docs[id]
            docs = list(reversed(sorted(docs, key=lambda x:x[1])))
            docs_nums = [x[0] for x in docs]
            not_found_docs = set(docs_nums) - set(all_texts)
            num_not_found_docs = len(not_found_docs)
            if num_not_found_docs > 0:
                print(f"not found: {num_not_found_docs}")

            chosen_docs = docs[:int(topk)] if topk >= 1 else [x for x in docs if x[1] >= topk]
            try:
                chosen_docs = [[x[0], all_texts[x[0]]['text'], x[1]] for x in chosen_docs]  # We may want to include year later?
            except:
                missing_lit_counter += 1
            if id in self.train_data:
                self.train_data[id]['pubmed_docs'] = chosen_docs
            elif id in self.dev_data:
                self.dev_data[id]['pubmed_docs'] = chosen_docs
            elif id in self.test_data:
                self.test_data[id]['pubmed_docs'] = chosen_docs

            print(f"added docs to {i + 1}/{len(rankings_files)} ehr files", end="\r", flush=True)


    def add_literature_matrices(self, lit_embed_file):
        lit_embeds = pickle.load(open(lit_embed_file, 'rb'))

        if self.do_train:
            for id in self.train_data:
                self.train_data[id]['pubmed_doc_embeds'] = {x[0]:lit_embeds[x[0]] for x in self.train_data[id]['pubmed_docs']}
            for id in self.dev_data:
                self.dev_data[id]['pubmed_doc_embeds'] = {x[0]:lit_embeds[x[0]] for x in self.dev_data[id]['pubmed_docs']}

        if self.do_test:
            for id in self.test_data:
                self.test_data[id]['pubmed_doc_embeds'] = {x[0]:lit_embeds[x[0]] for x in self.test_data[id]['pubmed_docs']}
